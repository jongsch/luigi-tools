import abc
import gc
import json
import os
import random
import subprocess
import tempfile

import boto3
import pathlib

import tensorflow as tf
import zipfile
from tensorflow.keras import Model
import luigi
import pandas as pd
from luigi.contrib.s3 import S3Target
from luigi.task import flatten

from luigi_tools.db import query_athena, write_csv, read_parquet, read_df, write_df
from luigi_tools.aws import format_s3_path, S3_BUCKET

DEFAULT_VERSION = '3'


def run_luigi_job_with_subprocess(_file, task_name, task_args=[]):
    # when calling, pass in __file__ for _file arg
    module_name = os.path.splitext(os.path.basename(_file))[0]
    # use subprocess to ensure no ta2 settings confusion
    command_args = ['luigi',
                    '--module',
                    module_name,
                    task_name,
                    ]
    if task_args:
        command_args.extend(task_args)
    command_args.extend(['--workers', '1'])
    print(command_args)
    subprocess.check_call(command_args,
                          # assume all modules containing luigi Tasks are in the same directory as this file
                          cwd=os.path.dirname(os.path.abspath(__file__)))


class DefaultLuigiParametersMixin:
    version = luigi.Parameter(default=DEFAULT_VERSION)
    production_mode = luigi.BoolParameter(default=False, significant=False,
                                          parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    def default_parameters(self):
        return {'version': self.version, 'production_mode': self.production_mode}


class S3DataTask(luigi.Task, DefaultLuigiParametersMixin):
    def output(self):
        return S3Target(format_s3_path(self.key()))

    @abc.abstractmethod
    def key(self):
        pass


class QueryAthenaTask(S3DataTask):
    # def client(self):
    #     return S3Client()

    def run(self):
        # ideally: query s3, move to this bucket. data never leaves s3, more efficient.
        # problem is, pyathena's .as_pandas() does some conversion
        # https://github.com/laughingman7743/PyAthena/blob/bcdb753fe98528aeaa53facfff4cccd8c1327504/pyathena/pandas/result_set.py#L164-L178
        # without that, downstream tasks fail.
        # so (for now at least), pull it down, format, push it back up

        # b, k = query_athena_to_s3(self.query())
        # self.client().move(format_s3_path(k, bucket=b), self.output().path)

        df = query_athena(self.query())
        with self.output().open('w') as out:
            # write to the tmp path - it will automatically upload to s3 upon closing the file context
            output_storage_format = pathlib.Path(self.output().path).suffix
            # be sure to specify format since it will not be correctly inferred from out.tmp_path
            write_df(df, out.tmp_path, format=output_storage_format)

    @abc.abstractmethod
    def query(self):
        pass


class DataFrameTransformTask(S3DataTask):
    # subclasses can override this to 'parquet'

    @abc.abstractmethod
    def transform_df(self, *dfs, **kwargs):
        pass

    def transform_kwargs(self):
        return {}

    @property
    def input_df(self):
        df_input_path = flatten(self.input())[0].path
        df = read_df(df_input_path, csv_converters=self.csv_converters())
        return df

    @property
    def output_df(self):
        return self.transform_df(self.input_df, **self.transform_kwargs())

    def run(self):
        with self.output().open('w') as out:
            # write to the tmp path - it will automatically upload to s3 upon closing the file context
            output_storage_format = pathlib.Path(self.output().path).suffix
            # be sure to specify format since it will not be correctly inferred from out.tmp_path
            write_df(self.output_df, out.tmp_path, format=output_storage_format)

    def csv_converters(self):
        return None


class MultiInputDataFrameTransformTask(DataFrameTransformTask):
    # this can probably be merged into DataFrameTransformTask
    @property
    def input_dfs(self):
        dfs = []
        for i in self.input():
            df = read_df(i.path, csv_converters=self.csv_converters())
            dfs.append(df)
        return dfs

    @property
    def output_df(self):
        return self.transform_df(self.input_dfs)


class DataFrameReduceTask(MultiInputDataFrameTransformTask):
    @property
    def input_df(self):
        return pd.concat(
            self.input_dfs,
            # append all the rows together
            axis=0
        )

    @property
    def output_df(self):
        return self.transform_df(self.input_df)


class CreateModelTask(S3DataTask):
    def run(self):
        model = self.create_model()
        model_json = model.to_json()
        with self.output().open('w') as f:
            json.dump(json.loads(model_json),
                      f,
                      sort_keys=True,
                      indent=4,
                      ensure_ascii=False)

    @abc.abstractmethod
    def create_model(self) -> tf.keras.Model:
        pass


class ModelTask(S3DataTask):
    def __init__(self, custom_keras_objects=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if custom_keras_objects is None:
            custom_keras_objects = {}
        self.custom_keras_objects = custom_keras_objects

    def initialize_model(self) -> Model:
        tf.keras.backend.clear_session()
        gc.collect()
        # return Model(...)
        with self.requires_model().output().open('r') as f:
            model_json = f.read()
        model = tf.keras.models.model_from_json(model_json, custom_objects=self.custom_keras_objects)
        if self.requires_pretraining():
            model = self.load_pretrained_weights(model)
        return model

    def load_pretrained_weights(self, model) -> Model:
        """
        Override this to load pretrained weights for subcomponents of the model.

        Create one or more new Model(...) instance(s) from the relevant model components,
        with architecture(s) matching the saved weights.
        The weight file(s) may be obtained via the task(s) in self.requires_pretraining().

        Additionally, set trainable = False here for layers as needed.
        """
        return model

    def load_data(self):
        """
        Assumes one or more csv inputs to be loaded as dataframes.
        Override this for more complicated data dependencies.
        """
        data = []
        for r in flatten(self.requires_data()):
            # consider making the read function (read_parquet or read_csv) a property
            # or method of the required task r
            data.append(read_parquet(r.output().path))
        return data

    @abc.abstractmethod
    def requires_data(self):
        """The task(s) outputting the training data"""
        pass

    @abc.abstractmethod
    def requires_model(self):
        """The task that creates the model definition json"""
        pass

    def requires_pretraining(self):
        """Any tasks that pre-train a subset of the model's weights"""
        return []

    def requires(self):
        return flatten(self.requires_model()) \
               + flatten(self.requires_data()) \
               + flatten(self.requires_pretraining())


class TrainModelTask(ModelTask):
    @abc.abstractmethod
    def train(self, model, data) -> Model:
        # model.compile(....)
        # model.fit(...)
        pass

    def run(self):
        model = self.initialize_model()
        data = self.load_data()
        model = self.train(model, data)
        with tempfile.TemporaryDirectory() as tmpdir:
            filename_base = self.key().strip('.zip').split('/')[-1]
            local_path = os.path.join(tmpdir, filename_base)
            model.save_weights(local_path, save_format='tf')
            print(os.listdir(tmpdir))
            with zipfile.ZipFile(f'{tmpdir}.zip', 'w') as z:
                for f in os.listdir(tmpdir):
                    z.write(os.path.join(tmpdir, f), f)
            # originally, tried to write the contents of z to self.output().
            # however, this did not work because luigi S3Target can't be opened in binary mode.
            # fortunately, upload should be atomic, so same difference.
        s3 = boto3.client('s3')
        s3.upload_file(f'{tmpdir}.zip', S3_BUCKET, self.key())


def load_model_weights_from_training_task(model: Model, training_task: TrainModelTask):
    with tempfile.NamedTemporaryFile('r', suffix='.zip') as tmp:
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET, training_task.key(), tmp.name)
        tmpdir = f'/tmp/luigi-{random.randint(10000000, 99999999)}'
        with zipfile.ZipFile(tmp.name) as z:
            z.extractall(tmpdir)
            filename_base = training_task.key().strip('.zip').split('/')[-1]
            local_path = os.path.join(tmpdir, filename_base)
            model.load_weights(local_path)


class ModelPredictionTask(ModelTask):
    @abc.abstractmethod
    def predict(self, model, data):
        pass

    @abc.abstractmethod
    def requires_pretraining(self) -> TrainModelTask:
        pass

    def load_pretrained_weights(self, model) -> Model:
        load_model_weights_from_training_task(model, self.requires_pretraining())
        return model

    def run(self):
        model = self.initialize_model()
        data = self.load_data()
        predictions = self.predict(model, data)
        predictions = pd.DataFrame(predictions)
        with self.output().open('w') as out:
            write_csv(predictions, out.tmp_path)

