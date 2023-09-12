import os

import boto3
import pathlib

import joblib
import tensorflow as tf
import zipfile


# fill in with appropriate bucket
S3_BUCKET = ''


def format_s3_path(key, bucket=S3_BUCKET):
    return f's3://{bucket}/{key}'


def save_model_to_s3(model, model_filename):
    pathlib.Path('/tmp').mkdir(parents=True, exist_ok=True)
    local_path = os.path.join('/tmp/', model_filename)
    model.save(local_path)
    s3_key = 'models/' + model_filename
    s3 = boto3.client('s3')
    s3.upload_file(local_path, S3_BUCKET, s3_key)


def load_model_from_s3(model_filename):
    pathlib.Path('/tmp').mkdir(parents=True, exist_ok=True)
    local_path = os.path.join('/tmp/', model_filename)
    s3_key = 'models/' + model_filename
    s3 = boto3.client('s3')
    s3.download_file(S3_BUCKET, s3_key, local_path)
    model = tf.keras.models.load_model(local_path)
    return model


def save_model_weights_to_s3(model, filename):
    # model.save_weights writes multiple files, .index and .data-00000-of-00001, etc
    # so zip them up then upload to s3
    tmp_dir = '/tmp'
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    local_path = os.path.join(tmp_dir, filename)
    model.save_weights(local_path)
    with zipfile.ZipFile(f'{local_path}.zip', 'w') as z:
        [z.write(os.path.join(tmp_dir, f), f) for f in os.listdir(tmp_dir) if filename in f and '.zip' not in f]
    s3_key = 'models/' + filename + '.zip'
    s3 = boto3.client('s3')
    s3.upload_file(f'{local_path}.zip', S3_BUCKET, s3_key)


def load_model_weights_from_s3(model, filename):
    tmp_dir = '/tmp'
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    local_path = os.path.join(tmp_dir, filename)
    s3_key = 'models/' + filename + '.zip'
    s3 = boto3.client('s3')
    s3.download_file(S3_BUCKET, s3_key, f'{local_path}.zip')
    with zipfile.ZipFile(f'{local_path}.zip', 'r') as z:
        z.extractall(tmp_dir)
    model.load_weights(local_path)


def save_pickle_to_s3(obj, s3_key):
    pathlib.Path('/tmp').mkdir(parents=True, exist_ok=True)
    local_path = os.path.join('/tmp', s3_key)
    with open(local_path, 'wb') as f:
        joblib.dump(obj, f)
    # for now, put everything in models dir
    s3 = boto3.client('s3')
    s3.upload_file(local_path, S3_BUCKET, s3_key)


def load_pickle_from_s3(s3_key):
    local_path = os.path.join('/tmp', s3_key)
    pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client('s3')
    print("downloading file\n\n")
    s3.download_file(S3_BUCKET, s3_key, local_path)
    print("file downloaded\n\n")
    with open(local_path, 'rb') as f:
        obj = joblib.load(f)
    return obj

