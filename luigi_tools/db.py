import json
import os

import boto3
import pandas as pd
import pathlib
import pyathena
from pyathena.pandas.cursor import PandasCursor
from pyathena.converter import DefaultTypeConverter
from pyathena.error import ProgrammingError
import pyathena.util

from .aws import S3_BUCKET


# fill in with bucket used by athena
S3_STAGING_BUCKET = ''


class CustomPandasTypeConverter(DefaultTypeConverter):
    def __init__(self):
        super(CustomPandasTypeConverter, self).__init__()
        self._mappings.update({'json': self._to_json})

    def _to_json(self, varchar_value):
        if varchar_value is None or varchar_value == '':
            return None
        return json.loads(varchar_value)


def get_athena_cursor(s3_staging_bucket=S3_STAGING_BUCKET):
    with pyathena.connect(
            s3_staging_dir=f's3://{s3_staging_bucket}/',
            cursor_class=PandasCursor,
            converter=CustomPandasTypeConverter(),
    ) as athena_conn:
        return athena_conn.cursor()


def query_athena(query):
    with get_athena_cursor() as athena_cursor:
        df = athena_cursor.execute(query).as_pandas()
        return df


def query_athena_to_s3(query):
    with get_athena_cursor() as athena_cursor:
        athena_cursor.execute(query)
        result_set = athena_cursor.result_set
        if not result_set.output_location:
            raise ProgrammingError("OutputLocation is none or empty.")
        bucket, key = pyathena.util.parse_output_location(result_set.output_location)
    return bucket, key


def download_from_s3(key, output_file):
    if os.path.exists(output_file):
        raise FileExistsError(f"Output file {output_file} already exists.")
    s3_client = boto3.client('s3')
    s3_client.download_file(S3_STAGING_BUCKET, key, output_file)


def query_athena_to_csv(query, output_file):
    # Query athena, get location of output csv, download it.
    if os.path.exists(output_file):
        raise FileExistsError(f"Output file {output_file} already exists.")
    bucket, key = query_athena_to_s3(query)
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, key, output_file)


def read_csv(source, csv_converters=None):
    return pd.read_csv(
        source,
        converters=csv_converters,
        keep_default_na=False,
    )


def write_csv(df, dest):
    df.to_csv(dest, na_rep='', index=False)


def read_parquet(source, engine='fastparquet'):
    return pd.read_parquet(source, engine=engine)


def write_parquet(df, dest, engine='fastparquet'):
    df.to_parquet(path=dest, engine=engine)


def read_df(path, csv_converters=None, format=None):
    if not format:
        format = pathlib.Path(path).suffix
    print(f"reading {format}")
    try:
        if format == '.csv':
            df = read_csv(path, csv_converters)
        elif format == '.parquet':
            df = read_parquet(path)
        else:
            raise Exception(f"Invalid input storage format: {format}")
    except pd.errors.EmptyDataError:
        df = pd.DataFrame([])
    print(f"{format} loaded")
    return df


def write_df(df, path, format=None):
    if not format:
        format = pathlib.Path(path).suffix
    if format == '.csv':
        write_csv(df, path)
    elif format == '.parquet':
        write_parquet(df, path, engine='fastparquet')
    else:
        raise Exception(f"Invalid output storage format: {format}")


def read_df_from_s3(key, bucket=S3_BUCKET, csv_converters=None):
    # requires s3fs package
    # df = pd.read_csv(f's3://{S3_STAGING_BUCKET}/{key}')
    source = f's3://{bucket}/{key}'
    df = read_df(source, csv_converters)
    return df


def write_df_to_s3(df, key, bucket=S3_BUCKET, format=None):
    dest = f's3://{bucket}/{key}'
    if not format:
        format = pathlib.Path(dest).suffix
    if format == '.parquet':
        write_parquet(df, dest)
    elif format == '.csv':
        write_csv(df, dest)
