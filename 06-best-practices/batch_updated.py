#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os
from datetime import datetime

INPUT_FILE_PATTERN  = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
OUTPUT_FILE_PATTERN = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
S3_ENDPOINT_URL     = os.getenv('S3_ENDPOINT_URL')

def read_data(filename):
    if S3_ENDPOINT_URL:
        storage_options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }
        df = pd.read_parquet(filename, storage_options=storage_options)
    else:
        df = pd.read_parquet(filename)
    return df

def prepare_data(df):
    df = df.copy()
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
    df['PULocationID'] = df['PULocationID'].fillna(-1).astype(int)
    df['DOLocationID'] = df['DOLocationID'].fillna(-1).astype(int)
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    return df

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = INPUT_FILE_PATTERN or default_input_pattern
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = 'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_pattern = OUTPUT_FILE_PATTERN or default_output_pattern
    return output_pattern.format(year=year, month=month)

def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(input_file)
    df = prepare_data(df)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('Predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
