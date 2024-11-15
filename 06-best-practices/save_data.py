#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd

def save_data(filename, categorical):
    storage_options = {
        'key': 'test',
        'secret': 'test',
        'client_kwargs': {
            'endpoint_url': 'http://localhost:4566'
        }
    }

    df = pd.read_parquet(filename, storage_options=storage_options)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    input_file = f's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'
    output_file = f's3://nyc-duration/out/{year:04d}-{month:02d}-save_data.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = save_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
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
