from datetime import datetime
import pandas as pd
from pandas.testing import assert_frame_equal

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def prepare_data(df):
    df = df.copy()
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
    df['PULocationID'] = df['PULocationID'].fillna(-1).astype(int)
    df['DOLocationID'] = df['DOLocationID'].fillna(-1).astype(int)
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    return df

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df = pd.DataFrame(data, columns=columns)

df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

expected_data = [
    (-1, -1, dt(1, 1), dt(1, 10), 9.0, '-1_-1'),
    (1, 1, dt(1, 2), dt(1, 10), 8.0, '1_1'),
]
expected_columns = [
    'PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 
    'tpep_dropoff_datetime', 'duration', 'PU_DO'
]
df_expected = pd.DataFrame(expected_data, columns=expected_columns)
df_actual = prepare_data(df)
df_actual = df_actual.reset_index(drop=True)
df_expected = df_expected.reset_index(drop=True)
assert_frame_equal(df_actual, df_expected)

print(f"How many rows should be there in the expected dataframe?\n{len(df_expected)}")
