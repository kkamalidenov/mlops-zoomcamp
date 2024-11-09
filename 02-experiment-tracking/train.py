import os
import pickle
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("random-forest-train")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

mlflow.sklearn.autolog()

data_path = "./output"
X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

with mlflow.start_run() as run:
    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"RMSE: {rmse}")

run_id = run.info.run_id
client = mlflow.tracking.MlflowClient()
run_data = client.get_run(run_id).data
params = run_data.params

print("\nLogged Model Parameters:")
for param_name, param_value in params.items():
    print(f"{param_name}: {param_value}")

print(f"\nThe value of min_samples_split is: {params.get('min_samples_split')}")
