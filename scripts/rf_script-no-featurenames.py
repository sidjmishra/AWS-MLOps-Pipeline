import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == "__main__":

    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train-data", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test-data", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="datasets/train-test/train.csv")
    parser.add_argument("--test-file", type=str, default="datasets/train-test/test.csv")
#     parser.add_argument(
#         "--features", type=str
#     )  # in this script we ask user to explicitly name features
#     parser.add_argument(
#         "--target", type=str
#     )  # in this script we ask user to explicitly name the target

    args = parser.parse_args()
    print(args.train_data)
    print(args.train_file)

    print("reading data")
    train_df = pd.read_csv(f"{args.train_data}/train.csv")
    test_df = pd.read_csv(f"{args.test_data}/test.csv")

    print("building training and testing datasets")
#     X_train = train_df[args.features.split()]
#     X_test = test_df[args.features.split()]
#     y_train = train_df[args.target]
#     y_test = test_df[args.target]
    
    X_train = train_df.drop("failure", axis = 1)
    X_test = test_df.drop("failure", axis = 1)
    y_train = train_df["failure"]
    y_test = test_df["failure"]

    # train
    print("training model")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf, n_jobs=-1
    )

    model.fit(X_train.values, y_train)
    predictions = model.predict(X_test.values)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy Score: {accuracy}")

    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("Model persisted at " + path)
    print(args.min_samples_leaf)