import pandas as pd
import boto3
from io import StringIO
import awswrangler as wr

base_dir = "/opt/ml/processing"
bucket = "ideaaiml-demo"
prefix = "mlops/predictive-maintenance"

def upload_file_s3(df, name):
    boto3.setup_default_session(region_name = "us-east-1")
    s3_client = boto3.client("s3", region_name = "us-east-1")
    with StringIO() as csv_buffer:
        df.to_csv(csv_buffer, index = False)

        response = s3_client.put_object(
            Bucket = bucket, Key = f"{prefix}/data/train-test/{name}.csv", Body = csv_buffer.getvalue()
        )
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}")
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}")

def train_test_split_script(labeled_features):
    threshold_dates = [[pd.to_datetime('2015-07-31 01:00:00'), pd.to_datetime('2015-08-01 01:00:00')],
                   [pd.to_datetime('2015-08-31 01:00:00'), pd.to_datetime('2015-09-01 01:00:00')],
                   [pd.to_datetime('2015-09-30 01:00:00'), pd.to_datetime('2015-10-01 01:00:00')]]
    
    for last_train_date, first_test_date in threshold_dates:
        # split out training and test data
        print(labeled_features['datetime'][0])
        train_y = labeled_features.loc[labeled_features['datetime'] < last_train_date, 'failure']
        train_data = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] < last_train_date].drop(['datetime',
                                                                                                            'machineID',
                                                                                                              'failure'], axis = 1))
        test_y = labeled_features.loc[labeled_features['datetime'] > last_train_date, 'failure']
        test_data = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] > first_test_date].drop(['datetime',
                                                                                                           'machineID',
                                                                                                             'failure'], axis = 1))
    
    train_data['failure'] = train_y
    test_data['failure'] = test_y
    
    upload_file_s3(train_data, "train")
    upload_file_s3(test_data, "test")

    pd.DataFrame(train_data).to_csv(f"{base_dir}/train/train.csv", index = False)
    pd.DataFrame(test_data).to_csv(f"{base_dir}/test/test.csv", index = False)
    
if __name__ == "__main__":
    final_data_uri = f"s3://{bucket}/{prefix}/data/preprocessed/preprocessed.csv"
    final_data = wr.s3.read_csv(final_data_uri)
    final_data['datetime'] = pd.to_datetime(final_data['datetime'], format="%Y-%m-%d %H:%M:%S")
    train_test_split_script(final_data)
