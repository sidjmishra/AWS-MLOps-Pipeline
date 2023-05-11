import numpy as np
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
            Bucket = bucket, Key = f"{prefix}/data/preprocessed/{name}.csv", Body = csv_buffer.getvalue()
        )
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}")
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}")

# Convert to datetime datatype
def datetime_datatype(df):
    print("Converting to type datetime")
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")
    return df


# Convert to category datatype
def category_datatype(df, column_name):
    print("Converting to type category")
    df[column_name] = df[column_name].astype('category')
    return df


# Lag Features from Telemetry
def telemetry_features(df):
    df = datetime_datatype(df)
    # Calculate mean values for telemetry features -- 3 hours rolling window
    print("Calculate mean values for telemetry features -- 3 hours rolling window")
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(df,
                                   index = 'datetime',
                                   columns = 'machineID',
                                   values = col).resample('3H', closed = 'left', label = 'right').mean().unstack())
    telemetry_mean_3h = pd.concat(temp, axis = 1)
    telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
    telemetry_mean_3h.reset_index(inplace = True)

    # repeat for standard deviation
    print("Calculate standard deviation for telemetry features -- 3 hours rolling window")
    temp = []
    for col in fields:
        temp.append(pd.pivot_table(df,
                                   index = 'datetime',
                                   columns = 'machineID',
                                   values = col).resample('3H', closed = 'left', label = 'right').std().unstack())
    telemetry_sd_3h = pd.concat(temp, axis = 1)
    telemetry_sd_3h.columns = [i + 'sd_3h' for i in fields]
    telemetry_sd_3h.reset_index(inplace = True)
    
    # Calculate mean values for telemetry features -- 24 hours rolling window
    print("Calculate mean values for telemetry features -- 24 hours rolling window")
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(df,
                                   index = 'datetime',
                                   columns = 'machineID',
                                   values = col)
                    .resample('3H', closed = 'left', label = 'right')
                    .first()
                    .unstack()
                    .rolling(window = 24, center = False).mean())
    telemetry_mean_24h = pd.concat(temp, axis = 1)
    telemetry_mean_24h.columns = [i + 'mean_24h' for i in fields]
    telemetry_mean_24h.reset_index(inplace = True)
    telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h['voltmean_24h'].isnull()]

    # repeat for standard deviation
    print("Calculate standard deviation for telemetry features -- 24 hours rolling window")
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(df,
                                   index = 'datetime',
                                   columns = 'machineID',
                                   values = col)
                    .resample('3H', closed='left', label='right')
                    .first()
                    .unstack()
                    .rolling(window = 24, center = False).std())
    telemetry_sd_24h = pd.concat(temp, axis = 1)
    telemetry_sd_24h.columns = [i + 'sd_24h' for i in fields]
    telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h['voltsd_24h'].isnull()]
    telemetry_sd_24h.reset_index(inplace = True)
    
    telemetry_feat = pd.concat([telemetry_mean_3h,
                            telemetry_sd_3h.iloc[:, 2:6],
                            telemetry_mean_24h.iloc[:, 2:6],
                            telemetry_sd_24h.iloc[:, 2:6]], axis = 1).dropna()

    upload_file_s3(telemetry_feat, "telemetry")
    
    return telemetry_feat


# Lag Features for Errors
def errors_lag_features(df):
    df = datetime_datatype(df)
    df = category_datatype(df, 'errorID')
    print("Lag features for errors")
    error_count = pd.get_dummies(df.set_index('datetime')).reset_index()
    error_count.columns = ['datetime', 'machineID', 'error1', 'error2', 'error3', 'error4', 'error5']
    error_count = error_count.groupby(['machineID', 'datetime']).sum().reset_index()
    error_count = telemetry[['datetime', 'machineID']].merge(error_count, on = ['machineID', 'datetime'], how = 'left').fillna(0.0)
    temp = []
    fields = ['error%d' % i for i in range(1, 6)]
    for col in fields:
        temp.append(pd.pivot_table(error_count,
                                   index = 'datetime',
                                   columns = 'machineID',
                                   values = col)
                    .resample('3H', closed='left', label='right')
                    .first()
                    .unstack()
                    .rolling(window = 24, center = False).sum())
    error_count = pd.concat(temp, axis = 1)
    error_count.columns = [i + 'count' for i in fields]
    error_count.reset_index(inplace = True)
    error_count = error_count.dropna()
    
    upload_file_s3(error_count, "errors")
    
    return error_count


# Maintenance Features
def maintenance_features(df):
    df = datetime_datatype(df)
    df = category_datatype(df, 'comp')
    print("Maintenance Features -- Days since last replacement")
    comp_rep = pd.get_dummies(df.set_index('datetime')).reset_index()
    comp_rep.columns = ['datetime', 'machineID', 'comp1', 'comp2', 'comp3', 'comp4']

    # combine repairs for a given machine in a given hour
    comp_rep = comp_rep.groupby(['machineID', 'datetime']).sum().reset_index()

    # add timepoints where no components were replaced
    comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep,
                                                          on=['datetime', 'machineID'],
                                                          how='outer').fillna(0).sort_values(by=['machineID', 'datetime'])
    components = ['comp1', 'comp2', 'comp3', 'comp4']
    for comp in components:
        comp_rep.loc[comp_rep[comp] < 1, comp] = None
        comp_rep.loc[-comp_rep[comp].isnull(),
                     comp] = comp_rep.loc[-comp_rep[comp].isnull(), 'datetime']
        comp_rep[comp] = comp_rep[comp].fillna(method = 'ffill')

    comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]
    for comp in components:
        comp_rep[comp] = (comp_rep["datetime"] - pd.to_datetime(comp_rep[comp])) / np.timedelta64(1, "D")
        
    upload_file_s3(comp_rep, "maint")
    
    return comp_rep


# Failures Features
def failure_features(df):
    print("Failure features")
    df = datetime_datatype(df)
    df = category_datatype(df, 'failure')
    upload_file_s3(df, "failures")
    return df


# Final Features
def final_features(telemetry_df, errors_df, maint_df, machines_df):
    upload_file_s3(machines_df, "machines")
    print("Final features")
    final_feat = telemetry_df.merge(errors_df, on = ['datetime', 'machineID'], how = 'left')
    final_feat = final_feat.merge(maint_df, on = ['datetime', 'machineID'], how = 'left')
    final_feat = final_feat.merge(machines_df, on = ['machineID'], how = 'left')
    return final_feat


# Label Construction
def label_construct(tele_df, error_df, maint_df, machine_df, failure_df):
    print("----- Final Features -----")
    final_feat = final_features(tele_df, error_df, maint_df, machine_df)
    
    print("----- Label Construction -----")
    labeled_features = pd.DataFrame()
    labeled_features = final_feat.merge(
        failure_df, on = ['datetime', 'machineID'], how = 'left')
    labeled_features['failure'] = labeled_features['failure'].astype(str)
    labeled_features['failure'] = labeled_features['failure'].fillna(method = 'bfill', limit = 7)
    labeled_features['failure'] = labeled_features['failure'].replace('nan', 'none')
    print("----- Preprocessing completed -----")
    
    upload_file_s3(labeled_features, "preprocessed")
#     pd.DataFrame(labeled_features).to_csv(f"{base_dir}/preprocessed/final_data.csv", index = False)


if __name__ == "__main__":

    telemetry_data_uri = f"s3://{bucket}/{prefix}/data/raw/PdM_telemetry.csv"
    errors_data_uri = f"s3://{bucket}/{prefix}/data/raw/PdM_errors.csv"
    maint_data_uri = f"s3://{bucket}/{prefix}/data/raw/PdM_maint.csv"
    failures_data_uri = f"s3://{bucket}/{prefix}/data/raw/PdM_failures.csv"
    machines_data_uri = f"s3://{bucket}/{prefix}/data/raw/PdM_machines.csv"
    
    telemetry = wr.s3.read_csv(telemetry_data_uri)
    errors = wr.s3.read_csv(errors_data_uri)
    maint = wr.s3.read_csv(maint_data_uri)
    failures = wr.s3.read_csv(failures_data_uri)
    machines = wr.s3.read_csv(machines_data_uri)
    
    telemetry_df = telemetry_features(telemetry)
    errors_df = errors_lag_features(errors)
    maint_df = maintenance_features(maint)
    failures_df = failure_features(failures)
    machines_df = category_datatype(machines, 'model')
    
    label_construct(telemetry_df, errors_df, maint_df, machines_df, failures_df)
