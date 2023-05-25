import numpy as np
import pandas as pd
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
import awswrangler as wr

base_dir = "/opt/ml/processing"
bucket = "BUCKET-NAME"
prefix = "mlops/predictive-maintenance"

boto_session = boto3.Session(region_name = "us-east-1")
sagemaker_boto_client = boto_session.client("sagemaker")
featurestore_runtime = boto_session.client(
    service_name = "sagemaker-featurestore-runtime", region_name = "us-east-1"
)
try:
    sagemaker_role = sagemaker.get_execution_role()
    print(f"Sagemaker Role for Feature Store file: {sagemaker_role}")
except ValueError:
    sagemaker_role = 'SAGENAKER-ROLE'
    
feature_store_session = sagemaker.Session(
    boto_session = boto_session,
    sagemaker_client = sagemaker_boto_client,
    sagemaker_featurestore_runtime_client = featurestore_runtime,
)

# ------------------------------------ Read Data
telemetry_data_uri = f"s3://{bucket}/{prefix}/data/preprocessed/telemetry.csv"
errors_data_uri = f"s3://{bucket}/{prefix}/data/preprocessed/errors.csv"
maint_data_uri = f"s3://{bucket}/{prefix}/data/preprocessed/maint.csv"
failures_data_uri = f"s3://{bucket}/{prefix}/data/preprocessed/failures.csv"
machines_data_uri = f"s3://{bucket}/{prefix}/data/preprocessed/machines.csv"

telemetry = wr.s3.read_csv(telemetry_data_uri)
errors = wr.s3.read_csv(errors_data_uri)
maint = wr.s3.read_csv(maint_data_uri)
failures = wr.s3.read_csv(failures_data_uri)
machines = wr.s3.read_csv(machines_data_uri)

# ------------------------------------ Add Timestamp
telemetry['event_time'] = pd.to_datetime("now").timestamp()
errors['event_time'] = pd.to_datetime("now").timestamp()
maint['event_time'] = pd.to_datetime("now").timestamp()
failures['event_time'] = pd.to_datetime("now").timestamp()
machines['event_time'] = pd.to_datetime("now").timestamp()

# ------------------------------------ Create Feature Group
telemetry_feature_group = FeatureGroup(name = 'telemetry_fg', sagemaker_session = feature_store_session)
errors_feature_group = FeatureGroup(name = 'errors_fg', sagemaker_session = feature_store_session)
maintenance_feature_group = FeatureGroup(name = 'maintenance_fg', sagemaker_session = feature_store_session)
failures_feature_group = FeatureGroup(name = 'failures_fg', sagemaker_session = feature_store_session)
machines_feature_group = FeatureGroup(name = 'machines_fg', sagemaker_session = feature_store_session)

# ------------------------------------ Loading Definitions
telemetry_feature_group.load_feature_definitions(data_frame = telemetry)
errors_feature_group.load_feature_definitions(data_frame = errors)
maintenance_feature_group.load_feature_definitions(data_frame = maint)
failures_feature_group.load_feature_definitions(data_frame = failures)
machines_feature_group.load_feature_definitions(data_frame = machines)

record_identifier_feature_name = "machineID"
event_time_feature_name = "event_time"

# ------------------------------------ Telemetry Feature Store
try:
    telemetry_feature_group.create(
        s3_uri = f"s3://{bucket}/{prefix}/feature_store_data",
        record_identifier_name = record_identifier_feature_name,
        event_time_feature_name = event_time_feature_name,
        role_arn = sagemaker_role,
        enable_online_store = True,
    )
    time.sleep(30)
    print(f'Create "telemetry" feature group: SUCCESS')
except Exception as e:
    code = e.response.get("Error").get("Code")
    if code == "ResourceInUse":
        print("Using existing feature group")
    else:
        raise (e)

# ------------------------------------ Errors Feature Store      
try:
    errors_feature_group.create(
        s3_uri = f"s3://{bucket}/{prefix}/feature_store_data",
        record_identifier_name = record_identifier_feature_name,
        event_time_feature_name = event_time_feature_name,
        role_arn = sagemaker_role,
        enable_online_store = True,
    )
    time.sleep(30)
    print(f'Create "errors" feature group: SUCCESS')
except Exception as e:
    code = e.response.get("Error").get("Code")
    if code == "ResourceInUse":
        print("Using existing feature group")
    else:
        raise (e)

# ------------------------------------ Maintenance Feature Store
try:
    maintenance_feature_group.create(
        s3_uri = f"s3://{bucket}/{prefix}/feature_store_data",
        record_identifier_name = record_identifier_feature_name,
        event_time_feature_name = event_time_feature_name,
        role_arn = sagemaker_role,
        enable_online_store = True,
    )
    time.sleep(30)
    print(f'Create "maintenance" feature group: SUCCESS')
except Exception as e:
    code = e.response.get("Error").get("Code")
    if code == "ResourceInUse":
        print("Using existing feature group")
    else:
        raise (e)

# ------------------------------------ Failures Feature Store
try:
    failures_feature_group.create(
        s3_uri = f"s3://{bucket}/{prefix}/feature_store_data",
        record_identifier_name = record_identifier_feature_name,
        event_time_feature_name = event_time_feature_name,
        role_arn = sagemaker_role,
        enable_online_store = True,
    )
    time.sleep(30)
    print(f'Create "failures" feature group: SUCCESS')
except Exception as e:
    code = e.response.get("Error").get("Code")
    if code == "ResourceInUse":
        print("Using existing feature group")
    else:
        raise (e)

# ------------------------------------ Machines Feature Store
try:
    machines_feature_group.create(
        s3_uri = f"s3://{bucket}/{prefix}/feature_store_data",
        record_identifier_name = record_identifier_feature_name,
        event_time_feature_name = event_time_feature_name,
        role_arn = sagemaker_role,
        enable_online_store = True,
    )
    time.sleep(30)
    print(f'Create "machines" feature group: SUCCESS')
except Exception as e:
    code = e.response.get("Error").get("Code")
    if code == "ResourceInUse":
        print("Using existing feature group")
    else:
        raise (e)

# ------------------------------------ Ingesting Data
while (
    telemetry_feature_group.describe()['FeatureGroupStatus'] == 'Creating'):
    print("Feature Group Creating")
    time.sleep(60)
else:
    print("Feature Group Created")
    ## Below code needs to run only once to ingest the data into feature store
    #---------------------------------------------------------------
#     telemetry_feature_group.ingest(data_frame = telemetry, max_workers = 3, wait = True)
#     errors_feature_group.ingest(data_frame = errors, max_workers = 3, wait = True)
#     maintenance_feature_group.ingest(data_frame = maint, max_workers = 3, wait = True)
#     failures_feature_group.ingest(data_frame = failures, max_workers = 3, wait = True)
#     machines_feature_group.ingest(data_frame = machines, max_workers = 3, wait = True)
#     print("Feature Data Ingested")
    #---------------------------------------------------------------
