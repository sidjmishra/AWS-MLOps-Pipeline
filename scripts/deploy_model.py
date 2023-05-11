import argparse
import time
import sagemaker
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.pipeline_context import PipelineSession
import boto3

# Parse argument variables passed via the DeployModel processing step
parser = argparse.ArgumentParser()
# parser.add_argument("--model-name", type=str)

# ------------------------------------------------------ 1
parser.add_argument("--model-data",type = str)
# ------------------------------------------------------ 1

parser.add_argument("--region", type = str)
parser.add_argument("--endpoint-instance-type", type = str)
parser.add_argument("--endpoint-name", type = str)
args = parser.parse_args()

region = args.region
boto3.setup_default_session(region_name = region)
sagemaker_boto_client = boto3.client("sagemaker")

# ------------------------------------------------------ 2
sagemaker_role = sagemaker.get_execution_role()
# pipeline_session = PipelineSession()
boto_session = boto3.Session(region_name = region)
sagemaker_session = sagemaker.session.Session(
    boto_session = boto_session, sagemaker_client = sagemaker_boto_client
)
# ------------------------------------------------------ 2

bucket = "ideaaiml-demo"
prefix = "mlops/predictive-maintenance"

# ------------------------------------------------------ 3
# Create a model - PreDeployment
model = SKLearnModel(
    model_data = args.model_data,
    role = sagemaker_role,
    entry_point = "rf_script.py",
    framework_version = "1.2-1",
    sagemaker_session = sagemaker_session,
)

# model.create(name = "PdM-Model", instance_type = args.endpoint_instance_type)

data_capture_config = DataCaptureConfig(
    enable_capture = True, 
    sampling_percentage = 100, 
    destination_s3_uri = f"s3://{bucket}/{prefix}/data-capture-model-monitor",
    capture_options = ['REQUEST', 'RESPONSE'], 
    csv_content_types = ['text/csv'], 
    json_content_types = ['application/json']
)

model.deploy(
    initial_instance_count = 1,
    instance_type = args.endpoint_instance_type,
    endpoint_name = args.endpoint_name,
    data_capture_config = data_capture_config,
)

# ------------------------------------------------------ 3

# name truncated per sagameker length requirememnts (63 char max)
# endpoint_config_name = f"{args.model_name[:56]}-config"
# existing_configs = sagemaker_boto_client.list_endpoint_configs(NameContains = endpoint_config_name)[
#     "EndpointConfigs"
# ]

# if not existing_configs:
#     create_ep_config_response = sagemaker_boto_client.create_endpoint_config(
#         EndpointConfigName = endpoint_config_name,
#         ProductionVariants = [
#             {
#                 "InstanceType": args.endpoint_instance_type,
#                 "InitialVariantWeight": 1,
#                 "InitialInstanceCount": 1,
#                 "ModelName": args.model_name,
#                 "VariantName": "AllTraffic",
#             }
#         ],
# #         DataCaptureConfig = {
# #         'EnableCapture': True,
# #         'InitialSamplingPercentage': 100,
# #         'DestinationS3Uri': f's3://{bucket}/{prefix}/data-capture-model-monitor',
# #         'CaptureOptions': [
# #             {
# #                 'CaptureMode': 'InputAndOutput'
# #             },
# #         ],
# #         'CaptureContentTypeHeader': {
# #             'JsonContentTypes': [
# #                 'application/json',
# #             ]
# #         }
# #     }
#     )

# existing_endpoints = sagemaker_boto_client.list_endpoints(NameContains = args.endpoint_name)[
#     "Endpoints"
# ]

# if not existing_endpoints:
#     create_endpoint_response = sagemaker_boto_client.create_endpoint(
#         EndpointName = args.endpoint_name, EndpointConfigName = endpoint_config_name
#     )

# endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName = args.endpoint_name)
# endpoint_status = endpoint_info["EndpointStatus"]

# while endpoint_status == "Creating":
#     endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName = args.endpoint_name)
#     endpoint_status = endpoint_info["EndpointStatus"]
#     print("Endpoint status:", endpoint_status)
#     if endpoint_status == "Creating":
#         time.sleep(60)