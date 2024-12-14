SageMaker Endpoint Deployment

This repository contains code and scripts to deploy a machine learning model as an endpoint using AWS SageMaker. It also includes a custom inference script to handle predictions and input processing.

Files in the Repository

create_endpoint.ipynb
This Jupyter Notebook provides step-by-step instructions to:

Deploy a machine learning model stored in an S3 bucket to an AWS SageMaker endpoint. Define the model, configure the endpoint, and deploy it using Boto3.

inference.py
This Python script handles the inference logic for the deployed SageMaker endpoint. It includes:

model_fn: Loads the model from the directory. input_fn: Processes incoming requests (e.g., base64-encoded images). predict_fn: Makes predictions using the loaded model. output_fn: Formats the response for the client.

Model Artifact Structure

The model.tar.gz file should include a code folder containing the inference.py script for custom predictions and a requirements.txt file listing dependencies.
