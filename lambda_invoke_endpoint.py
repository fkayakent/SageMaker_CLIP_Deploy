import boto3
import json
import os
import base64
from PIL import Image

# Initialize AWS clients
s3 = boto3.client('s3')
sagemaker_client = boto3.client('runtime.sagemaker')

# Define constants
ENDPOINT_NAME = "your-endpoint"

# Resizing parameters
RESIZE_DIMENSIONS = (224, 224)  # Resize to 224x224

def resize_image(input_path, output_path, size=(224, 224)):
    """Resize the image to the specified dimensions."""
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")  # Ensure image is in RGB format
            img = img.resize(size, Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality resizing
            img.save(output_path, "JPEG")
            print(f"Image resized to {size} and saved at {output_path}")
    except Exception as e:
        print(f"Error resizing image: {str(e)}")
        raise

def lambda_handler(event, context):
    for record in event['Records']:
        try:
            # Extract bucket name and image key for the current record
            source_bucket = record['s3']['bucket']['name']
            image_key = record['s3']['object']['key']
            print(f"Processing image: {image_key} from bucket: {source_bucket}")
        except KeyError as e:
            print(f"Error extracting bucket or image key from event: {str(e)}")
            continue  # Skip to the next record

        # Process the image
        filename = os.path.basename(image_key)
        download_path = f"/tmp/original_{filename}"
        resized_path = f"/tmp/resized_{filename}"

        # Download image from S3
        try:
            s3.download_file(source_bucket, image_key, download_path)
            print(f"Downloaded {image_key} to {download_path}")
        except Exception as e:
            print(f"Error downloading file: {str(e)}")
            continue

        # Validate the image extension
        if not image_key.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            print(f"File is not an image: {image_key}")
            continue  # Skip to the next record

        # Resize the image
        try:
            resize_image(download_path, resized_path)
        except Exception as e:
            print(f"Error during image resizing: {str(e)}")
            continue

        # Prepare the resized image for prediction
        try:
            with open(resized_path, "rb") as f:
                base64_encoded = base64.b64encode(f.read()).decode("utf-8")
            request_body = {
                "inputs": base64_encoded,
                "parameters": {
                    "candidate_labels": ["cat", "dog", "cat and dog"]
                }
            }
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            continue

        # Invoke SageMaker endpoint
        try:
            response = sagemaker_client.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(request_body)
            )
            response_body = json.loads(response['Body'].read().decode())
            print(f"Response from SageMaker: {response_body}")
        except Exception as e:
            print(f"Error invoking SageMaker endpoint: {str(e)}")
            continue
        finally:
            # Clean up temporary files
            for path in [download_path, resized_path]:
                if os.path.exists(path):
                    os.remove(path)

    return {
        'statusCode': 200,
        'body': json.dumps("Prediction completed successfully.")
    }
