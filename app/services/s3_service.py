import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile
from decouple import config
import aiofiles
import os
from typing import Optional
import logging

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY'),
            region_name=config('AWS_REGION')
        )
        self.bucket_name = config('S3_BUCKET_NAME')
    
    async def upload_file(self, file: UploadFile, s3_key: str) -> str:
        """
        Upload file to S3 bucket
        """
        try:
            # Save file temporarily
            temp_path = f"/tmp/{file.filename}"
            
            async with aiofiles.open(temp_path, 'wb') as temp_file:
                content = await file.read()
                await temp_file.write(content)
            
            # Upload to S3
            self.s3_client.upload_file(temp_path, self.bucket_name, s3_key)
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Return S3 URL
            return f"s3://{self.bucket_name}/{s3_key}"
            
        except ClientError as e:
            logging.error(f"Failed to upload file to S3: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during S3 upload: {e}")
            raise
    
    async def download_file(self, s3_key: str, local_path: str) -> str:
        """
        Download file from S3 to local path
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download from S3
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            return local_path
            
        except ClientError as e:
            logging.error(f"Failed to download file from S3: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during S3 download: {e}")
            raise
    
    async def upload_processed_file(self, local_path: str, s3_key: str) -> str:
        """
        Upload processed file to S3
        """
        try:
            # Upload to S3
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            
            # Return S3 URL
            return f"s3://{self.bucket_name}/{s3_key}"
            
        except ClientError as e:
            logging.error(f"Failed to upload processed file to S3: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during processed file S3 upload: {e}")
            raise
    
    async def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """
        Generate presigned URL for downloading file
        """
        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return response
            
        except ClientError as e:
            logging.error(f"Failed to generate presigned URL: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during presigned URL generation: {e}")
            raise
    
    async def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            return True
            
        except ClientError as e:
            logging.error(f"Failed to delete file from S3: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error during S3 file deletion: {e}")
            return False
    
    async def file_exists(self, s3_key: str) -> bool:
        """
        Check if file exists in S3
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                return False
            else:
                raise
        except Exception as e:
            logging.error(f"Unexpected error checking S3 file existence: {e}")
            raise
