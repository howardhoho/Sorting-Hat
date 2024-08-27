import boto3
import io

# Initialize S3 client
s3 = boto3.client('s3')

# Your bucket name
BUCKET_NAME = 'your-bucket-name'

def upload_image_to_s3(image, filename, file_format):
    """Uploads an image to an S3 bucket in the original format."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=file_format)  # Use the original file format
    img_byte_arr.seek(0)

    s3_file_name = f"uploads/{filename}"

    try:
        s3.upload_fileobj(img_byte_arr, BUCKET_NAME, s3_file_name)
    except Exception as e:
        raise RuntimeError(f"Failed to upload image to S3: {e}")
    
    return s3_file_name
