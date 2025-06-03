from flask import render_template, request, jsonify
from PIL import Image
import numpy as np
from hogwarts_sorter.myapp import app  
from hogwarts_sorter.myapp.utils import make_prediction  
from hogwarts_sorter.myapp.utils.s3_upload import upload_image_to_s3
from hogwarts_sorter.myapp.utils.rds_upload import insert_into_db

# Register HEIF opener once at startup so PIL can handle HEIC/HEIF files
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("✅ pillow-heif registered - HEIC/HEIF files supported")
except ImportError:
    print("⚠️ pillow-heif not available - HEIC/HEIF files not supported")

@app.route('/', methods=['GET'])  
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check the file extension and determine format
    allowed_extensions = {'png', 'jpg', 'jpeg', 'heic', 'heif'}
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Unsupported file format'}), 400

    # Map file extension to proper image format for S3 upload
    format_map = {
        'png': 'PNG',
        'jpg': 'JPEG',
        'jpeg': 'JPEG',
        'heic': 'JPEG',  # Convert HEIC to JPEG for S3 storage
        'heif': 'JPEG'   # Convert HEIF to JPEG for S3 storage
    }

    image_format = format_map.get(file_ext)

    if not image_format:
        return jsonify({'error': 'Unsupported file format'}), 400

    # Open image (works for all formats including HEIC/HEIF thanks to pillow-heif registration)
    try:
        image = Image.open(file)
        image = image.convert("RGB")
        print(f"✅ Successfully opened {file_ext.upper()} image: {image.size}")
    except Exception as e:
        print(f"❌ Failed to open image: {str(e)}")
        return jsonify({'error': f'Failed to open image: {str(e)}'}), 400

    # Upload the image to S3
    try:
        s3_file_name = upload_image_to_s3(image, file.filename, image_format)
        print(f"✅ Uploaded to S3: {s3_file_name}")
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500

    # Convert PIL image to numpy array for OpenCV processing
    img_cv = np.array(image)

    # Get raw prediction and images
    result = make_prediction.process_image(img_cv)
    
    # Extract the prediction and image data
    prediction_label = result['prediction']
    resized_img = result['resized_img']
    landmarked_img = result['landmarked_img']

    # Upload the s3 url to RDS
    try:
        insert_into_db(file.filename, f"s3://{s3_file_name}", prediction_label)
    except Exception as e:
        print(f"Error during DB insert: {e}")
        return jsonify({'error': str(e)}), 500

    # Return the result and the status code
    return jsonify({
        'prediction': prediction_label,
        'resized_img': resized_img,
        'landmarked_img': landmarked_img
    }), 200

if __name__ == "__main__":
    app.run(debug=True)
