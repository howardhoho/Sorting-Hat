from flask import render_template, request, jsonify
from PIL import Image, ExifTags
import numpy as np
import cv2
import io
from hogwarts_sorter.myapp import app  
from hogwarts_sorter.myapp.utils import make_prediction  
from hogwarts_sorter.myapp.utils.s3_upload import upload_image_to_s3
from hogwarts_sorter.myapp.utils.rds_upload import insert_into_db

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
    allowed_extensions = {'png', 'jpg', 'jpeg', 'heif', 'heic'}
    
    if '.' not in file.filename:
        return jsonify({'error': 'File must have an extension'}), 400
    
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'Unsupported file format: {file_ext}'}), 400

    # Map file extension to proper image format
    format_map = {
        'png': 'PNG',
        'jpg': 'JPEG',
        'jpeg': 'JPEG',
        'heif': 'JPEG',  # Convert HEIF to JPEG for upload
        'heic': 'JPEG'   # Convert HEIC to JPEG for upload
    }

    image_format = format_map.get(file_ext)  

    if not image_format:
        return jsonify({'error': 'Unsupported file format'}), 400

    # Read raw bytes from the uploaded file
    raw_data = file.read()

    # Handle HEIF/HEIC files by converting to JPEG first
    if file_ext in ('heic', 'heif'):
        image = None
        
        # Method 1: Try imageio (often most reliable for HEIC)
        try:
            import imageio.v3 as iio
            image_array = iio.imread(raw_data, extension=f'.{file_ext}')
            image = Image.fromarray(image_array)
            
        except Exception:
            # Method 2: Try using macOS built-in converter (sips command)
            try:
                import tempfile
                import os
                import subprocess
                
                with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as heic_temp:
                    heic_temp.write(raw_data)
                    heic_path = heic_temp.name
                
                jpeg_path = heic_path.replace(f'.{file_ext}', '.jpg')
                
                result = subprocess.run([
                    'sips', '-s', 'format', 'jpeg', heic_path, '--out', jpeg_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    image = Image.open(jpeg_path)
                else:
                    raise Exception(f"sips conversion failed: {result.stderr}")
                
                # Clean up temp files
                os.unlink(heic_path)
                if os.path.exists(jpeg_path):
                    os.unlink(jpeg_path)
                    
            except Exception:
                # Method 3: Try Wand (ImageMagick binding)
                try:
                    from wand.image import Image as WandImage
                    
                    with WandImage(blob=raw_data) as wand_img:
                        wand_img.format = 'jpeg'
                        wand_img.compression_quality = 85
                        
                        jpeg_blob = wand_img.make_blob()
                        image = Image.open(io.BytesIO(jpeg_blob))
                        
                except Exception:
                    # Method 4: Try pillow-heif as last resort
                    try:
                        from pillow_heif import register_heif_opener
                        register_heif_opener()
                        
                        bytes_io = io.BytesIO(raw_data)
                        bytes_io.seek(0)
                        image = Image.open(bytes_io)
                        image.load()
                        
                    except Exception:
                        return jsonify({
                            'error': 'Unable to convert HEIC file. Please convert to JPEG manually.',
                            'suggestion': 'On iPhone: Open Photos → Select image → Share → Save to Files (this converts to JPEG)',
                            'alternative': 'Or change iPhone settings: Settings → Camera → Formats → Most Compatible'
                        }), 400
        
        if image is None:
            return jsonify({
                'error': 'Failed to convert HEIC file. Please convert to JPEG manually.',
                'suggestion': 'On iPhone: Open Photos → Select image → Share → Save to Files'
            }), 400
            
        # Fix EXIF orientation if needed
        try:
            exif = image.getexif()
            if exif:
                orientation = exif.get(0x0112)  # Orientation tag
                if orientation:
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
        except Exception:
            pass  # Continue if EXIF processing fails

        # Convert to RGB
        image = image.convert("RGB")
            
    else:
        # Handle regular JPEG/PNG files
        try:
            image = Image.open(io.BytesIO(raw_data)).convert("RGB")
        except Exception as e:
            return jsonify({'error': f'Cannot open image: {e}'}), 400

    # Upload the image to S3 in its converted format
    try:
        s3_file_name = upload_image_to_s3(image, file.filename, image_format)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500

    # Convert to OpenCV format for processing
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Get prediction and processed images
    result = make_prediction.process_image(img_cv)
    
    # Handle both tuple and dict returns
    if isinstance(result, tuple):
        if len(result) == 3:
            prediction_label, resized_img, landmarked_img = result
        else:
            return jsonify({'error': 'Unexpected prediction result format'}), 500
    elif isinstance(result, dict):
        prediction_label = result['prediction']
        resized_img = result['resized_img']
        landmarked_img = result['landmarked_img']
    else:
        return jsonify({'error': 'Unexpected prediction result format'}), 500

    # Convert output images from BGR back to RGB for frontend display
    if isinstance(resized_img, np.ndarray):
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    if isinstance(landmarked_img, np.ndarray):
        landmarked_img = cv2.cvtColor(landmarked_img, cv2.COLOR_BGR2RGB)

    # Upload the s3 url to RDS
    try:
        insert_into_db(file.filename, f"s3://{s3_file_name}", prediction_label)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Return the result
    return jsonify({
        'prediction': prediction_label,
        'resized_img': resized_img,
        'landmarked_img': landmarked_img
    }), 200

if __name__ == "__main__":
    app.run(debug=True)
    