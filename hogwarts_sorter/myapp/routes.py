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
        print(f"🔍 Processing HEIC/HEIF file: {file.filename}")
        print(f"🔍 File size: {len(raw_data)} bytes")
        image = None
        
        # Method 1: Try imageio (often most reliable for HEIC)
        try:
            print("🔍 Method 1: Trying imageio...")
            import imageio.v3 as iio
            image_array = iio.imread(raw_data, extension=f'.{file_ext}')
            print(f"✅ imageio read successful - Array shape: {image_array.shape}, dtype: {image_array.dtype}")
            image = Image.fromarray(image_array)
            print(f"✅ PIL Image created - Size: {image.size}, Mode: {image.mode}")
            
        except Exception as e1:
            print(f"❌ Method 1 (imageio) failed: {e1}")
            
            # Method 2: Try using macOS built-in converter (sips command)
            try:
                print("🔍 Method 2: Trying macOS sips...")
                import tempfile
                import os
                import subprocess
                
                with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as heic_temp:
                    heic_temp.write(raw_data)
                    heic_path = heic_temp.name
                
                print(f"🔍 Temp HEIC file created: {heic_path}")
                jpeg_path = heic_path.replace(f'.{file_ext}', '.jpg')
                
                result = subprocess.run([
                    'sips', '-s', 'format', 'jpeg', heic_path, '--out', jpeg_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ sips conversion successful")
                    image = Image.open(jpeg_path)
                    print(f"✅ PIL Image loaded from converted JPEG - Size: {image.size}")
                else:
                    raise Exception(f"sips conversion failed: {result.stderr}")
                
                # Clean up temp files
                os.unlink(heic_path)
                if os.path.exists(jpeg_path):
                    os.unlink(jpeg_path)
                print("🔍 Temp files cleaned up")
                    
            except Exception as e2:
                print(f"❌ Method 2 (sips) failed: {e2}")
                
                # Method 3: Try Wand (ImageMagick binding)
                try:
                    print("🔍 Method 3: Trying Wand/ImageMagick...")
                    from wand.image import Image as WandImage
                    
                    with WandImage(blob=raw_data) as wand_img:
                        print(f"🔍 Wand image loaded - Size: {wand_img.size}")
                        wand_img.format = 'jpeg'
                        wand_img.compression_quality = 85
                        
                        jpeg_blob = wand_img.make_blob()
                        print(f"🔍 JPEG blob created - Size: {len(jpeg_blob)} bytes")
                        image = Image.open(io.BytesIO(jpeg_blob))
                        print(f"✅ PIL Image created from Wand - Size: {image.size}")
                        
                except Exception as e3:
                    print(f"❌ Method 3 (Wand) failed: {e3}")
                    
                    # Method 4: Try pillow-heif as last resort
                    try:
                        print("🔍 Method 4: Trying pillow-heif...")
                        from pillow_heif import register_heif_opener
                        register_heif_opener()
                        print("🔍 HEIF opener registered")
                        
                        bytes_io = io.BytesIO(raw_data)
                        bytes_io.seek(0)
                        image = Image.open(bytes_io)
                        image.load()
                        print(f"✅ pillow-heif successful - Size: {image.size}")
                        
                    except Exception as e4:
                        print(f"❌ Method 4 (pillow-heif) failed: {e4}")
                        print(f"❌ All HEIC conversion methods failed!")
                        return jsonify({
                            'error': 'Unable to convert HEIC file. Please convert to JPEG manually.',
                            'suggestion': 'On iPhone: Open Photos → Select image → Share → Save to Files (this converts to JPEG)',
                            'alternative': 'Or change iPhone settings: Settings → Camera → Formats → Most Compatible'
                        }), 400
        
        if image is None:
            print("❌ No image object created after all methods")
            return jsonify({
                'error': 'Failed to convert HEIC file. Please convert to JPEG manually.',
                'suggestion': 'On iPhone: Open Photos → Select image → Share → Save to Files'
            }), 400
            
        print("🔍 Starting EXIF orientation processing...")
        # Fix EXIF orientation if needed
        try:
            exif = image.getexif()
            if exif:
                orientation = exif.get(0x0112)  # Orientation tag
                if orientation:
                    print(f"🔄 Found EXIF orientation: {orientation}")
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                        print("🔄 Rotated 180°")
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                        print("🔄 Rotated 270°")
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
                        print("🔄 Rotated 90°")
                else:
                    print("ℹ️ No orientation tag found")
            else:
                print("ℹ️ No EXIF data found")
        except Exception as exif_error:
            print(f"⚠️ EXIF processing error: {exif_error}")

        print("🔍 Converting to RGB...")
        # Convert to RGB
        image = image.convert("RGB")
        print(f"✅ RGB conversion complete - Size: {image.size}, Mode: {image.mode}")
        
        print("🔍 Starting image normalization...")
        # Normalize HEIC image to match JPG format expectations
        # Ensure consistent size and format for prediction processing
        width, height = image.size
        print(f"🔍 Original dimensions: {width}x{height}")
        
        # If image is too large, resize it to reasonable dimensions
        max_dimension = 2048
        if width > max_dimension or height > max_dimension:
            print(f"🔍 Image too large, resizing from {width}x{height}")
            if width > height:
                new_width = max_dimension
                new_height = int(height * max_dimension / width)
            else:
                new_height = max_dimension
                new_width = int(width * max_dimension / height)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"✅ Resized to: {new_width}x{new_height}")
        else:
            print("ℹ️ Image size is acceptable, no resizing needed")
        
        # Ensure the image has the expected properties
        image = image.convert("RGB")  # Double-check RGB conversion
        print(f"✅ Final HEIC image properties - Size: {image.size}, Mode: {image.mode}")
        
        # Verify image is valid
        if image.size[0] == 0 or image.size[1] == 0:
            print("❌ Invalid image dimensions detected")
            return jsonify({'error': 'Invalid image dimensions after HEIC conversion'}), 400
        
        print("✅ HEIC processing completed successfully")
            
    else:
        print(f"🔍 Processing regular {file_ext.upper()} file: {file.filename}")
        # Handle regular JPEG/PNG files
        try:
            image = Image.open(io.BytesIO(raw_data)).convert("RGB")
            print(f"✅ Regular image loaded - Size: {image.size}, Mode: {image.mode}")
        except Exception as e:
            print(f"❌ Regular image processing failed: {e}")
            return jsonify({'error': f'Cannot open image: {e}'}), 400

    # Upload the image to S3 in its converted format
    try:
        print(f"🔍 Uploading to S3 - Format: {image_format}")
        s3_file_name = upload_image_to_s3(image, file.filename, image_format)
        print(f"✅ S3 upload successful: {s3_file_name}")
    except RuntimeError as e:
        print(f"❌ S3 upload failed: {e}")
        return jsonify({'error': str(e)}), 500

    print("🔍 Preparing image for OpenCV processing...")
    # Ensure image is in the correct format for OpenCV processing
    # This is especially important for HEIC-converted images
    try:
        # Verify image properties
        print(f"🔍 Image mode before OpenCV prep: {image.mode}")
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("🔄 Converted to RGB mode")
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        print(f"🔍 Numpy array created - Shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        # Verify array properties
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
            print("🔄 Converted to uint8 dtype")
        
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            print(f"❌ Invalid array shape: {img_array.shape}")
            return jsonify({'error': 'Invalid image format for processing'}), 400
        
        # Convert to OpenCV format (RGB to BGR)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        print(f"✅ OpenCV image ready - Shape: {img_cv.shape}, dtype: {img_cv.dtype}")
        
    except Exception as cv_error:
        print(f"❌ OpenCV preparation failed: {cv_error}")
        return jsonify({'error': f'Failed to prepare image for processing: {str(cv_error)}'}), 500

    print("🔍 Starting prediction processing...")
    # Get prediction and processed images
    try:
        result = make_prediction.process_image(img_cv)
        print(f"✅ Prediction completed - Result type: {type(result)}")
        if isinstance(result, tuple):
            print(f"🔍 Tuple result length: {len(result)}")
        elif isinstance(result, dict):
            print(f"🔍 Dict result keys: {list(result.keys())}")
    except Exception as prediction_error:
        print(f"❌ Prediction failed: {prediction_error}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Image processing failed: {str(prediction_error)}',
            'details': 'There was an error during the prediction process. Please try with a different image.'
        }), 500
    
    # Handle both tuple and dict returns
    try:
        if isinstance(result, tuple):
            if len(result) == 3:
                prediction_label, resized_img, landmarked_img = result
            else:
                return jsonify({'error': 'Unexpected prediction result format - tuple length mismatch'}), 500
        elif isinstance(result, dict):
            prediction_label = result.get('prediction')
            resized_img = result.get('resized_img')
            landmarked_img = result.get('landmarked_img')
            
            if prediction_label is None:
                return jsonify({'error': 'Missing prediction in result'}), 500
        else:
            return jsonify({'error': f'Unexpected prediction result type: {type(result)}'}), 500
    except Exception as parse_error:
        return jsonify({
            'error': f'Failed to parse prediction results: {str(parse_error)}',
            'result_type': str(type(result))
        }), 500

    # Convert output images from BGR back to RGB for frontend display
    try:
        if isinstance(resized_img, np.ndarray):
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        if isinstance(landmarked_img, np.ndarray):
            landmarked_img = cv2.cvtColor(landmarked_img, cv2.COLOR_BGR2RGB)
    except Exception as color_error:
        return jsonify({
            'error': f'Failed to convert image colors: {str(color_error)}'
        }), 500

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
    