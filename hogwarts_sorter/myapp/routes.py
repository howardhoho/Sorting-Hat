from flask import render_template, request, jsonify
from PIL import Image, ExifTags
import numpy as np
import pyheif
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
    
    # 1) Expand allowed extensions to include HEIC/HEIF
    allowed_extensions = {'png', 'jpg', 'jpeg', 'heic', 'heif'}
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Unsupported file format'}), 400
    
    # 2) Map file extension to PIL format name (for S3 upload)
    format_map = {
        'png': 'PNG',
        'jpg': 'JPEG',
        'jpeg': 'JPEG',
        'heic': 'JPEG',   # We’ll convert HEIC → JPEG when uploading
        'heif': 'JPEG'
    }
    image_format = format_map.get(file_ext)
    if not image_format:
        return jsonify({'error': 'Unsupported file format'}), 400

    # 3) Read raw bytes from the uploaded file
    raw_data = file.read()

    # 4) If HEIC/HEIF, decode via pyheif → Pillow → ensure RGB
    if file_ext in ('heic', 'heif'):
        try:
            # Decode the HEIC container
            heif_file = pyheif.read(raw_data)
            # Build a Pillow Image from the HEIC data
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            # (Optional) fix EXIF orientation if iPhone inserted a rotation tag
            try:
                exif = image._getexif()
                if exif is not None:
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        if tag == "Orientation":
                            if value == 3:
                                image = image.rotate(180, expand=True)
                            elif value == 6:
                                image = image.rotate(270, expand=True)
                            elif value == 8:
                                image = image.rotate(90, expand=True)
                            break
            except Exception:
                pass

            # Convert to normal RGB (discard any alpha/CMYK channels)
            image = image.convert("RGB")
        except Exception as e:
            return jsonify({'error': f'Failed to decode HEIC: {e}'}), 400

    else:
        # 5) If already JPEG/PNG, let Pillow open it directly
        try:
            image = Image.open(io.BytesIO(raw_data)).convert("RGB")
        except Exception as e:
            return jsonify({'error': f'Cannot open image: {e}'}), 400

    # 6) Upload the (converted) image to S3 as JPEG/PNG
    try:
        s3_file_name = upload_image_to_s3(image, file.filename, image_format)  
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500

    # 7) Convert the Pillow image to an OpenCV BGR numpy array
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 8) Run the existing prediction pipeline
    result = make_prediction.process_image(img_cv)
    if 'error' in result:
        return jsonify(result), result.get('status_code', 500)

    prediction_label = result['prediction']
    resized_img = result['resized_img']
    landmarked_img = result['landmarked_img']

    # 9) Insert into RDS
    try:
        insert_into_db(file.filename, f"s3://{s3_file_name}", prediction_label)
    except Exception as e:
        return jsonify({'error': f'DB insert failed: {e}'}), 500

    # 10) Return JSON response
    return jsonify({
        'prediction': prediction_label,
        'resized_img': resized_img,
        'landmarked_img': landmarked_img
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


