from flask import render_template, request, jsonify
from PIL import Image
import numpy as np
from hogwarts_sorter.myapp import app  # Import the Flask app instance
from hogwarts_sorter.myapp.utils import make_prediction  # Correct spelling of the file

@app.route('/', methods=['GET'])  # Removed endpoint='home'
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    

    image = Image.open(file)
    image = image.convert("RGB")
    img_cv = np.array(image)

    result, status = make_prediction.process_image(img_cv)
    
    # Return the result and the status code
    return result, status


if __name__ == "__main__":
    # Start the Flask app
    app.run(debug=True)



