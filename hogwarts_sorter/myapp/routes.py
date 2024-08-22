from flask import render_template, request, jsonify
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
    
    # Process the image and get the result
    upload_folder = '/Users/howardhoho/Desktop/Sorting-Hat/hogwarts_sorter/uploads'
    result, status = make_prediction.process_image(file, upload_folder)
    
    # Return the result and the status code
    return result, status


if __name__ == "__main__":
    # Start the Flask app
    app.run(debug=True)



