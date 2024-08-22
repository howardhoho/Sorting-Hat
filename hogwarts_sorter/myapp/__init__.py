from flask import Flask

# Initialize the Flask app
app = Flask(__name__)

# Import routes so that they are registered with the app
from hogwarts_sorter.myapp import routes  # Relative import




