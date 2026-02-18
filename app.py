"""
ReefSpark Dashboard - Main Application
A unified platform for oceanographic data and reef health monitoring
"""
from flask import Flask
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import blueprints
from routes.pages import pages
from routes.litters import litters_api
from routes.ocean_data import ocean_api
from routes.reef_stress import reef_api
from routes.chat import chat_api

# Create Flask app
app = Flask(__name__)

# Register blueprints
app.register_blueprint(pages)
app.register_blueprint(litters_api)
app.register_blueprint(ocean_api)
app.register_blueprint(reef_api)
app.register_blueprint(chat_api)

if __name__ == '__main__':
    app.run(debug=True)
