from flask import Flask
from flask_socketio import SocketIO
from dotenv import load_dotenv
import os

load_dotenv()

from routes.pages import pages
from routes.litters import litters_api
from routes.ocean_data import ocean_api
from routes.reef_stress import reef_api
from routes.chat import chat_api
from routes.bleaching import bleaching_api
from routes.rover import rover_api, register_rover_events

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'reefsparkdev')

# SocketIO — threading mode for dev, switch to eventlet in production
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# Register HTTP blueprints
app.register_blueprint(pages)
app.register_blueprint(litters_api)
app.register_blueprint(ocean_api)
app.register_blueprint(reef_api)
app.register_blueprint(chat_api)
app.register_blueprint(bleaching_api)
app.register_blueprint(rover_api)

# Register SocketIO events
register_rover_events(socketio)

if __name__ == '__main__':
    # socketio.run instead of app.run when SocketIO is active
    socketio.run(app, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)