"""
Rover WebSocket blueprint
Handles real rover telemetry streaming via Flask-SocketIO.
Falls back to browser-side simulation when no rover is connected.

Install: pip install flask-socketio eventlet
"""
from flask import Blueprint, request, jsonify
from flask_socketio import emit, join_room, leave_room
import threading
import time

rover_api = Blueprint('rover_api', __name__, url_prefix='/api/rover')

# Shared state — tracks which rover clients are connected
_connected_rovers = set()
_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────
# HTTP ENDPOINTS
# ─────────────────────────────────────────────────────────────

@rover_api.route('/status')
def status():
    """Dashboard polls this to know if a real rover is connected."""
    with _lock:
        rover_count = len(_connected_rovers)
    return jsonify({
        'connected':    rover_count > 0,
        'rover_count':  rover_count,
        'mode':         'live' if rover_count > 0 else 'simulation'
    })


# ─────────────────────────────────────────────────────────────
# SOCKETIO EVENTS
# Register these in app.py after socketio = SocketIO(app)
# ─────────────────────────────────────────────────────────────

def register_rover_events(socketio):
    """
    Call this in app.py:
        from routes.rover import register_rover_events
        register_rover_events(socketio)
    """

    @socketio.on('join_dashboard')
    def on_join_dashboard():
        """Dashboard client joins the dashboard room to receive rover updates"""
        join_room('dashboard')
        print('[Dashboard] Client joined dashboard room')

    @socketio.on('rover_connect')
    def on_rover_connect(data):
        """
        Rover hardware emits this on connection.
        data = { 'rover_id': 'ROVER-01', 'token': '...' }
        """
        rover_id = data.get('rover_id', 'unknown')
        join_room('rovers')
        with _lock:
            _connected_rovers.add(rover_id)
        emit('ack', {'status': 'connected', 'rover_id': rover_id})
        # Notify all dashboard clients
        socketio.emit('rover_status', {
            'connected': True,
            'rover_id':  rover_id,
            'mode':      'live'
        }, room='dashboard')
        print(f'[Rover] {rover_id} connected')

    @socketio.on('rover_disconnect')
    def on_rover_disconnect(data):
        rover_id = data.get('rover_id', 'unknown')
        leave_room('rovers')
        with _lock:
            _connected_rovers.discard(rover_id)
        socketio.emit('rover_status', {
            'connected': len(_connected_rovers) > 0,
            'mode':      'live' if _connected_rovers else 'simulation'
        }, room='dashboard')
        print(f'[Rover] {rover_id} disconnected')

    @socketio.on('rover_telemetry')
    def on_rover_telemetry(data):
        """
        Rover hardware streams telemetry packets here.

        Expected data shape:
        {
            'rover_id':  'ROVER-01',
            'depth_m':   42.3,
            'temp_c':    24.1,
            'sal_psu':   34.8,
            'pres_bar':  5.23,
            'oxy_mgl':   6.4,
            'turb_ntu':  1.2,
            'lidar': [
                { 'x': 1.2, 'y': -40.1, 'z': 3.4 },
                ...
            ]
        }
        """
        # Broadcast to all dashboard clients
        print(f'[Rover] Broadcasting telemetry to dashboard: {data}')
        socketio.emit('rover_update', data, room='dashboard')
        print(f'[Rover] Telemetry sent')

    @socketio.on('dashboard_join')
    def on_dashboard_join():
        """Frontend emits this on page load to receive rover broadcasts."""
        join_room('dashboard')
        with _lock:
            is_live = len(_connected_rovers) > 0
        emit('rover_status', {
            'connected': is_live,
            'mode':      'live' if is_live else 'simulation'
        })