"""
Virtual rover simulator - emits telemetry to Flask app via WebSocket
Generates realistic LIDAR sweep data as rover descends
"""
import socketio
import time
import random
import math

sio = socketio.Client()

# Rover state
rover_depth = 0.0
lidar_angle = 0.0

@sio.event
def connect():
    print("[Simulator] Connected to Flask app")
    sio.emit('rover_connect', {
        'rover_id': 'ROVER-SIM-01',
        'token': 'simulator'
    })

@sio.on('ack')
def on_ack(data):
    print(f"[Simulator] Server ack: {data}")

def seafloor_height(x, z, base_depth):
    """Generate realistic seafloor topography with features"""
    height = base_depth + 200  # Seafloor ~200m below rover
    
    # Large-scale terrain variation
    height += math.sin(x * 0.02) * 30
    height += math.cos(z * 0.025) * 25
    
    # Medium-scale hills and valleys
    height += math.sin(x * 0.08) * 15
    height += math.cos(z * 0.07) * 12
    
    # Small-scale rocks and roughness
    height += math.sin(x * 0.3) * math.cos(z * 0.3) * 5
    
    # Random noise for natural variation
    height += (random.random() - 0.5) * 8
    
    return height

def is_coral_formation(x, z):
    """Detect potential coral/rock formations for vertical features"""
    # Create clustered formations
    formations = [
        {'x': 0, 'z': 0, 'radius': 8},
        {'x': 30, 'z': 20, 'radius': 6},
        {'x': -25, 'z': 35, 'radius': 7},
        {'x': 15, 'z': -30, 'radius': 5},
        {'x': -35, 'z': -15, 'radius': 8},
    ]
    
    for formation in formations:
        dist = math.sqrt((x - formation['x'])**2 + (z - formation['z'])**2)
        if dist < formation['radius']:
            return True, formation
    return False, None

def generate_lidar_sweep(rover_x, rover_y, rover_z):
    """
    Generate a 360° LIDAR sweep with realistic underwater terrain featuring:
    - Coral formations (tall structures)
    - Ridges and valleys
    - Complex seafloor topology
    """
    points = []
    num_rays = 64  # More rays for denser coverage
    vertical_angles = 8  # More vertical angles for better detail
    
    # Seafloor is approximately 200m below rover
    base_floor_depth = rover_y - 200  
    
    for ray_idx in range(num_rays):
        angle_deg = (ray_idx / num_rays) * 360
        angle_rad = math.radians(angle_deg)
        
        # Scan at multiple vertical angles (looking down mostly)
        for elev_idx in range(vertical_angles):
            # Elevation from -50° (down) to +40° (up)
            elev_angle = -50 + (elev_idx * 12.86)
            elev_rad = math.radians(elev_angle)
            
            # Ray direction
            dx = math.cos(angle_rad) * math.cos(elev_rad)
            dz = math.sin(angle_rad) * math.cos(elev_rad)
            dy = math.sin(elev_rad)
            
            # Cast ray at multiple ranges to create depth
            for distance in range(5, 60, 4):  # Denser range sampling
                # Ray endpoint
                hit_x = rover_x + dx * distance
                hit_y = rover_y + dy * distance
                hit_z = rover_z + dz * distance
                
                # Multi-scale terrain generation for realistic features
                terrain_var = 0
                
                # Scale 1: Very large features (continental scale)
                terrain_var += math.sin(hit_x * 0.01) * 40
                terrain_var += math.cos(hit_z * 0.012) * 35
                
                # Scale 2: Large features (ridges, valleys)
                terrain_var += math.sin(hit_x * 0.04) * 25
                terrain_var += math.cos(hit_z * 0.035) * 20
                
                # Scale 3: Medium features (hills, slopes)
                terrain_var += math.sin(hit_x * 0.12) * 12
                terrain_var += math.cos(hit_z * 0.1) * 10
                
                # Scale 4: Small features (rocks, details)
                terrain_var += math.sin(hit_x * 0.3) * math.cos(hit_z * 0.35) * 6
                
                # Scale 5: Micro-roughness
                terrain_var += math.sin(hit_x * 0.8) * math.cos(hit_z * 0.7) * 2
                
                # Random noise
                terrain_var += (random.random() - 0.5) * 3
                
                floor_y = base_floor_depth + terrain_var
                
                # If point is between rover and seafloor, record it
                if hit_y < rover_y and hit_y > floor_y:
                    # Add point with minimal noise to preserve structure
                    points.append({
                        'x': hit_x + (random.random() - 0.5) * 0.8,
                        'y': hit_y + (random.random() - 0.5) * 0.8,
                        'z': hit_z + (random.random() - 0.5) * 0.8
                    })
                    
                    # Coral formations - more varied and realistic
                    is_coral, formation = is_coral_formation(hit_x, hit_z)
                    if is_coral:
                        # Tall coral structures (10-40m high depending on location)
                        coral_base_height = 15 + (random.random() * 25)
                        for coral_height in range(0, int(coral_base_height), 2):
                            coral_y = floor_y + coral_height + (random.random() * 1.5)
                            if rover_y > coral_y > floor_y:
                                # Multiple points per coral height for structure
                                for _ in range(2):
                                    points.append({
                                        'x': hit_x + (random.random() - 0.5) * 4,
                                        'y': coral_y,
                                        'z': hit_z + (random.random() - 0.5) * 4
                                    })
                    break  # Stop ray at first intersection
    
    return points

def generate_telemetry(depth):
    """Generate realistic sensor data based on depth"""
    # Temperature decreases with depth (thermocline effect)
    if depth < 30:
        temp = 28 - depth * 0.07 + random.gauss(0, 0.3)
    elif depth < 200:
        temp = 26 - ((depth - 30) / 170) * 17 + random.gauss(0, 0.3)
    else:
        temp = 9 - ((depth - 200) / 100) * 5 + random.gauss(0, 0.3)
    
    # Salinity increases slightly with depth
    sal = 34.0 + (depth / 1000) * 2.2 + random.gauss(0, 0.1)
    
    # Pressure increases with depth
    pres = 1 + depth / 10.0
    
    # Dissolved oxygen varies with depth (oxygen minimum zone ~500-1500m)
    oxy = max(2, 8.5 - (depth / 200) * 3.5 + random.gauss(0, 0.3))
    
    # Turbidity decreases with depth
    turb = max(0.1, 1.5 - (depth / 200) + random.gauss(0, 0.2))
    
    return {
        'temp_c': round(temp, 2),
        'sal_psu': round(sal, 2),
        'pres_bar': round(pres, 2),
        'oxy_mgl': round(oxy, 2),
        'turb_ntu': round(turb, 2)
    }

if __name__ == '__main__':
    sio.connect('http://localhost:5000')
    
    try:
        start_time = time.time()
        
        while True:
            # Simulate descent at ~0.5 m/s
            elapsed = time.time() - start_time
            rover_depth = min(300, elapsed * 0.5)
            
            # Generate sensor telemetry
            sensor_data = generate_telemetry(rover_depth)
            
            # Generate LIDAR sweep every packet (every 1 second)
            lidar_points = generate_lidar_sweep(0, -rover_depth, 0)
            
            # Build telemetry packet
            telemetry = {
                'rover_id': 'ROVER-SIM-01',
                'depth_m': round(rover_depth, 2),
                'temp_c': sensor_data['temp_c'],
                'sal_psu': sensor_data['sal_psu'],
                'pres_bar': sensor_data['pres_bar'],
                'oxy_mgl': sensor_data['oxy_mgl'],
                'turb_ntu': sensor_data['turb_ntu'],
                'lidar': lidar_points
            }
            
            # Send telemetry
            sio.emit('rover_telemetry', telemetry)
            print(f"[Simulator] Sent: depth={telemetry['depth_m']:.1f}m, "
                  f"temp={telemetry['temp_c']:.1f}°C, "
                  f"lidar_points={len(lidar_points)}")
            
            time.sleep(1)  # Send every second
            
            # Stop at max depth
            if rover_depth >= 300:
                print("[Simulator] Max depth reached, stopping")
                break
                
    except KeyboardInterrupt:
        print("[Simulator] Shutting down")
        sio.emit('rover_disconnect', {'rover_id': 'ROVER-SIM-01'})
        sio.disconnect()
