import sys
import os

# Add local site-packages to path
local_packages = os.path.join(os.path.dirname(__file__), 'site-packages')
if local_packages not in sys.path:
    sys.path.insert(0, local_packages)

from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
import base64
import numpy as np
from camera import FaceMorpher

USE_SERVER_CAMERA = os.environ.get("USE_SERVER_CAMERA", "false").lower() == "true"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

morpher = FaceMorpher()
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def gen_frames():
    cap = get_camera()
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera. Please verify camera permissions and index.", flush=True)
            break
        
        # Process frame
        processed_frame = morpher.process_frame(frame)
        if processed_frame is None:
            continue
            
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

round_counter = 0

def start_round():
    global round_counter
    # Even rounds = Mona Lisa (starting at 0)
    # Odd rounds = Da Vinci
    if round_counter % 2 == 0:
        gallery_path = os.path.join(app.root_path, 'static', 'gallery', 'MONA LISA.png')
        if not os.path.exists(gallery_path):
            gallery_path = os.path.join(app.root_path, 'MONA LISA.png')
    else:
        gallery_path = os.path.join(app.root_path, 'static', 'gallery', 'DAVICI.png')
        if not os.path.exists(gallery_path):
            gallery_path = os.path.join(app.root_path, 'DAVICI.png')
            
    morpher.set_reference(gallery_path)

@app.route('/')
def index():
    global round_counter
    round_counter = 0
    start_round()
    return redirect(url_for('experience'))

@app.route('/autostart')
def autostart():
    # Map to the new standard flow
    return redirect(url_for('index'))

@app.route('/experience')
def experience():
    return render_template('experience.html', use_server_camera=USE_SERVER_CAMERA)

@app.route('/video_feed')
def video_feed():
    if not USE_SERVER_CAMERA:
        return "Server camera disabled", 403
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frontend_frame():
    if USE_SERVER_CAMERA:
        return "Frontend camera disabled", 403
        
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"image": None}), 400
        
    try:
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Process frame
        processed_frame = morpher.process_frame(frame)
        if processed_frame is None:
            # Fall back to original frame if no face is detected or during loading
            processed_frame = frame
            
        _, buffer = cv2.imencode('.jpg', processed_frame)
        encoded = base64.b64encode(buffer).decode("utf-8")
        return jsonify({"image": f"data:image/jpeg;base64,{encoded}"})
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print("Error processing frame:", err_msg)
        return jsonify({"image": None, "error": str(e), "trace": err_msg})

@app.route('/status')
def get_status():
    return {"complete": morpher.sequence_complete}

@app.route('/restart')
def restart():
    global round_counter
    round_counter += 1
    start_round()
    return redirect(url_for('experience'))

@app.route('/grid_result')
def get_grid_result():
    if morpher.grid_image is not None:
        ret, buffer = cv2.imencode('.jpg', morpher.grid_image)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return "Not ready", 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=True
    )
