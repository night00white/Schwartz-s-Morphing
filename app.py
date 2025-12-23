import sys
import os

# Add local site-packages to path
local_packages = os.path.join(os.path.dirname(__file__), 'site-packages')
if local_packages not in sys.path:
    sys.path.insert(0, local_packages)

from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from camera import FaceMorpher

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
            break
        
        # Process frame
        processed_frame = morpher.process_frame(frame)
        if processed_frame is None:
            continue
            
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('intro.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # 1. Check for uploaded file
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'reference.jpg')
            file.save(filepath)
            morpher.set_reference(filepath)
            return redirect(url_for('experience'))

    # 2. Check for gallery selection
    if 'selected_image' in request.form and request.form['selected_image']:
        sel_img = request.form['selected_image']
        # Securely construct path
        gallery_path = os.path.join(app.root_path, 'static', 'gallery', sel_img)
        if os.path.exists(gallery_path):
            print(f"Using gallery image: {gallery_path}")
            morpher.set_reference(gallery_path)
    
    # Regardless, go to experience
    return redirect(url_for('experience'))

@app.route('/experience')
def experience():
    return render_template('experience.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=8080, use_reloader=False) # use_reloader=False to avoid double camera open
