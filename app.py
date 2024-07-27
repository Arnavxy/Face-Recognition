import sys
print("Python interpreter being used:", sys.executable)  # Print the Python interpreter being used

from flask import Flask, render_template, request, Response
import imageio
import cv2
import numpy as np
import os
from utils import load_config, load_known_faces_from_db, save_face_encoding, initialize_database

app = Flask(__name__)
config = load_config()
initialize_database()

known_faces = load_known_faces_from_db()
known_face_encodings = [encoding for _, encoding in known_faces]
known_face_names = [name for name, _ in known_faces]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + config['model'])

camera_active = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        name = request.form['name']
        file = request.files['file']
        if file:
            file_path = os.path.join('known_faces', file.filename)
            file.save(file_path)
            message = add_face(name, file_path)
            return render_template('upload.html', message=message)
    return render_template('upload.html')

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    if request.method == 'POST':
        name = request.form['name']
        file_path = capture_image_with_imageio(name)
        if file_path:
            message = add_face(name, file_path)
            return render_template('camera.html', message=message)
        else:
            return render_template('camera.html', message="Error: Could not capture image.")
    return render_template('camera.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

def gen_frames():
    global camera_active
    camera_active = True

    try:
        reader = imageio.get_reader('<video0>')
    except Exception as e:
        print(f"Error: {e}")
        camera_active = False
        return

    min_threshold = 20.0  # Lower bound of the threshold range
    max_threshold = 35.0  # Upper bound of the threshold range

    for frame in reader:
        if not camera_active:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (128, 128))  # Increase resolution
            face_encoding = face.flatten() / 255.0

            # Debug information
            print(f"Face Encoding: {face_encoding.shape}")

            matches = []
            for known_face in known_face_encodings:
                if known_face.shape == face_encoding.shape:  # Ensure shapes match
                    matches.append(np.linalg.norm(face_encoding - known_face))

            # Print all match values for debugging
            print(f"Matches: {matches}")
            name = "Unknown"

            if matches:
                # Find the index of the smallest distance (best match)
                best_match_index = np.argmin(matches)
                best_match_value = matches[best_match_index]  # Get the best match value
                print(f"Best Match Index: {best_match_index}, Match Value: {best_match_value}")
                # Adjusted threshold for matching
                if min_threshold <= best_match_value <= max_threshold:  # Adjusted range for threshold
                    name = known_face_names[best_match_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    reader.close()
    print("Camera released.")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    name = request.form['name']
    file_path = capture_image_with_imageio(name)
    if file_path:
        message = add_face(name, file_path)
        return render_template('capture.html', message=message)
    else:
        return render_template('capture.html', message="Error: Could not capture image.")

def capture_image_with_imageio(name):
    try:
        reader = imageio.get_reader('<video0>')
    except Exception as e:
        print(f"Error: {e}")
        return None

    for i, frame in enumerate(reader):
        if i == 10:  # Capture after a few frames to allow camera to adjust
            file_path = os.path.join('known_faces', f"{name}.jpg")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, frame)
            reader.close()  # Ensure the camera resource is released
            return file_path

    reader.close()
    return None

def add_face(name, image_path):
    image = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + config['model'])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return f"No faces found in {image_path}"

    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]
    face_encoding = cv2.resize(face, (128, 128)).flatten() / 255.0  # Increase resolution

    # Debug information
    print(f"Saved Face Encoding for {name}: {face_encoding.shape}")

    save_face_encoding(name, face_encoding)
    global known_face_encodings, known_face_names
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    return f"Added {name} to the database."

@app.route('/release_camera', methods=['GET'])
def release_camera():
    global camera_active
    camera_active = False
    return "Camera released.", 200

if __name__ == '__main__':
    app.run(debug=True)
