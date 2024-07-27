import cv2
import sys
import numpy as np
from utils import setup_logging, load_config, load_known_faces_from_db

logger = setup_logging()
config = load_config()

known_faces = load_known_faces_from_db()
known_face_encodings = [encoding for _, encoding in known_faces]
known_face_names = [name for name, _ in known_faces]

# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + config['model'])

def recognize_faces(image_path, output_path='output.jpg'):
    """
    Recognizes faces in an image using OpenCV and labels them.

    Parameters:
    image_path (str): Path to the input image.
    output_path (str): Path to save the output image.

    Returns:
    None
    """
    try:
        logger.info(f"Loading image from {image_path}")
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except FileNotFoundError:
        logger.error(f"The file {image_path} was not found.")
        return
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return

    logger.info("Detecting faces")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_encoding = cv2.resize(face, (128, 128)).flatten() / 255.0

        matches = [np.linalg.norm(face_encoding - known_face) for known_face in known_face_encodings]
        name = "Unknown"

        if matches:
            best_match_index = np.argmin(matches)
            if matches[best_match_index] < 0.6:
                name = known_face_names[best_match_index]

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    logger.info(f"Saving output image to {output_path}")
    cv2.imwrite(output_path, image)

    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def recognize_faces_from_camera():
    """
    Recognizes faces in real-time from the webcam using OpenCV and labels them.

    Returns:
    None
    """
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_encoding = cv2.resize(face, (128, 128)).flatten() / 255.0

            matches = [np.linalg.norm(face_encoding - known_face) for known_face in known_face_encodings]
            name = "Unknown"

            if matches:
                best_match_index = np.argmin(matches)
                if matches[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python face_recognition_opencv.py <path_to_image> or python face_recognition_opencv.py --camera")
    else:
        if sys.argv[1] == "--camera":
            recognize_faces_from_camera()
        else:
            image_path = sys.argv[1]
            recognize_faces(image_path, output_path=config['output_path'])
