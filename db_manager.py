import sqlite3
import cv2
import numpy as np
import sys
from utils import initialize_database, save_face_encoding, load_known_faces_from_db

initialize_database()

def add_face(name, image_path, db_path='faces.db'):
    image = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print(f"No faces found in {image_path}")
        return

    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]
    face_encoding = cv2.resize(face, (128, 128)).flatten() / 255.0

    save_face_encoding(name, face_encoding, db_path)
    print(f"Added {name} to the database.")

def list_faces(db_path='faces.db'):
    known_faces = load_known_faces_from_db(db_path)
    if not known_faces:
        return []
    known_face_names = [name for name, _ in known_faces]
    return known_face_names

def remove_face(name, db_path='faces.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('DELETE FROM faces WHERE name=?', (name,))
    conn.commit()
    conn.close()
    print(f"Removed {name} from the database.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python db_manager.py --add <name> <path_to_image>")
        print("       python db_manager.py --list")
        print("       python db_manager.py --remove <name>")
        sys.exit(1)

    command = sys.argv[1]

    if command == '--add':
        if len(sys.argv) != 4:
            print("Usage: python db_manager.py --add <name> <path_to_image>")
            sys.exit(1)
        name = sys.argv[2]
        image_path = sys.argv[3]
        add_face(name, image_path)
    
    elif command == '--list':
        faces = list_faces()
        if faces:
            print("Known faces:")
            for face in faces:
                print(face)
        else:
            print("No faces found in the database.")
    
    elif command == '--remove':
        if len(sys.argv) != 3:
            print("Usage: python db_manager.py --remove <name>")
            sys.exit(1)
        name = sys.argv[2]
        remove_face(name)
        print(f"Removed {name} from the database.")
    
    else:
        print("Unknown command.")
        sys.exit(1)
