import logging
import yaml
import sqlite3
import numpy as np

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger()

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_database(db_path='faces.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def load_known_faces_from_db(db_path='faces.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT name, encoding FROM faces')
    known_faces = c.fetchall()
    conn.close()

    known_face_encodings = []
    known_face_names = []
    for name, encoding_blob in known_faces:
        face_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
    
    return list(zip(known_face_names, known_face_encodings))

def save_face_encoding(name, face_encoding, db_path='faces.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    encoding_blob = sqlite3.Binary(face_encoding.astype(np.float64).tobytes())
    c.execute('INSERT INTO faces (name, encoding) VALUES (?, ?)', (name, encoding_blob))
    conn.commit()
    conn.close()
