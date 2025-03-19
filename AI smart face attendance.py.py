import face_recognition
import cv2
import numpy as np
import csv
import os
import time
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Ask operator to select a folder
def select_image_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    return filedialog.askdirectory(title="Select Folder Containing Images")

image_folder = select_image_folder()

# Check if folder is selected
if not image_folder:
    print("No folder selected. Exiting program.")
    exit()

# Load known images and create encodings
known_face_encodings = []
known_face_names = {}

def load_encoding(image_path):
    """Load face encoding from an image."""
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model="hog")  # Faster than CNN
    encodings = face_recognition.face_encodings(image, face_locations)
    return encodings[0] if encodings else None  # Use only first detected face

# Read images from the selected folder
for file_name in os.listdir(image_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process images
        image_path = os.path.join(image_folder, file_name)
        name = os.path.splitext(file_name)[0]  # Use file name as person's name

        encoding = load_encoding(image_path)
        if encoding is not None:
            known_face_encodings.append(encoding)
            known_face_names[len(known_face_encodings) - 1] = name
            print(f"Loaded: {name}")
        else:
            print(f"Warning: No face detected in {file_name}")

# Check if faces were loaded
if not known_face_encodings:
    print("No valid faces found in the selected folder. Exiting.")
    exit()

# Create CSV file
current_date = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"{current_date}.csv"

# If CSV file doesn't exist, create it with a header
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Name", "Time"])

print("\nAttendance system started. Press 'q' to exit.")

# Initialize camera
video_capture = cv2.VideoCapture(0)

# Track students who haven't been marked yet
students = set(known_face_names.values())

# Face recognition function using threading
def recognize_faces(face_encodings):
    """Threaded function to match face encodings quickly."""
    face_names = []
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(distances)

        if distances[best_match_index] < 0.5:  # Lower threshold for better accuracy
            name = known_face_names.get(best_match_index, "Unknown")
        else:
            name = "Unknown"

        face_names.append(name)
    return face_names

# Use threading for fast recognition
executor = ThreadPoolExecutor(max_workers=4)

frame_count = 0  # Frame skipping counter
while True:
    start_time = time.time()  # Track processing time

    # Capture frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame_count += 1
    if frame_count % 5 != 0:  # Process every 5th frame (for speed)
        continue

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces using HOG
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Run face recognition in a separate thread
    future = executor.submit(recognize_faces, face_encodings)
    face_names = future.result()

    # Mark attendance
    for name in face_names:
        if name in students:
            students.remove(name)
            current_time = datetime.now().strftime("%H:%M:%S")
            with open(csv_filename, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([name, current_time])
            print(f"{name} marked at {current_time}")

    # Display video feed with bounding boxes
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2  # Scale back
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # Show frame
    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
executor.shutdown()
