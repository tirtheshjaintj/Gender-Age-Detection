import cv2
import numpy as np
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Load the face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Load pre-trained age and gender models
age_net = cv2.dnn.readNetFromCaffe('models/deploy_age.prototxt', 'models/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('models/deploy_gender.prototxt', 'models/gender_net.caffemodel')

# Age and gender labels
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Initialize statistics
gender_stats = {'Male': 0, 'Female': 0}
emotion_stats = {'Happy': 0, 'Not Happy': 0}
age_data = []

# Directory to save video
video_dir = 'videos'
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

# Log file for storing data
log_file = 'detected_data.csv'
log_data = []

# Function to list available cameras
def list_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Function to plot gender distribution
def plot_gender_distribution():
    plt.clf()  # Clear the current figure
    plt.bar(gender_stats.keys(), gender_stats.values())
    plt.title('Gender Distribution')
    plt.pause(0.1)  # To update the plot dynamically

# List all available cameras
available_cameras = list_available_cameras()
if not available_cameras:
    print("No cameras found.")
    exit()

# Display available cameras and get user input
print("Available cameras:")
for index in available_cameras:
    print(f"Camera index: {index}")

camera_index = int(input("Choose a camera index from the above list: "))
if camera_index not in available_cameras:
    print("Error: Invalid camera index selected.")
    exit()

# Create a VideoCapture object
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get frame width and height for video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video codec and create VideoWriter object
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_file_path = os.path.join(video_dir, f'video_{timestamp}.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_file_path, fourcc, 20.0, (frame_width, frame_height))

# Adjustable parameters for face detection
scale_factor = float(input("Enter scale factor for face detection (e.g., 1.1): "))
min_neighbors = int(input("Enter minimum neighbors for face detection (e.g., 5): "))

# Prepare the plot for gender distribution
plt.ion()  # Enable interactive mode for live plotting
plt.figure()

# Loop to continuously capture frames
while True:
    try:
        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to capture image.")
    except Exception as e:
        print(f"Error: {e}")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))

    # Reset gender counts for each frame
    gender_counts = {'Male': 0, 'Female': 0}

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_roi = frame[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1.0, size=(227, 227), 
                                      mean=(78.4263377603, 87.7689143744, 114.895847746), 
                                      swapRB=False, crop=False)

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        gender_counts[gender] += 1  # Update gender counts for current frame
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]
        
        age_data.append(age)  # Store age data

        # Detect smiles
        smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=25, minSize=(25, 25))
        if len(smiles) > 0:
            emotion_stats['Happy'] += 1  # Count as happy
            label = f"{gender}, {age}, Emotion: Happy"
        else:
            emotion_stats['Not Happy'] += 1  # Count as not happy
            label = f"{gender}, {age}, Emotion: Not Happy"

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Update total gender stats based on the current frame
    gender_stats['Male'] += gender_counts['Male']
    gender_stats['Female'] += gender_counts['Female']

    # Calculate total people detected
    total_people = gender_counts['Male'] + gender_counts['Female']

    # Log data
    for (x, y, w, h) in faces:
        log_data.append([datetime.datetime.now(), gender, age])

    # Display statistics on the frame
    stats_text = (f"Total People: {total_people}")
    cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Write the frame to the video file
    video_writer.write(frame)

    # Update the gender distribution plot
    plot_gender_distribution()

    cv2.imshow('Age & Gender Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        confirm_exit = input("Are you sure you want to quit? (y/n): ")
        if confirm_exit.lower() == 'y':
            break

# Save log data to CSV
log_df = pd.DataFrame(log_data, columns=['Timestamp', 'Gender', 'Age'])
log_df.to_csv(log_file, index=False)

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()
plt.ioff()  # Disable interactive mode
plt.show()  # Show the final plot
