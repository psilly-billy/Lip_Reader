import cv2
import dlib
import os
import json
from utils import load_data_2, num_to_char
from modelutil import load_model
import tensorflow as tf


# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the codec and initialize video writer at 25 FPS
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
if not os.path.exists('recorded_video'):
    os.makedirs('recorded_video')
out = cv2.VideoWriter('recorded_video/output.mp4', fourcc, 25.0, (640,480)) 

# Dictionary to hold the lip coordinates
lip_coordinates = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # Loop through detected faces
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Get the coordinates of the lips and save to the dictionary
        lip_coordinates = {'x1': landmarks.part(48).x, 'y1': landmarks.part(50).y, 
                            'x2': landmarks.part(54).x, 'y2': landmarks.part(57).y}

        # Draw a rectangle around the lips on the frame for visualization purpose
        frame_vis = frame.copy()
        cv2.rectangle(frame_vis, (lip_coordinates['x1'], lip_coordinates['y1']), (lip_coordinates['x2'], lip_coordinates['y2']), (0, 255, 0), 2)

    # Write the original frame (without visualization rectangle) to file
    out.write(frame)

    # Display the frame with rectangle around lips
    cv2.imshow("Frame", frame_vis)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
out.release()
cv2.destroyAllWindows()

# Save the lip coordinates to a json file
with open('recorded_video/lip_coordinates.json', 'w') as file:
    json.dump(lip_coordinates, file)


# Load the lip coordinates from the JSON file
with open('recorded_video/lip_coordinates.json', 'r') as file:
    lip_coordinates = json.load(file)

# Path to the video file
path = 'recorded_video/output.mp4'


# Load the video data
frames = load_data_2(path, lip_coordinates)
print(frames)

print(frames.shape)  # Should be (None, 75, 46, 140, 1)

model = load_model()
yhat = model.predict(tf.expand_dims(frames, axis=-1))
decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
print(converted_prediction)