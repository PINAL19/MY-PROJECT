import cv2
import face_recognition
import os
from datetime import datetime

import numpy as np

# Define the path to the directory containing images
path = 'D:/MY PROJECT/mydirectorys'
images = []
classNames = []

# Get the list of files in the specified directory
mylist = os.listdir(path)

# Loop through each file in the directory
for cl in mylist:
    # Check if the file is an image file
    if cl.endswith(('.jpg', '.jpeg', '.png')):
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

# Function to find face encodings in a list of images
def findEncodings(images):
    encodeList = []
    for img in images:
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(img)
            encode_face = face_recognition.face_encodings(img, face_locations)
            if len(encode_face) > 0:
                encodeList.append(encode_face[0])
    return encodeList

# Compute face encodings for the images
encoded_face_train = findEncodings(images)

# Function to mark attendance in CSV file
def markAttendance(name):
    with open('Attendance.csv', 'a') as f:  # Open the file in append mode
        now = datetime.now()
        time = now.strftime('%I:%M:%S:%p')
        date = now.strftime('%d-%B-%Y')
        f.write(f'{name}, {time}, {date}\n')

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            y1, x2, y2, x1 = faceloc
            # Scale back to original size
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
