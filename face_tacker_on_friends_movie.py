# -*- coding: utf-8 -*-

#!pip install face_recognition

import face_recognition
import cv2
from IPython.display import HTML
from base64 import b64encode

PATH = 'friends_files/'

# Open the input movie file
input_movie = cv2.VideoCapture(PATH+"friends_movie.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output_movie.avi', fourcc, 23, (1280, 720))

# Load pictures if the characters, which will be used to to recognize them.
c1_image = face_recognition.load_image_file(PATH+"chandler_bing.png")
c1_face_encoding = face_recognition.face_encodings(c1_image)[0]

c2_image = face_recognition.load_image_file(PATH+"monica_geller.jpeg")
c2_face_encoding = face_recognition.face_encodings(c2_image)[0]

c3_image = face_recognition.load_image_file(PATH+"phoebe_buffet.jpg")
c3_face_encoding = face_recognition.face_encodings(c3_image)[0]

c4_image = face_recognition.load_image_file(PATH+"rachel_green.jpg")
c4_face_encoding = face_recognition.face_encodings(c4_image)[0]

c5_image = face_recognition.load_image_file(PATH+"ross_geller.jpg")
c5_face_encoding = face_recognition.face_encodings(c5_image)[0]

c6_image = face_recognition.load_image_file(PATH+"joey_tribbiani.jpg")
c6_face_encoding = face_recognition.face_encodings(c6_image)[0]

character_faces = [
    c1_face_encoding,
    c2_face_encoding,
    c3_face_encoding,
    c4_face_encoding,
    c5_face_encoding,
    c6_face_encoding]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # read a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # break when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # compare the face to the our known faces

        match = face_recognition.compare_faces(character_faces, face_encoding, tolerance=0.50)

        name = None
        if match[0]:
            name = "Chandler Bing"
        elif match[1]:
            name = "Monica Geller"
        elif match[2]:
            name = "Phoebe Buffet"
        elif match[3]:
            name = "Rachel Green"
        elif match[4]:
            name = "Ross Geller"
        elif match[5]:
            name = "Joey Tribbiani"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

!ffmpeg -i output_movie.avi output_movie.mp4 -y

# display video on colab notebook
mp4 = open('output_movie.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
