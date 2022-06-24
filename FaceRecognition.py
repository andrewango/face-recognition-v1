import cv2

# Load pre-trained data on face frontals from OpenCV (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Assign our default webcam as the video
webcam = cv2.VideoCapture(0)

# LOOP OUR DETECTION ON EACH VIDEO FRAME
while (True):
    
    # Read current frame - .read() returns boolean and frame image
    successful_frame_read, frame = webcam.read()

    # Grayscale our frame, machine just learns facial features/relations
    bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use pre-trained algorithm to detect our faces in different sizes
    face_coordinates = trained_face_data.detectMultiScale(bw_frame)
    # print(face_coordinates)

    for (x, y, w, h) in face_coordinates:
        # Draw rectangles around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    # Show our image in a program window
    cv2.imshow('Face Recognition Program', frame)

    # Wait until key is pressed to update our image, ensures user sees it
    # Parameter (1) infinitely updates our image every millisecond
    key = cv2.waitKey(1)

    if key == 13:
        break

# Deallocate webcam video data
webcam.release()

"""
# Image to detect faces with
img = cv2.imread('RMHat.jpg')
img = cv2.imread('RMCoverFace.jpg')
"""

# Make sure our code executed
print("Code completed!")
