from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.conf import settings
from imutils import face_utils
from scipy.spatial import distance
import cv2
import dlib
import imutils
import os
import pygame
import pyttsx3

# Initialize pygame
pygame.mixer.init()

# Initialize pyttsx3
engine = pyttsx3.init()

# Set voice properties for female voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Index 1 usually corresponds to a female voice

# Load alert sound
alert_sound = pygame.mixer.Sound(os.path.join(settings.BASE_DIR, 'drowsiness_detection_app', 'alert.wav'))

# Initialize dlib detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(settings.BASE_DIR, 'drowsiness_detection_app', 'shape_predictor_68_face_landmarks.dat'))

# Define facial landmarks indices
(l_start, l_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Global variables for monitoring dashboard
total_frames_processed = 0
drowsiness_alerts_triggered = 0

def calculate_eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def gen_frames():
    global total_frames_processed, drowsiness_alerts_triggered

    cap = cv2.VideoCapture(0)
    flag = 0
    alert_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames_processed += 1

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[l_start:l_end]
            right_eye = shape[r_start:r_end]
            left_ear = calculate_eye_aspect_ratio(left_eye)
            right_ear = calculate_eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < 0.25:
                flag += 1
                if flag >= 20 and not alert_active:
                    # Increment drowsiness alerts counter
                    drowsiness_alerts_triggered += 1
                    # Play alert sound
                    alert_sound.play()
                    # Speak alert message
                    engine.say("You are drowsy, please wake up!")
                    engine.runAndWait()
                    alert_active = True
            else:
                flag = 0
                if alert_active:
                    # Stop alert sound
                    alert_sound.stop()
                    alert_active = False

            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def home(request):
    global total_frames_processed, drowsiness_alerts_triggered
    context = {
        'total_frames_processed': total_frames_processed,
        'drowsiness_alerts_triggered': drowsiness_alerts_triggered
    }
    return render(request, 'home.html', context)

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def alert_feed(request):
    return JsonResponse({'message': ''})  
