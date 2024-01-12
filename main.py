import cv2
import mediapipe as mp
from urllib import request
import numpy as np
import keyboard

# Set the volume change step
VOLUME_CHANGE_STEP = 5

def calculate_distance(hand_landmarks, point1, point2):
    x1, y1 = hand_landmarks.landmark[point1].x, hand_landmarks.landmark[point1].y
    x2, y2 = hand_landmarks.landmark[point2].x, hand_landmarks.landmark[point2].y
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance

def change_volume(hand_landmarks):
    # Check if the required landmarks are present
    if len(hand_landmarks.landmark) >= 9:
        # Calculate the distance between thumb (point 4) and index finger (point 8)
        distance_8_4 = calculate_distance(hand_landmarks, 8, 4)

        # Adjust volume based on hand movement
        if distance_8_4 < 0.1:  # Adjust the threshold as needed
            keyboard.press_and_release('ctrl+down')  # Decrease volume
            print("down")
        elif distance_8_4 > 0.3:  # Adjust the threshold as needed
            keyboard.press_and_release('ctrl+up')  # Increase volume
            print("up")

def track_hand(camera_url):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Open a window with size 800x600
    cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Tracking', 800, 600)

    while True:
        # Fetch image from the camera URL
        with request.urlopen(camera_url) as response:
            img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)

        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Change volume based on hand gesture
                change_volume(hand_landmarks)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils

    # Specify the camera URL
    camera_url = 'http://192.168.1.4:8080/shot.jpg'

    track_hand(camera_url)

