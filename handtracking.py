import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

line_spec = mpDraw.DrawingSpec(color = (0, 255, 255), thickness = 2)
circle_spec = mpDraw.DrawingSpec(color = (255, 0, 0), thickness = -1, circle_radius = 8)

def count_fingers(hand_landmarks):
    finger_tips_numbers = [4, 8, 12, 16, 20]
    fingers_up = 0

    for tip_id in finger_tips_numbers:
        if tip_id != 4:
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                fingers_up += 1
        else:
            if hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x: # sol
                if hand_landmarks.landmark[4].x > hand_landmarks.landmark[2].x:
                    fingers_up += 1
            else:
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x:
                    fingers_up += 1

    return fingers_up

while True:
    success, image = camera.read()
    image = cv2.flip(image, 1)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        total_fingers = 0

        for handLandmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, handLandmarks, mpHands.HAND_CONNECTIONS, circle_spec, line_spec)
            finger_count = count_fingers(handLandmarks)
            total_fingers += finger_count
    else:
        total_fingers = 0
    
    cv2.putText(image, f"Total Fingers: {total_fingers}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)
    cv2.imshow("Image", image)
    cv2.waitKey(1)
