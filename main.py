import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

success, frame = cap.read()

if not success:
    print("Camera not found!")
    exit()

frame = cv2.flip(frame, 1)

# Canvas
canvas = np.zeros_like(frame)

# Variables
prev_x = 0
prev_y = 0

draw_color = (255, 0, 255)  # Purple
brush_thickness = 5


def index_up(hand):
    return hand.landmark[8].y < hand.landmark[6].y


def middle_up(hand):
    return hand.landmark[12].y < hand.landmark[10].y


while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Toolbar
    cv2.rectangle(frame, (10, 10), (80, 60), (255, 0, 255), -1)      # Purple
    cv2.rectangle(frame, (90, 10), (160, 60), (255, 0, 0), -1)       # Blue
    cv2.rectangle(frame, (170, 10), (240, 60), (0, 255, 0), -1)      # Green
    cv2.rectangle(frame, (250, 10), (320, 60), (0, 0, 255), -1)      # Red
    cv2.rectangle(frame, (330, 10), (400, 60), (255, 255, 255), -1)  # Eraser

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape

            # Index Finger
            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            # Thumb Finger
            thumb_x = int(hand.landmark[4].x * w)
            thumb_y = int(hand.landmark[4].y * h)

            # Brush Size Control
            distance = math.hypot(
                thumb_x - x,
                thumb_y - y
            )

            brush_thickness = int(distance / 5)
            brush_thickness = max(2, min(30, brush_thickness))

            # Show brush size
            cv2.putText(
                frame,
                f"Brush: {brush_thickness}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Visual line between thumb & index
            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (x, y),
                (0, 255, 255),
                2
            )

            cv2.circle(
                frame,
                (x, y),
                12,
                (255, 0, 255),
                -1
            )

            idx = index_up(hand)
            mid = middle_up(hand)

            # DRAW MODE
            if idx and not mid:

                cv2.putText(
                    frame,
                    "DRAW MODE",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                if prev_x == 0 and prev_y == 0:
                    prev_x = x
                    prev_y = y

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    draw_color,
                    brush_thickness
                )

                prev_x = x
                prev_y = y

            # SELECT MODE
            elif idx and mid:

                cv2.putText(
                    frame,
                    "SELECT MODE",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                prev_x = 0
                prev_y = 0

                if y < 60:

                    if 10 < x < 80:
                        draw_color = (255, 0, 255)

                    elif 90 < x < 160:
                        draw_color = (255, 0, 0)

                    elif 170 < x < 240:
                        draw_color = (0, 255, 0)

                    elif 250 < x < 320:
                        draw_color = (0, 0, 255)

                    elif 330 < x < 400:
                        draw_color = (0, 0, 0)

            else:
                prev_x = 0
                prev_y = 0

    result = cv2.add(frame, canvas)

    cv2.imshow("Gesture Whiteboard", result)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas[:] = 0

    if key == ord('s'):
        cv2.imwrite("drawing.png", canvas)
        print("Drawing Saved!")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()