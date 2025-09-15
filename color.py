import cv2
import mediapipe as mp
import numpy as np
import math

# webcam
webcam = cv2.VideoCapture(0)

# mediapipe hands
my_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

# canvas for drawing
canvas = None

# default brush
brush_color = (0, 0, 255)  # red
brush_thickness = 8
eraser_thickness = 50

# color palette (4 colors: Red, Green, Blue, Eraser)
colors = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Eraser": (0, 0, 0)
}
color_boxes = [(50, 50), (150, 50), (250, 50), (350, 50)]  # x,y for boxes

x1 = y1 = x2 = y2 = 0

while True:
    ret, image = webcam.read()
    if not ret:
        break

    image = cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape

    if canvas is None:
        canvas = np.zeros((frame_height, frame_width, 3), np.uint8)

    # convert to RGB for mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark

            # index fingertip
            x1 = int(landmarks[8].x * frame_width)
            y1 = int(landmarks[8].y * frame_height)

            # thumb tip
            x2 = int(landmarks[4].x * frame_width)
            y2 = int(landmarks[4].y * frame_height)

            # distance between index and thumb
            dist = math.hypot(x2 - x1, y2 - y1)

            # draw line between thumb & index
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # === COLOR SELECTION ===
            if dist < 40:  # pinch gesture
                for i, (cx, cy) in enumerate(color_boxes):
                    if cx < x1 < cx + 80 and cy < y1 < cy + 80:
                        brush_color = list(colors.values())[i]

            # === DRAWING MODE ===
            if y1 > 120:  # avoid touching palette
                if brush_color == (0, 0, 0):  # eraser
                    cv2.circle(canvas, (x1, y1), eraser_thickness, (0, 0, 0), -1)
                else:
                    cv2.circle(canvas, (x1, y1), brush_thickness, brush_color, -1)

    # overlay canvas on webcam feed
    combined = cv2.addWeighted(image, 0.7, canvas, 0.8, 0)

    # draw color palette
    for i, (cx, cy) in enumerate(color_boxes):
        color = list(colors.values())[i]
        cv2.rectangle(combined, (cx, cy), (cx + 80, cy + 80), color, -1)
        cv2.putText(combined, list(colors.keys())[i], (cx + 5, cy + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Drawing + Color Picker", combined)

    if cv2.waitKey(1)==ord('q'):  # ESC to quit
        break

webcam.release()
cv2.destroyAllWindows()
