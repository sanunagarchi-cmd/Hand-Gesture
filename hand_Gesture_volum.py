import cv2
import mediapipe as mp
import pyautogui
import math


x1 = y1 = x2 = y2 = 0
webcam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

while True:
    ret, image = webcam.read()
    if not ret:
        break

    image = cv2.flip(image, 1)  # mirror image
    frame_height, frame_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image,hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x=int(landmark.x * frame_width)
                y=int(landmark.y * frame_height)
                if id == 8:
                    cv2.circle(image,(x,y),8,(0,255,255),-1)
                    x1, y1 = x, y
                if id == 4:
                    cv2.circle(image,(x,y),8,(0,0,255),-1)
                    x2, y2 = x, y

                dist = math.hypot(x2 - x1, y2 - y1)
                cv2.line(image,(x1,y1),(x2,y2),(0,255,0),5)
                if dist > 50:
                    pyautogui.press("volumeup")
                elif dist < 30:
                    pyautogui.press("volumedown")

    cv2.imshow("Hand volume control using python", image)

    if cv2.waitKey(10)==ord('q'): # ✅ fixed (was waitkey instead of waitKey)
        break
webcam.release()
cv2.destroyAllWindows()