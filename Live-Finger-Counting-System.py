import site

site.addsitedir(r'C:\Users\madha\AppData\Roaming\Python\Python38\site-packages')

import cv2
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Finger tip and PIP joint indices
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        finger_count = 0

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                landmarks = hand_landmarks.landmark

                # Thumb
                if landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_PIPS[0]].x:
                    finger_count += 1

                # Other fingers
                for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
                    if landmarks[tip].y < landmarks[pip].y:
                        finger_count += 1

        cv2.rectangle(frame, (0, 0), (250, 60), (0, 0, 0), -1)

        cv2.putText(
            frame,
            f'Fingers: {finger_count}',
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.imshow("Hand & Finger Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

cap.release()
cv2.destroyAllWindows()