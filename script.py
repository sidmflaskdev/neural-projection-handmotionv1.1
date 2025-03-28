import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

# Initial circle sizes
left_radius = 50
right_radius = 50
max_radius = 100
min_radius = 30

# Finger tip indexes
tips_ids = [8, 12, 16, 20]

def is_hand_open(landmarks):
    open_fingers = 0
    for tip_id in tips_ids:
        if landmarks.landmark[tip_id].y < landmarks.landmark[tip_id - 2].y:
            open_fingers += 1
    return open_fingers >= 4

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    black_screen = np.zeros((h, w, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            hand_open = is_hand_open(hand_landmarks)

            # Draw hand for debugging
            mp_draw.draw_landmarks(black_screen, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Adjust corresponding circle
            if label == 'Left':
                if hand_open:
                    left_radius = min(left_radius + 3, max_radius)
                else:
                    left_radius = max(left_radius - 3, min_radius)
            elif label == 'Right':
                if hand_open:
                    right_radius = min(right_radius + 3, max_radius)
                else:
                    right_radius = max(right_radius - 3, min_radius)

    # Draw the circles
    cv2.circle(black_screen, (int(w * 0.3), int(h / 2)), left_radius, (255, 0, 255), -1)
    cv2.circle(black_screen, (int(w * 0.7), int(h / 2)), right_radius, (255, 0, 255), -1)

    cv2.imshow("Hand Controlled Circles", black_screen)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
