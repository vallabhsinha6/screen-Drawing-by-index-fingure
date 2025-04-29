import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)
    drawing = False
    canvas = None
    prev_x, prev_y = None, None

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            if canvas is None:
                canvas = np.zeros_like(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the index finger tip position
                h, w, _ = frame.shape
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Check if index finger is up (simple heuristic)
                index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                if index_finger_tip.y < index_finger_mcp.y:
                    drawing = True
                else:
                    drawing = False
                    prev_x, prev_y = None, None  

                if drawing:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 8)
                    prev_x, prev_y = x, y

            # Combine frame and canvas
            combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

            cv2.putText(combined, "Press 'c' to clear, 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Hand Gesture Drawing", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                canvas = np.zeros_like(frame)
                prev_x, prev_y = None, None
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
