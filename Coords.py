import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Euclidean distance
def compute_distance(landmark1, landmark2):
    return ((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)**0.5

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        continue
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Compute and print distances between point 0 and all other points
            landmark0 = hand_landmarks.landmark[0]
            for i, landmark in enumerate(hand_landmarks.landmark[1:], start=1):
                distance = compute_distance(landmark0, landmark)
                print(f"Distance between point 0 and point {i}: {distance:.4f}")
                
                # Print the landmark number on each point
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[1])
                cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display the image
    cv2.imshow("Hand Landmarks with Numbers", image)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
