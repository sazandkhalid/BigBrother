import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# Function to detect if the mouth is closed
def detect_closed_mouth(landmarks):
    upper_lip_bottom = landmarks[62]  # Bottom point of the upper lip
    lower_lip_top = landmarks[187]  # Top point of the lower lip
    left_eye = landmarks[33]  # Left eye midpoint
    right_eye = landmarks[263]  # Right eye midpoint

    # Calculate the distance between upper and lower lips
    lip_distance = euclidean_distance(upper_lip_bottom, lower_lip_top)

    # Calculate the distance between the midpoint of the eyes
    eye_distance = euclidean_distance(left_eye, right_eye)

    # Calculate the ratio of lip distance to eye distance
    lip_eye_ratio = lip_distance / eye_distance

    #.495
    return lip_eye_ratio < 0.482  # Adjust threshold as needed

# Main function
def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw facial landmarks
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Detect if mouth is closed
                mouth_closed = detect_closed_mouth([(lm.x, lm.y) for lm in face_landmarks.landmark])
                if mouth_closed:
                    cv2.putText(frame, "Mouth Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Face Mesh Mouth Closed Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
