import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
blinking = []

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# Function to detect blinks
def detect_blink(landmarks):
    left_eye = [landmarks[159][0], landmarks[159][1]]  # left eye outer corner
    right_eye = [landmarks[386][0], landmarks[386][1]]  # right eye outer corner
    top_left_eye = [landmarks[158][0], landmarks[158][1]]  # left eye top
    bottom_left_eye = [landmarks[145][0], landmarks[145][1]]  # left eye bottom
    top_right_eye = [landmarks[385][0], landmarks[385][1]]  # right eye top
    bottom_right_eye = [landmarks[374][0], landmarks[374][1]]  # right eye bottom

    left_eye_distance = euclidean_distance(top_left_eye, bottom_left_eye)
    right_eye_distance = euclidean_distance(top_right_eye, bottom_right_eye)

    # Calculate the aspect ratio of eyes
    aspect_ratio = (left_eye_distance + right_eye_distance) / (2.0 * euclidean_distance(left_eye, right_eye))

    return aspect_ratio

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

                # Detect blinks
                aspect_ratio = detect_blink([(lm.x, lm.y) for lm in face_landmarks.landmark])
                if aspect_ratio < 0.19:  # Threshold for blinking, adjust as needed
                    cv2.putText(frame, "Blink Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    blinking.append(1)
                else:
                    blinking.append(0)

        cv2.imshow('Face Mesh Blink Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    df = pd.DataFrame({
        'blinking': blinking
    })

    # Export DataFrame to CSV
    df.to_csv('blinkingoutput.csv', index=False)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
