import cv2
import pandas as pd
import numpy as np

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize dataframe to store fX and fY coordinates
df = pd.DataFrame(columns=['fX', 'fY'])

# Open the video file
cap = cv2.VideoCapture("q.mp4")
fy = []
fx = []
while(cap.isOpened()):
    # Read a frame from the video
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Iterate through each detected face
    for (x, y, w, h) in faces:
        # Calculate the center of the face
        fX = x + w/2
        fY = y + h/2
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Append fX and fY to the dataframe
        fy.append(fY)
        fx.append(fX)
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Press 'q' to exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
print(fx)
df = pd.DataFrame({'fx' :fx,'fy':fy})

def calculate_stddev_per_13_frames(df):
    # Initialize arrays to store standard deviations
    stddev_fX_values = []
    stddev_fY_values = []
    
    # Calculate the total number of frames
    total_frames = len(df)
    
    # Iterate through every 13 frames
    for i in range(0, total_frames, 13):
        # Select the subset of dataframe containing 13 frames
        subset_df = df.iloc[i:i+13]
        
        # Calculate the standard deviation of fX and fY for the subset
        stddev_fX = subset_df['fx'].std()
        stddev_fY = subset_df['fy'].std()
        
        # Append the calculated standard deviations to the arrays
        stddev_fX_values.append(stddev_fX)
        stddev_fY_values.append(stddev_fY)
    
    # Create a dataframe from the arrays
    result_df = pd.DataFrame({'stddev_fX': stddev_fX_values, 'stddev_fY': stddev_fY_values})
    
    return result_df

# Example usage
# Load your dataframe containing fX and fY coordinates

# Call the function to calculate standard deviation per 13 frames
result_df = calculate_stddev_per_13_frames(df)

# Print the resulting dataframe
print(result_df)


# Export the dataframe to a CSV file
calculate_stddev_per_13_frames(df).to_csv('face_tracking_data.csv', index=False)

# Calculate standard deviation of fX and fY columns and print
print("Standard Deviation of fX:", df['fx'].std())
print("Standard Deviation of fY:", df['fy'].std())

