import cv2
import numpy as np
import mediapipe as mp
import math
import statistics
import pandas as pd
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = round(fps)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

L_H_LEFT = [33]     
L_H_RIGHT = [133]   
R_H_LEFT = [362]    
R_H_RIGHT = [263]   

leftcenter = []
rightcenter= []
ranchor = []
lanchor = []

ratios = []
blinking = []
yapping = []

ratiocoefvar = []
blinkcoefvar = []
yapcoefvar = []

def calccoefvar(list):
    if statistics.mean(list) ==0:
        return 0
    else:
        return statistics.stdev(list)/statistics.mean(list)

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

    return lip_eye_ratio #< 0.482  # Adjust threshold as needed

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position =""
    if ratio <= 0.42:
        iris_position="right"
    elif ratio > 0.42 and ratio <= 0.57:
        iris_position="center"
    else:
        iris_position = "left"
    return iris_position, ratio

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx,r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            
            cv2.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)
            leftcenter.append(center_left)
            rightcenter.append(center_right)

            
            cv2.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 255, 255), -1, cv2.LINE_AA)
            ranchor.append(mesh_points[R_H_RIGHT][0])
            lanchor.append(mesh_points[R_H_LEFT][0])

            mcenter_to_right_dist = euclidean_distance(center_right, mesh_points[R_H_RIGHT][0])
            mtotal_distance = euclidean_distance(mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0])
            mratio = mcenter_to_right_dist/mtotal_distance
            ratios.append(mratio)

            #iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])

            #print(iris_pos)
            for face_landmarks in results.multi_face_landmarks:
                # Draw facial landmarks
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Detect blinks
                aspect_ratio = detect_blink([(lm.x, lm.y) for lm in face_landmarks.landmark])
                blinking.append(aspect_ratio)

                # Detect if mouth is closed
                mouth_closed = detect_closed_mouth([(lm.x, lm.y) for lm in face_landmarks.landmark])
                yapping.append(mouth_closed)
                
        cv2.imshow("img", frame)
        key = cv2.waitKey(1)
        if key ==ord("q"):
            break
cap.release()
df = pd.DataFrame({
    'leftcenter': leftcenter,
    'rightcenter': rightcenter,
    'ranchor': ranchor,
    'lanchor': lanchor,
    'ratios': ratios,
    'blinking' : blinking,
    'yapping' : yapping
})

for i in range(0, len(ratios), 13):
    # Calculate coefficient of variation for each sublist
    ratiocoefvar.append(statistics.stdev(ratios[i:i+10])*2)
    blinkcoefvar.append(statistics.stdev(blinking[i:i+10])*15)
    yapcoefvar.append(statistics.stdev(yapping[i:i+10])*15)

df2 = pd.DataFrame({
    'ratiocoefvar': ratiocoefvar,
    'blinkcoefvar' : blinkcoefvar,
    'yapcoefvar' : yapcoefvar
})


# Export DataFrame to CSV
df.to_csv('pupilblinkyapoutput.csv', index=False)
df2.to_csv('graphedcoefvars.csv', index=False)
#end_time = ((13/fps)*len(ratiocoefvar) + (13/fps))*100
#end_time = round(end_time)
x = range(1,len(ratiocoefvar)+1)

#x /= 100
# Plotting the data
plt.plot(x, ratiocoefvar, label='Eye Movement')
plt.plot(x, blinkcoefvar, label='Blinking')
plt.plot(x, yapcoefvar, label='Yapping')

# Adding labels and title
plt.xlabel('Frames')
plt.ylabel('Stddev of Metric')
plt.title('Stddev of Metric vs. Frame')
plt.legend()
end_time = len(ratiocoefvar)+1
minPos = plt.Slider(plt.axes([0.2,0.1,0.65,0.03]), 'Min', 0.1, end_time)
maxPos = plt.Slider(plt.axes([0.2,0.05,0.65, 0.03]), 'Max', minPos.val, end_time, valinit = end_time)
def update(val):
        if minPos.val > maxPos.val:
                maxPos.set_val(minPos.val)
        elif maxPos.val < minPos.val:
                minPos.set_val(maxPos.val)
        ax2.axis([minPos.val,maxPos.val,-0.25,7.5])
        fig2.canvas.draw_idle()

minPos.on_changed(update)
maxPos.on_changed(update)

# Adding legend
#plt.legend()

# Displaying the plot
plt.grid(True)
plt.savefig('stddev_blinkyappupil.png', dpi=300)  # Save as PNG with 300 dpi resolution
plt.show()

cv2.destroyAllWindows()
