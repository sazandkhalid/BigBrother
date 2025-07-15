import cv2 
import mediapipe as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

#place camera in a way so that, when resting, the interviewee's hands are not in fram

cap = cv2.VideoCapture(0)  #change to your camera index
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

fps = cap.get(cv2.CAP_PROP_FPS)
fps = round(fps)
frame_count = 0
print_time = 1

handsarray = []
time = []

while cap.isOpened():
        success, frame = cap.read()
        if not success:
                break
        timestamp = frame_count/fps
        if (timestamp == print_time):
                if success:
                        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                        result = hand.process(frame)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        if result.multi_hand_landmarks:
                                handsarray.append(1)
                                time.append(print_time)
                                #handsarray.append([print_time,1])
                                for hand_landmarks in result.multi_hand_landmarks:
                                        x_max = 0 
                                        y_max = 0 
                                        x_min = frame.shape[1]
                                        y_min = frame.shape[0]
                                        for lm in hand_landmarks.landmark:
                                                x, y = int(lm.x*frame.shape[1]), int(lm.y*frame.shape[0])
                                                if x > x_max:
                                                        x_max = x
                                                if y > y_max: 
                                                        y_max = y 
                                                if x < x_min:
                                                        x_min = x
                                                if y < y_min:
                                                        y_min = y 
                                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0),2)
                        else:
                                #handsarray.append([print_time,0])
                                handsarray.append(0)
                                time.append(print_time)
                        cv2.imshow("capture image", frame)
                        print_time += 1
                else:
                        print_time += 1

                if cv2.waitKey(2) == ord('q'):
                        break 
        frame_count += 1 

hand.close()
cv2.destroyAllWindows()

handsdf = pd.DataFrame({'Time': time, 'Hand in Frame': handsarray})
handsdf.to_csv("hands.csv", index = False)


#####Code for Pie Chart#####

counts = np.bincount(handsdf.iloc[:,1])
#labels = 'Out of Frame', 'In Frame'
fig, ax = plt.subplots()
ax.pie(counts, autopct='%1.1f%%', shadow=True, startangle=90)
plt.savefig("handsPie.pdf", format="pdf", bbox_inches="tight")



#####Code for Line Graph#####

fig2, ax2 = plt.subplots()
plt.subplots_adjust(bottom=0.25)

plt.plot(handsdf.iloc[:,0], handsdf.iloc[:,1])
plt.ylabel('Hand in Frame')
plt.xlabel('Time')

minPos = plt.Slider(plt.axes([0.2, 0.1, 0.65, 0.03]), 'Min', 0.1, print_time)
maxPos = plt.Slider(plt.axes([0.2, 0.05, 0.65, 0.03]), 'Max', minPos.val, print_time, valinit=print_time)

def update(val):
        if minPos.val > maxPos.val:
                maxPos.set_val(minPos.val)
        elif maxPos.val < minPos.val:
                minPos.set_val(maxPos.val)
        ax2.axis([minPos.val,maxPos.val,-0.5,1.5])
        fig2.canvas.draw_idle()

minPos.on_changed(update)
maxPos.on_changed(update)
plt.show()
