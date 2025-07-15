import cv2
from deepface import DeepFace
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

#update camera index line 13
#update file path line 54

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = 0
fps=round(fps)
print_time = 1

emotionarray = ([])

while video.isOpened():   #fix while loop to close upon commands
    ret,frame = video.read()
    if not ret:
     break
    timestamp = frame_count/fps
    if (timestamp == print_time):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for x, y, w, h in face:
            #img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            try:
                analyze = DeepFace.analyze(img_path = frame, actions = ["emotion"])
                text = str(analyze[0])
                m = re.search("'dominant_emotion': '(.+?)', 'region'", text)
                if m:
                    found = m.group(1)
                    emotionarray.append([print_time,found])      
            except:
                emotionarray.append([print_time, "no face found"])
        print_time += 1
    #cv2.imshow('video',frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
emotiondf = pd.DataFrame(data = emotionarray)
emotiondf.to_csv("emotion.csv", index = False)

#close video
#video.release()
cv2.destroyAllWindows()

#output
df = pd.read_csv("emotion.csv")
#got a wierd error and the column names weren't changing when I used the "emotiondf"
#but when i exported and re-imported it ran perfectly

#prep dataset
df = df.rename({'0': 'time', '1': 'emotion'}, axis=1)
df = df.assign(emotion_index=np.nan)
df = df.drop_duplicates()
print(df.columns)

neutral = 0
no_face = 0
anger = 0
sad = 0
surprise = 0
fear = 0
happy = 0
disgust = 0

for i in df.index:
    if df.loc[i,"emotion"] == "neutral":
        df.loc[i,"emotion_index"] = 1
        neutral += 1
    if df.loc[i,"emotion"] == "anger":
        df.loc[i,"emotion_index"] = 2
        anger += 1
    if df.loc[i,"emotion"] == "sad":
        df.loc[i,"emotion_index"] = 3
        sad += 1
    if df.loc[i,"emotion"] == "surprise":
        df.loc[i,"emotion_index"] = 4
        surprise += 1
    if df.loc[i,"emotion"] == "fear":
        df.loc[i,"emotion_index"] = 5
        fear += 1
    if df.loc[i,"emotion"] == "happy":
        df.loc[i,"emotion_index"] = 6
        happy += 1
    if df.loc[i,"emotion"] == "disgust":
        df.loc[i,"emotion_index"] = 7
        disgust += 1
    if df.loc[i,"emotion"] == "no face found":
        df.loc[i,"emotion_index"] = 0
        no_face += 1
key = "no face = 0 \neutral = 1 \nanger = 2 \nsad = 3 \nsurprise = 4 \nfear = 5 \nhappy = 6 \ndisgust = 7"
print(key)

# create precentages
sum_e = no_face+neutral + anger + sad + surprise + fear + happy + disgust

no_face = (no_face/sum_e)*100
neutral = (neutral/sum_e)*100
anger = (anger/sum_e)*100
sad = (sad/sum_e)*100
surprise = (surprise/sum_e)*100
fear = (fear/sum_e)*100
happy = (happy/sum_e)*100
disgust = (disgust/sum_e)*100

##  ----- Bar Chart for total emotions over entire interview -----

fig, ax = plt.subplots()
emotions = ["No Face Found","Neutral","Anger","Sad","Surprise","Fear","Happy","Digust"]
counts = [no_face, neutral, anger, sad, surprise, fear, happy, disgust]

bar_container = ax.bar(emotions, counts)

#colors?
#legend needed?

ax.bar(emotions,counts)
ax.set(ylabel = "Time shown (% of totatl)",title = "Overall Emotions")
ax.bar_label(bar_container, fmt= '{:,.0f} %')

plt.show()

## ----- Emotions over time -----
fig2, ax2 = plt.subplots()
plt.subplots_adjust(bottom=0.25)

plt.scatter(df.iloc[:,0], df.iloc[:,2])
plt.plot(df.iloc[:,0], df.iloc[:,2], "o--", alpha=.15)
plt.ylabel('Emotion')
plt.xlabel('Time (s)')
#might want to think about putting minutes on xlabel???
plt.yticks([0,1,2,3,4,5,6,7],emotions)
plt.axhline(y= 0,color = "r", linestyle = "-", alpha = .05)
plt.axhline(y= 1,color = "r", linestyle = "-", alpha = .05)
plt.axhline(y= 2,color = "r", linestyle = "-", alpha = .05)
plt.axhline(y= 3,color = "r", linestyle = "-", alpha = .05)
plt.axhline(y= 4,color = "r", linestyle = "-", alpha = .05)
plt.axhline(y= 5,color = "r", linestyle = "-", alpha = .05)
plt.axhline(y= 6,color = "r", linestyle = "-", alpha = .05)
plt.axhline(y= 7,color = "r", linestyle = "-", alpha = .05)


minPos = plt.Slider(plt.axes([0.2, 0.1, 0.65, 0.03]), 'Min', 0.1, print_time)
maxPos = plt.Slider(plt.axes([0.2, 0.05, 0.65, 0.03]), 'Max', minPos.val, print_time, valinit=print_time)

def update(val):
        if minPos.val > maxPos.val:
                maxPos.set_val(minPos.val)
        elif maxPos.val < minPos.val:
                minPos.set_val(maxPos.val)
        ax2.axis([minPos.val,maxPos.val,-0.25,7.5])
        fig2.canvas.draw_idle()

minPos.on_changed(update)
maxPos.on_changed(update)
plt.show()








