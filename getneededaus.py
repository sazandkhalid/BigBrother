from feat import Detector
import cv2
import numpy as np
import pandas as pd
from feat.plotting import imshow
import matplotlib.pyplot as plt

frames = []
au = []

detector = Detector()

au_model_kwards = {
    'threshold': 0.5,
    'use_gpu': True
}

results = detector.detect_video("C:\\Users\\cbond\\OneDrive\\Desktop\\Hoyalytics\\BB_AU_application\\WIN_20240401_23_10_21_Pro.mp4", 5, batch_size=5)

data2 = np.array(results.aus)
df2 = pd.DataFrame(data2, columns=results.au_columns)  # Set column names here

# Save the DataFrame to a CSV file
df2.to_csv('AUS.csv', index=False)  # Set index=False to avoid saving the index

# Now, select the required columns
selected_columns = ["AU05", "AU10", "AU07", "AU12"]
dfaus = df2[selected_columns]

# Save the selected columns DataFrame
dfaus.to_csv('neededAUS.csv', index=False)  # Set index=False to avoid saving the index

x = range(1, len(dfaus['AU05']) + 1)

# Plotting the data
plt.plot(x, dfaus['AU05'], label='AU05')
plt.plot(x, dfaus['AU07'], label='AU07')
plt.plot(x, dfaus['AU10'], label='AU10')
plt.plot(x, dfaus['AU12'], label='AU12')

# Adding labels and title
plt.xlabel('Frame')
plt.ylabel('AUS')
plt.title('AUS vs. Frame')

# Adding legend
plt.legend()

# Displaying the plot
plt.grid(True)
plt.savefig('au_graph.png', dpi=300)  # Save as PNG with 300 dpi resolution
#plt.show()