import pyaogmaneo as neo
import numpy as np
import cv2
from health_tracker import get_health_amounts
from config import *

# pre-load so we can determine the labels beforehand
health_bar_template = cv2.imread("health_bar_template_rgb.png") # discards alpha

max_healths = get_health_amounts(health_bar_template)

health_bar_size = (health_bar_template.shape[1], health_bar_template.shape[0])
health_bar_loc = (OUTPUT_SIZE[0] * 0.5, OUTPUT_SIZE[1] * HEALTHBAR_OFFSET_Y)

health_bar_lower_x = int(health_bar_loc[0] - health_bar_size[0] * 0.5)
health_bar_upper_x = health_bar_lower_x + health_bar_size[0]
health_bar_lower_y = int(health_bar_loc[1] - health_bar_size[1] * 0.5)
health_bar_upper_y = health_bar_lower_y + health_bar_size[1]

labels = []

last_labels_end = 0

frame_index = 0

for video_index in range(len(VIDEO_NAMES)):
    if video_index in SKIP_VIDEOS:
        continue

    video_filename = VIDEO_NAMES[video_index].split('.')[0] + "_proc.mp4"

    print(f"Labeling {video_filename}")

    cap = cv2.VideoCapture(video_filename)

    healths = []

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        health_bar_actual = frame[health_bar_lower_y:health_bar_upper_y, health_bar_lower_x:health_bar_upper_x]

        current_healths = get_health_amounts(health_bar_actual)

        health_ratios = np.divide(current_healths, max_healths)

        # detect new round by health reset
        if len(healths) > 0 and (health_ratios[0] > healths[-1][0] + HEALTHBAR_TOLERANCE or health_ratios[1] > healths[-1][1] + HEALTHBAR_TOLERANCE):
            # set labels
            label = int(healths[-1][0] > healths[-1][1])

            for i in range(last_labels_end, frame_index):
                labels[i] = label

            last_labels_end = frame_index

        healths.append(health_ratios)

        if frame.shape[0] != MODEL_IMG_SIZE[1] or frame.shape[1] != MODEL_IMG_SIZE[0]:
            frame = cv2.resize(frame, MODEL_IMG_SIZE)

        labels.append(-1) # unset

        frame_index += 1

        if frame_index % 1000 == 0:
            print(f"Frame {frame_index}")

    cap.release()

np.save("labels.npy", np.array(labels, dtype=np.int32))

print("Done.")
