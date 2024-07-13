import pyaogmaneo as neo
import numpy as np
import cv2
import os
import sys
import time
from health_tracker import *
from config import *

labels = []
pred_labels = []
round_ends = []

last_labels_end = 0

neo.set_num_threads(8)

h = neo.Hierarchy(file_name="model.ohr")
enc = neo.ImageEncoder(file_name="model.oenc")

frame_index = 0

for video_index in TEST_VIDEOS:
    print(f"Testing on {VIDEO_NAMES[video_index]}")

    video_filename = VIDEO_NAMES[video_index].split('.')[0] + "_proc.mp4"

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

            round_ends.append(frame_index - 1)

            for i in range(last_labels_end, frame_index):
                labels[i] = label

            last_labels_end = frame_index

        healths.append(health_ratios)

        show_frame = frame.copy()

        if ENABLE_HEALTH_MASK:
            mask_healthbar(frame)

        if frame.shape[0] != MODEL_IMG_SIZE[1] or frame.shape[1] != MODEL_IMG_SIZE[0]:
            frame = cv2.resize(frame, MODEL_IMG_SIZE)

        labels.append(-1) # unset

        pred_label = int(h.get_prediction_cis(1)[0])

        pred_labels.append(pred_label)
            
        enc.step([frame.ravel()], False)
        h.step([enc.get_hidden_cis(), h.get_prediction_cis(1)], False)

        frame_index += 1

        cv2.putText(show_frame, "Pred: {0}".format(pred_label), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)

        cv2.imshow("debug", show_frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            exit(0)

        if frame_index % 100 == 99:
            print(frame_index)

        if frame_index > 6000:
            break

    cap.release()

# find accuracy
num_correct = 0

for i in round_ends:
    offset_i = i - 1 # Go back one extra step

    if pred_labels[offset_i] == labels[offset_i]:
        num_correct += 1

accuracy = num_correct / len(round_ends)

print(f"Accuracy: {accuracy}")

print("Done.")
