import pyaogmaneo as neo
import numpy as np
import cv2
import os
import sys
import time
from config import *

neo.set_num_threads(8)

h = neo.Hierarchy(file_name="model.ohr")
enc = neo.ImageEncoder(file_name="model.oenc")

labels = np.load("labels.npy")

accuracy = 0.5

frame_index = 0

cv2.namedWindow("debug", cv2.WINDOW_NORMAL)

cv2.resizeWindow("debug", (512, 512))

for video_index in range(len(VIDEO_NAMES)):
    if video_index in SKIP_VIDEOS:
        continue

    print(f"Processing {VIDEO_NAMES[video_index]}")

    video_filename = VIDEO_NAMES[video_index].split('.')[0] + "_proc.mp4"

    cap = cv2.VideoCapture(video_filename)

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        show_frame = frame.copy()

        if frame.shape[0] != MODEL_IMG_SIZE[1] or frame.shape[1] != MODEL_IMG_SIZE[0]:
            frame = cv2.resize(frame, MODEL_IMG_SIZE)

        pred_label = int(h.get_prediction_cis(1)[0])

        correct = int(pred_label == labels[frame_index])
            
        accuracy += 0.01 * (correct - accuracy)

        enc.step([frame.ravel()], False)

        h.step([enc.get_hidden_cis(), h.get_prediction_cis(1)], False)

        li = 0

        vals = list(h.get_hidden_cis(li))

        os.system("clear")

        for y in range(h.get_hidden_size(li)[1]):
            s = ""

            for x in range(h.get_hidden_size(li)[0]):
                v = vals[x + y * h.get_hidden_size(li)[0]]
                s += str(v) + ("  " if v < 10 else " ")

            print(s)

        enc.reconstruct(enc.get_hidden_cis())

        recon = enc.get_reconstruction(0).reshape((MODEL_IMG_SIZE[1], MODEL_IMG_SIZE[0], 3)).astype(np.uint8)

        show_frame = recon.copy()

        frame_index += 1

        cv2.putText(show_frame, "P: {0}".format(pred_label), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
        cv2.putText(show_frame, "A: {0}".format(labels[frame_index]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=2)

        cv2.imshow("debug", show_frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            exit(0)

        if frame_index % 100 == 99:
            print(frame_index)
            print(accuracy)

    cap.release()

print("Done.")
