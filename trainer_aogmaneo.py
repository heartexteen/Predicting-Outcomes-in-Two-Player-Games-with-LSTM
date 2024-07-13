import pyaogmaneo as neo
import numpy as np
import cv2
from health_tracker import get_health_amounts
from config import *

skip_videos = { 1, 2, 7, 9 } # indices of videos that we want to skip (bad quality, for instance)

neo.set_num_threads(8)

lds = []

for i in range(6): # layers with exponential memory
    ld = neo.LayerDesc()

    ld.hidden_size = (8, 8, 32)

    lds.append(ld)

enc_hidden_size = (16, 16, 32)

enc = neo.ImageEncoder(enc_hidden_size, [neo.ImageVisibleLayerDesc((MODEL_IMG_SIZE[1], MODEL_IMG_SIZE[0], 3), 6)])
h = neo.Hierarchy([neo.IODesc(enc.get_hidden_size(), neo.prediction, up_radius=2)], lds)

for epoch in range(100):
    print(f"Epoch {epoch}")

    frame_index = 0

    for video_index in range(len(VIDEO_NAMES)):
        if video_index in skip_videos:
            continue

        video_filename = VIDEO_NAMES[video_index].split('.')[0] + "_proc.mp4"

        print(f"Processing {video_filename}")

        cap = cv2.VideoCapture(video_filename)

        cap_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()

            if frame is None:
                break

            frame = cv2.resize(frame, MODEL_IMG_SIZE)

            enc.step([frame.ravel()], True)
            h.step([enc.get_hidden_cis()], True)

            frame_index += 1

            if frame_index % 1000 == 0: # show that something is happening
                print(f"Frame {frame_index}")

            if frame_index % 10000 == 0: # save every now and then
                print("Saving...")

                enc.save_to_file("model.oenc")
                h.save_to_file("model.ohr")

                print("Saved.")

        cap.release()

        # save progress
        print("Saving...")

        enc.save_to_file("model.oenc")
        h.save_to_file("model.ohr")

        print("Saved.")

    print("Done.")
