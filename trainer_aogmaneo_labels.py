import pyaogmaneo as neo
import numpy as np
import cv2
from health_tracker import get_health_amounts
from config import *
from torch_dataset import mask_healthbar

neo.set_num_threads(8)

lds = []

for i in range(4): # layers with exponential memory
    ld = neo.LayerDesc()

    ld.hidden_size = (8, 8, 32)

    lds.append(ld)

enc_hidden_size = (16, 16, 16)

enc = neo.ImageEncoder(enc_hidden_size, [neo.ImageVisibleLayerDesc((MODEL_IMG_SIZE[1], MODEL_IMG_SIZE[0], 3), 4)])
h = neo.Hierarchy([neo.IODesc(enc.get_hidden_size(), neo.none, up_radius=4), neo.IODesc((1, 1, 2), neo.prediction, num_dendrites_per_cell=32, up_radius=0, down_radius=4)], lds)

h.params.ios[1].importance = 0.0

labels = np.load("labels.npy")

for epoch in range(100):
    print(f"Epoch {epoch}")

    frame_index = 0

    for video_index in range(len(VIDEO_NAMES)):
        if video_index in SKIP_VIDEOS:
            continue

        video_filename = VIDEO_NAMES[video_index].split('.')[0] + "_proc.mp4"

        print(f"Training on {video_filename}")

        cap = cv2.VideoCapture(video_filename)

        while cap.isOpened():
            ret, frame = cap.read()

            if frame is None:
                break

            if labels[frame_index] != -1:
                if ENABLE_HEALTH_MASK:
                    mask_healthbar(frame)

                if frame.shape[0] != MODEL_IMG_SIZE[1] or frame.shape[1] != MODEL_IMG_SIZE[0]:
                    frame = cv2.resize(frame, MODEL_IMG_SIZE)

                enc.step([frame.ravel()], True, False)
                h.step([enc.get_hidden_cis(), [labels[frame_index]]], True)

            frame_index += 1

            if frame_index % 1000 == 0:
                print(f"Frame {frame_index}")

        cap.release()

        # save progress
        print("Saving...")

        enc.save_to_file("model.oenc")
        h.save_to_file("model.ohr")

        print("Saved.")
