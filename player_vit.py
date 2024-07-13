import torch
import torch.nn
import torch.optim
from vit_pytorch import ViT
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import sys
import time
from config import *

torch.set_num_threads(8)

device = "cpu"

frame_stack = 4

model = ViT(
    image_size = MODEL_IMG_SIZE[0],
    channels=3 * frame_stack,
    patch_size = 16,
    num_classes = 2,
    dim = 512,
    depth = 2,
    heads = 4,
    mlp_dim = 512
).to(device)

model.load_state_dict(torch.load("trained-vit.pt", map_location=device))

model.eval()

image = np.zeros((1, 3 * frame_stack, MODEL_IMG_SIZE[0], MODEL_IMG_SIZE[0]), dtype=np.float32)

labels = np.load("labels.npy")

frame_index = 0

accuracy = 0.5

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

        if frame is None:
            break

        show_frame = frame.copy()

        if frame.shape[0] != MODEL_IMG_SIZE[1] or frame.shape[1] != MODEL_IMG_SIZE[0]:
            frame = cv2.resize(frame, MODEL_IMG_SIZE)

        frame = np.swapaxes(frame, 0, 2).astype(np.float32) / 255.0

        with torch.no_grad():
            # shift
            for t in range(frame_stack - 1, 1, -1):
                image[0, t * 3 : (t + 1) * 3, :, :] = image[0, (t - 1) * 3 : t * 3, :, :]

            # set new
            image[0, :3, :, :] = frame

            pred_label = np.argmax(model(torch.tensor(image, dtype=torch.float32).to(device)).numpy()[0])

            correct = int(pred_label == labels[frame_index])
                
            accuracy += 0.01 * (correct - accuracy)

        frame_index += 1

        cv2.putText(show_frame, "Pred: {0}".format(pred_label), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
        cv2.putText(show_frame, "Actual: {0}".format(labels[frame_index]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), thickness=2)

        cv2.imshow("debug", show_frame)

        if cv2.waitKey(10) & 0xff == ord('q'):
            exit(0)

        if frame_index % 100 == 99:
            print(frame_index)
            print(accuracy)

    cap.release()

print("Done.")
