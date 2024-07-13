import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from health_tracker import *
from config import *

class SFDataset(Dataset):
    def __init__(self, frame_stack=1):
        self.frame_stack = frame_stack

        self.images = []
        self.labels = []

        last_labels_end = 0

        frame_index = 0

        for video_index in range(len(VIDEO_NAMES)):
            if video_index in SKIP_VIDEOS:
                continue

            video_filename = VIDEO_NAMES[video_index].split('.')[0] + "_proc.mp4"

            print(f"Processing {video_filename}")

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
                        self.labels[i] = label

                    last_labels_end = frame_index

                healths.append(health_ratios)

                if ENABLE_HEALTH_MASK:
                    mask_healthbar(frame)

                if frame.shape[0] != MODEL_IMG_SIZE[1] or frame.shape[1] != MODEL_IMG_SIZE[0]:
                    frame = cv2.resize(frame, MODEL_IMG_SIZE)

                self.images.append(frame)
                self.labels.append(-1) # unset

                frame_index += 1

                if frame_index % 1000 == 0:
                    print(f"Frame {frame_index}")

            cap.release()

        # remove unlabled frames
        self.images = [ self.images[i] for i in range(len(self.images)) if self.labels[i] != -1]
        self.labels = [ self.labels[i] for i in range(len(self.labels)) if self.labels[i] != -1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.zeros((3 * self.frame_stack, MODEL_IMG_SIZE[0], MODEL_IMG_SIZE[0]), dtype=np.float32)

        for t in range(min(self.frame_stack, idx + 1)):
            image[t * 3 : (t + 1) * 3, :, :] = np.swapaxes(self.images[idx - t].astype(np.float32) / 255.0, 0, 2)

        return image, self.labels[idx]

    def get_raw(self, idx):
        return self.images[idx], self.labels[idx]
