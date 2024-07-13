import torch
import torch.nn
import torch.optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pyaogmaneo as neo
import numpy as np
import cv2
from health_tracker import get_health_amounts
from config import *
from torch_dataset import SFDataset

neo.set_num_threads(8)

num_frames = 32

lds = []

for i in range(4): # layers with exponential memory
    ld = neo.LayerDesc()

    ld.hidden_size = (8, 8, 32)

    lds.append(ld)

enc_hidden_size = (16, 16, 16)

enc = neo.ImageEncoder(enc_hidden_size, [neo.ImageVisibleLayerDesc((MODEL_IMG_SIZE[1], MODEL_IMG_SIZE[0], 3), 4)])
h = neo.Hierarchy([neo.IODesc(enc.get_hidden_size(), neo.none, up_radius=4), neo.IODesc((1, 1, 2), neo.prediction, num_dendrites_per_cell=32, up_radius=0, down_radius=4)], lds)

h.params.ios[1].importance = 0.1

train_data = SFDataset(frame_stack=num_frames)

average = 0.0

for it in range(100000):
    idx = np.random.randint(0, len(train_data))

    for t in range(num_frames):
        frame_index = idx - num_frames + t

        if frame_index < 0:
            continue

        image, label = train_data.get_raw(frame_index)

        if t == 1:
            average += 0.001 * (float(h.get_prediction_cis(1)[0] == label) - average)

        enc.step([image.ravel()], True, False)
        h.step([enc.get_hidden_cis(), [label]], True)

    if it % 1000 == 999:
        print(average)

        # save progress
        print("Saving...")

        enc.save_to_file("model.oenc")
        h.save_to_file("model.ohr")

        print("Saved.")
        
