import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion
import numpy as np
import cv2
from health_tracker import get_health_amounts
from config import *

device = "cuda"

skip_videos = { 1, 2, 7, 9 } # indices of videos that we want to skip (bad quality, for instance)

model = Unet3D(
    dim = MODEL_IMG_SIZE[0],
    dim_mults = (1, 2, 4, 8)
).to(device)

num_frames = 5

diffusion = GaussianDiffusion(
    model,
    image_size = MODEL_IMG_SIZE[0],
    num_frames = num_frames,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).to(device)

# gather dataset
frames = []

for video_index in range(1):
    if video_index in skip_videos:
        continue

    video_filename = VIDEO_NAMES[video_index].split('.')[0] + "_proc.mp4"

    print(f"Processing {video_filename}")

    cap = cv2.VideoCapture(video_filename)

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        frame = cv2.resize(frame, MODEL_IMG_SIZE)

        frames.append(frame / 255.0 * 2.0 - 1.0)

    cap.release()

print("Gathered frames.")

inputs = torch.zeros((1, 3, num_frames, MODEL_IMG_SIZE[0], MODEL_IMG_SIZE[0]), dtype=torch.float32).to(device)

print("Ready.")

for it in range(100000):
    t = np.random.randint(0, len(frames) - num_frames)

    input_frames = torch.tensor(frames[t:t + num_frames])
    
    input_frames = input_frames.swapaxes(1, 3)
    input_frames = input_frames.swapaxes(0, 1)

    inputs[0, :, :, :, :] = input_frames.to(device)

    loss = diffusion(inputs)
    loss.backward()

    print(it)

    if it % 100 == 99:
        print("Saving...")

        torch.save(model.state_dict(), './trained-model.pt')
        torch.save(diffusion.state_dict(), './trained-diffusion.pt')

        print("Saved.")

