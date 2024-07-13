import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion
import numpy as np
import cv2
import os
import sys
from PIL import Image
import time
import pygame
import pygame.surfarray
from config import *

device = "cuda"

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
    
model.load_state_dict(torch.load("trained-model.pt"))
diffusion.load_state_dict(torch.load("trained-diffusion.pt"))

model.eval()
diffusion.eval()

sampled_video = diffusion.sample(batch_size=1)[0].cpu().numpy()

np.save("video.npy", sampled_video)

screen = pygame.display.set_mode(show_size)

pressed_prev = pygame.key.get_pressed()

t = 0

while True:
    start_time = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    
    pressed = pygame.key.get_pressed()

    if pressed[pygame.K_q]:
        break

    if pressed[pygame.K_SPACE]:
        sampled_video = diffusion.sample(batch_size=1)[0].cpu().numpy()

    if t >= sampled_video.shape[0]:
        t = 0

    recon = sampled_video[:, t, :, :]

    recon = np.swapaxes(recon, 0, 2)

    recon = cv2.cvtColor(recon, cv2.COLOR_BGR2RGB)

    show_img = ((np.swapaxes(recon, 0, 1) * 0.5 + 0.5) * 255.0).astype(np.uint8)

    show_img = np.swapaxes(recon, 0, 1)

    surf = pygame.transform.scale(pygame.surfarray.make_surface(show_img.astype(np.uint8)), show_size)

    screen.blit(surf, (0, 0))

    pygame.display.flip()

    end_time = time.time()

    delta_time = end_time - start_time

    time.sleep(max(0.0, 1.0 / 20.0 - delta_time))

    t += 1
