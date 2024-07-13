import pyaogmaneo as neo
import numpy as np
import cv2
import os
import sys
from PIL import Image
import time
import pygame
import pygame.surfarray
from config import *

show_size = (512, 512)
temperature = 0.1

neo.set_num_threads(8)

h = neo.Hierarchy(file_name="model.ohr")
enc = neo.ImageEncoder(file_name="model.oenc")
    
screen = pygame.display.set_mode(show_size)

pressed_prev = pygame.key.get_pressed()

while True:
    start_time = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    
    pressed = pygame.key.get_pressed()

    if pressed[pygame.K_q]:
        break

    h.step([h.get_prediction_cis(0)], False)
    
    enc.reconstruct(h.get_prediction_cis(0))

    recon = enc.get_reconstruction(0).reshape((MODEL_IMG_SIZE[1], MODEL_IMG_SIZE[0], 3)).astype(np.uint8)

    recon = cv2.cvtColor(recon, cv2.COLOR_BGR2RGB)

    show_img = np.swapaxes(recon, 0, 1)

    surf = pygame.transform.scale(pygame.surfarray.make_surface(show_img.astype(np.uint8)), show_size)

    screen.blit(surf, (0, 0))

    pygame.display.flip()

    end_time = time.time()

    delta_time = end_time - start_time

    time.sleep(max(0.0, 1.0 / 20.0 - delta_time))
