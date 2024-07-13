import numpy as np
import cv2
from config import *

health_bar_template = cv2.imread("health_bar_template_rgb.png") # discards alpha

health_bar_size = (health_bar_template.shape[1], health_bar_template.shape[0])
health_bar_loc = (OUTPUT_SIZE[0] * 0.5, OUTPUT_SIZE[1] * HEALTHBAR_OFFSET_Y)

health_bar_lower_x = int(health_bar_loc[0] - health_bar_size[0] * 0.5)
health_bar_upper_x = health_bar_lower_x + health_bar_size[0]
health_bar_lower_y = int(health_bar_loc[1] - health_bar_size[1] * 0.5)
health_bar_upper_y = health_bar_lower_y + health_bar_size[1]

def mask_healthbar(frame):
    frame[health_bar_lower_y:health_bar_upper_y, health_bar_lower_x:health_bar_upper_x] = np.zeros_like(frame[health_bar_lower_y:health_bar_upper_y, health_bar_lower_x:health_bar_upper_x])

def get_health_amounts(health_bar):
    half_width = health_bar.shape[1] // 2

    #yellow = (int(61.0 / 360.0 * 255.0), int(0.81 * 255.0), int(0.91 * 255.0))
    lower_yellow = np.array([25, 128, 128])
    upper_yellow = np.array([35, 255, 255])

    hsv = cv2.cvtColor(health_bar, cv2.COLOR_BGR2HSV)
    
    mask_left = cv2.inRange(hsv[:, :half_width, :], lower_yellow, upper_yellow)
    mask_right = cv2.inRange(hsv[:, half_width:, :], lower_yellow, upper_yellow)

    return np.array([np.sum(mask_left, dtype=np.float32), np.sum(mask_right, dtype=np.float32)])

max_healths = get_health_amounts(health_bar_template)

