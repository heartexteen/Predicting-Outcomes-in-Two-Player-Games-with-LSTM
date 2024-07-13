import numpy as np
import cv2
from health_tracker import *
from config import *

health_bar_mag_inv = 1.0 / np.sqrt(np.sum(np.square(health_bar_template[:, :, :3].ravel() / 255.0)))

skip_frames = 2

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
output_fps = 30

min_health_match = 0.05
min_template_match = 0.7
min_round_frames = 20

show = False

if show:
    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("debug", (1024, 1024))

for video_index in range(len(VIDEO_NAMES)):
    print(f"Processing {VIDEO_NAMES[video_index]}")

    cap = cv2.VideoCapture(VIDEO_NAMES[video_index])

    video_filename = VIDEO_NAMES[video_index].split('.')[0] + "_proc.mp4"

    out = cv2.VideoWriter(video_filename, fourcc, output_fps, OUTPUT_SIZE)

    frame_index = 0

    non_video_data = []
    start_frames = []

    healths = []

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        for i in range(skip_frames):
            ret, frame = cap.read()

            if frame is None:
                break

        if frame is None:
            break

        # crop
        lower_x = int(CROP_BOUNDS[video_index][0] * frame.shape[1])
        upper_x = int(CROP_BOUNDS[video_index][1] * frame.shape[1])
        upper_y = int(CROP_BOUNDS[video_index][2] * frame.shape[0])
        lower_y = int(CROP_BOUNDS[video_index][3] * frame.shape[0])

        # crop
        frame = frame[lower_y:upper_y, lower_x:upper_x]

        frame = cv2.resize(frame, OUTPUT_SIZE, cv2.INTER_AREA)
        
        health_bar_actual = frame[health_bar_lower_y:health_bar_upper_y, health_bar_lower_x:health_bar_upper_x]

        current_healths = get_health_amounts(health_bar_actual)

        health_ratios = np.divide(current_healths, max_healths)

        template_match = np.dot(health_bar_actual[:, :, :3].ravel() / 255.0, health_bar_template[:, :, :3].ravel() / 255.0) * health_bar_mag_inv / max(0.0001, np.sqrt(np.sum(np.square(health_bar_actual[:, :, :3].ravel() / 255.0))))

        show_frame = frame.copy()

        if health_ratios[0] + health_ratios[1] >= min_health_match and template_match >= min_template_match: # if some health is on-screen and the template matches (weakly)
            frame_proc = frame.copy()

            # get more accurate health data
            if len(healths) == 0 or health_ratios[0] > healths[-1][0] and health_ratios[1] > healths[-1][1]:
                if len(healths) > min_round_frames:
                    # normalize
                    max_health_ratios = np.zeros(2)
                    min_health_ratios = np.ones(2) * 999999.0

                    for i in range(len(healths)):
                        max_health_ratios = np.maximum(max_health_ratios, healths[i])
                        min_health_ratios = np.minimum(min_health_ratios, healths[i])

                    max_health_ratio = max(max_health_ratios[0], max_health_ratios[1])
                    min_health_ratio = max(min_health_ratios[0], min_health_ratios[1])

                    # normalize and discard if no in-between values
                    contains_in_between = [False, False]

                    start_index = -1

                    for i in range(len(healths)):
                        healths[i] = np.divide(healths[i] - min_health_ratio, np.maximum(0.0001, max_health_ratio - min_health_ratio))

                        healths[i] = healths[i].tolist()

                        if healths[i][0] == 1.0 and healths[i][1] == 1.0:
                            start_index = i

                        if healths[i][0] > 0.0 and healths[i][0] < 1.0:
                            contains_in_between[0] = True

                        if healths[i][1] > 0.0 and healths[i][1] < 1.0:
                            contains_in_between[1] = True

                    if contains_in_between[0] and contains_in_between[1] and start_index != -1:
                        healths = healths[start_index:]

                        if len(healths) >= min_round_frames:
                            non_video_data.append(healths)
                            start_frames.append(frame_index - len(healths))

                healths = []

            healths.append(health_ratios)

            if show:
                show_frame[health_bar_lower_y:health_bar_upper_y, health_bar_lower_x:health_bar_upper_x] = health_bar_template

            out.write(frame_proc)

            #print("Health bar detected.")

        frame_index += 1

        if show:
            cv2.imshow("debug", show_frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                exit(0)

        if frame_index % 100 == 99:
            print(frame_index)

    out.release()

    cap.release()

print("Done.")
