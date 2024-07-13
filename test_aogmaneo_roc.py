import pyaogmaneo as neo
import numpy as np
import cv2
import os
import sys
import time
from health_tracker import *
from config import *
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import csv

def get_predictions():
    labels = []
    pred_acts = []
    round_ends = []
    video_names = []

    last_labels_end = 0

    neo.set_num_threads(8)

    h = neo.Hierarchy(file_name="model.ohr")
    enc = neo.ImageEncoder(file_name="model.oenc")

    frame_index = 0

    for video_index in TEST_VIDEOS:
        print(f"Testing on {VIDEO_NAMES[video_index]}")

        video_filename = VIDEO_NAMES[video_index].split('.')[0] + "_proc.mp4"

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

                round_ends.append(frame_index - 1)
                video_names.append(video_filename)

                for i in range(last_labels_end, frame_index):
                    labels[i] = label

                last_labels_end = frame_index

            healths.append(health_ratios)

            show_frame = frame.copy()

            if ENABLE_HEALTH_MASK:
                mask_healthbar(frame)

            if frame.shape[0] != MODEL_IMG_SIZE[1] or frame.shape[1] != MODEL_IMG_SIZE[0]:
                frame = cv2.resize(frame, MODEL_IMG_SIZE)

            labels.append(-1) # unset

            enc.step([frame.ravel()], False)
            h.step([enc.get_hidden_cis(), h.get_prediction_cis(1)], False)

            pred_acts.append(h.get_prediction_acts(1))
                
            frame_index += 1

        cap.release()

    return labels, pred_acts, round_ends, video_names

def get_frame_index(round_ends, round_index, ratio):
    round_start = 0

    if round_index > 0:
        round_start = round_ends[round_index - 1]

    round_end = round_ends[round_index]

    round_length = round_end - round_start

    round_offset = int(round_length * ratio)
    
    frame_index = round_start + round_offset

    return frame_index

def run():
    labels, pred_acts, round_ends, video_names = get_predictions()

    with open("aogmaneo_results.csv", "w") as f:
        writer = csv.writer(f)

        # write header
        writer.writerow(["Video", "Round", "PredictionForClass1", "GroundTruth"])

        last_end = 0

        for i in range(len(round_ends)):
            for j in range(last_end, round_ends[i]):
                writer.writerow([video_names[i], i, pred_acts[j][1], labels[j]])

            last_end = round_ends[i]

    num_thresholds = 1001

    for p in EVAL_PERCENTAGES:
        # aggragate statstics
        tprs = np.zeros(num_thresholds)
        fprs = np.zeros(num_thresholds)

        num_rounds = len(round_ends)

        # find all decisions
        evaluations = np.zeros(num_rounds)

        frame_indices = []

        # evaluate on data
        for i in range(len(round_ends)):
            frame_index = get_frame_index(round_ends, i, p / 100.0)

            frame_indices.append(frame_index)

            evaluations[i] = pred_acts[frame_index][1]

        for threshold_index in range(num_thresholds):
            threshold = threshold_index / (num_thresholds - 1)

            tp = 0
            fp = 0
            tn = 0
            fn = 0

            # evaluate on data
            for i in range(len(round_ends)):
                evaluation = evaluations[i]

                pred = int(evaluation > threshold)

                label = labels[frame_indices[i]]

                if label == -1:
                    continue

                if label == 1:
                    if pred == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if pred == 0:
                        tn += 1
                    else:
                        fp += 1

            tpr = tp / max(1, tp + fn)
            fpr = fp / max(1, fp + tn)

            tprs[threshold_index] = tpr
            fprs[threshold_index] = fpr

        print(f'AUC at {p}: {auc(fprs, tprs)}')

        with open("aogmaneo_results_short_" + str(int(p)) + ".csv", "w") as f:
            writer = csv.writer(f)

            # write header
            writer.writerow(["Video", "Round", "PredictionForClass1", "GroundTruth"])

            for i in range(len(round_ends)):
                frame_index = get_frame_index(round_ends, i, p / 100.0)

                writer.writerow([video_names[i], i, pred_acts[frame_index][1], labels[frame_index]])

        # plot ROC
        plt.plot(fprs, tprs, label='ROC')

        plt.show()
        
if __name__ == '__main__':
    run()
