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
from health_tracker import *
from config import *
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import csv

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_predictions():
    labels = []
    pred_acts = []
    round_ends = []
    video_names = []

    last_labels_end = 0

    torch.set_num_threads(8)

    device = "cpu"

    frame_stack = 16

    model = ViT(
        image_size = MODEL_IMG_SIZE[0],
        channels=3 * frame_stack,
        patch_size = 16,
        num_classes = 2,
        dim = 512,
        depth = 1,
        heads = 4,
        mlp_dim = 512
    ).to(device)

    model.load_state_dict(torch.load("trained-vit.pt", map_location=device))

    model.eval()

    image = np.zeros((1, 3 * frame_stack, MODEL_IMG_SIZE[0], MODEL_IMG_SIZE[0]), dtype=np.float32)

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

            frame = np.swapaxes(frame, 0, 2).astype(np.float32) / 255.0

            with torch.no_grad():
                # shift
                for t in range(frame_stack - 1, 1, -1):
                    image[0, t * 3 : (t + 1) * 3, :, :] = image[0, (t - 1) * 3 : t * 3, :, :]

                # set new
                image[0, :3, :, :] = frame

                pred_logits = model(torch.tensor(image, dtype=torch.float32).to(device)).numpy()[0]

                pred_acts.append(softmax(pred_logits))
                
            frame_index += 1

        cap.release()

    with open("vit_results.csv", "w") as f:
        writer = csv.writer(f)

        # write header
        writer.writerow(["PredictionForClass1", "GroundTruth"])

        for i in range(len(labels)):
            writer.writerow([pred_acts[i], labels[i]])

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

        with open("vit_results_full.csv", "w") as f:
            writer = csv.writer(f)

            # write header
            writer.writerow(["Video", "Round", "PredictionForClass1", "GroundTruth"])

            last_end = 0

            for i in range(len(round_ends)):
                for j in range(last_end, round_ends[i]):
                    writer.writerow([video_names[i], i, pred_acts[j][1], labels[j]])

                last_end = round_ends[i]

        with open("vit_results_short_" + str(int(p)) + ".csv", "w") as f:
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
