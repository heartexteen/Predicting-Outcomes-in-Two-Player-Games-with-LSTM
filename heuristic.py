import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from data_parser import SFData

def sigmoid(x):
    return np.tanh(x * 0.5) * 0.5 + 0.5

def evaluate_match(match_data, progress_ratio):
    num_rounds = len(match_data['round_winners'])

    max_healths = match_data['max_healths']

    num_values = len(match_data['series'])

    progress_index = int(progress_ratio * num_values)

    values_at_index = match_data['series'][progress_index]

    # count rounds won
    rounds_won0 = values_at_index[2]
    rounds_won1 = values_at_index[3]

    health_ratio0 = values_at_index[0] / max(1, max_healths[0])
    health_ratio1 = values_at_index[1] / max(1, max_healths[1])
    
    round_scale = 4.0 # scale importance of rounds relative to health

    points0 = rounds_won0 * round_scale + health_ratio0 - health_ratio1
    points1 = rounds_won1 * round_scale + health_ratio1 - health_ratio0

    squash_scale = 1.0

    return sigmoid((points1 - points0) * squash_scale)

def run():
    num_thresholds = 1001

    # aggragate statstics
    tprs = np.zeros(num_thresholds)
    fprs = np.zeros(num_thresholds)

    # load data
    data = SFData('data.csv')

    num_matches = len(data.matches)

    # find all decisions
    evaluations = np.zeros(num_matches)

    # evaluate on data
    match_index = 0

    for match_name, match_data in data.matches.items():
        print(f'Evaluating match: {match_name}')

        evaluations[match_index] = evaluate_match(match_data, 0.75)

        match_index += 1

    for threshold_index in range(num_thresholds):
        threshold = threshold_index / (num_thresholds - 1)

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        # evaluate on data
        match_index = 0

        for match_name, match_data in data.matches.items():
            evaluation = evaluations[match_index]

            pred = int(evaluation > threshold)

            label = match_data['match_winner']

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

            match_index += 1

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        tprs[threshold_index] = tpr
        fprs[threshold_index] = fpr

    print(f'AUC: {auc(fprs, tprs)}')

    # plot ROC
    plt.plot(fprs, tprs, label='ROC')

    plt.show()
        
if __name__ == '__main__':
    run()
