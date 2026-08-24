import numpy as np
from scipy import stats
from sklearn import metrics
import torch
from sklearn.metrics import precision_score, recall_score, auc as area_under_curve

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime
def custom_precision_recall_curve(custom_thresholds, y_test, y_scores):

    precisions = []
    recalls = []

    for t in custom_thresholds:
        predictions = (y_scores >= t).astype(int)
        
        p = precision_score(y_test, predictions, zero_division=0)
        r = recall_score(y_test, predictions)
        
        precisions.append(p)
        recalls.append(r)
    return np.array(precisions), np.array(recalls), np.array(custom_thresholds)

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = 2
    stats = []
    # print(target)
    # print(output)
    output_2 = 1-output
    output = np.concatenate((output[:, None],output_2[:, None]), axis=1)
    target = np.concatenate((target[:, None], (1-target)[:, None]), axis=1)
    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, axis=1), np.argmax(output,axis=1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:,k], output[:,k], average=None)
        # print(avg_precision)
        # AUC
        try:
            auc = metrics.roc_auc_score(target[:,k], output[:,k], average=None)

            # Precisions, recalls
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                target[:,k], output[:,k])
            # print(thresholds, len(thresholds))
            # FPR, TPR
            (fpr, tpr, thresholds) = metrics.roc_curve(target[:,k], output[:,k])
            
            fnr = 1 - tpr
    
            # 3. Find the index where the difference between FPR and FNR is smallest
            absolute_difference = np.abs(fpr - fnr)
            optimal_idx = np.argmin(absolute_difference)
            
            # 4. Extract the optimal threshold and the EER value
            optimal_threshold = thresholds[optimal_idx]

            # optimal_idx = np.argmax(j_scores)
            # optimal_threshold = thresholds[optimal_idx]
            # print(optimal_threshold)
            # exit()
            # acc = 0
            
            # for threshold in np.linspace(optimal_threshold-0.1, optimal_threshold+0.1, 500):
            #     new_acc = metrics.accuracy_score(np.argmax(target, axis=1), np.argmax(output>threshold,axis=1))
            #     if new_acc>acc:
            #         acc = new_acc
            #         optimal_threshold = threshold
            # print(acc)
            print("Optimal threshold", optimal_threshold)
            save_every_steps = 10000     # Sample statistics to reduce size
            dict = {'precisions': precisions[0::save_every_steps],
                    'recalls': recalls[0::save_every_steps],
                    'AP': avg_precision,
                    'fpr': fpr[0::save_every_steps],
                    'fnr': 1. - tpr[0::save_every_steps],
                    'auc': auc,
                    # note acc is not class-wise, this is just to keep consistent with other metrics
                    'acc': acc
                    }
        except Exception as e:
            dict = {'precisions': -1,
                    'recalls': -1,
                    'AP': avg_precision,
                    'fpr': -1,
                    'fnr': -1,
                    'auc': -1,
                    # note acc is not class-wise, this is just to keep consistent with other metrics
                    'acc': acc
                    }
            print('class {:s} no true sample'.format(str(k)))
        stats.append(dict)

    return stats