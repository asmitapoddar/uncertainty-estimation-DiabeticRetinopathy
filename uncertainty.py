import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

'''
a = open("filename.txt", "w")
for path, subdirs, files in os.walk(r'/home/asmita/sbn_infer/zambia_hm/VGGnonref_Resref_hm/data'):
     for filename in files:
        f = os.path.join(path, filename)
        a.write(str(f) + os.linesep)
print('File list created')
'''

# mean KL-divergence
from scipy.stats import entropy

def mean_kl(data, mean_dist):
    # estimate parameters over first axis
    means = np.zeros((data.shape[1]))
    for i in range(means.shape[0]):
#         kl_div_max = 0
        kl_div_acc = entropy([1-mean_dist[i],mean_dist[i]])*data.shape[0]
        for k in range(data.shape[0]):
            kl_div = entropy([1-data[k,i,1],data[k,i,1]],[1-mean_dist[i],mean_dist[i]])
            kl_div_acc += kl_div
#             if kl_div_max < kl_div:
#                 kl_div_max = kl_div
        means[i] = kl_div_acc/(data.shape[0])
#         means[i] = kl_div_max
    return means

def cross_entropy(data, mean_dist):
    # estimate parameters over first axis
    means = np.zeros((data.shape[1]))
    for i in range(means.shape[0]):
        ent_max = 0
        ent_p = entropy([1-mean_dist[i],mean_dist[i]])
        for k in range(data.shape[0]):
            kl_div = entropy([1-data[k,i,1],data[k,i,1]],[1-mean_dist[i],mean_dist[i]])
            ent = kl_div + ent_p
            if ent_max < ent:
                ent_max = ent
        means[i] = ent_max
#         means[i] = kl_div_max
    return means


def performance_at_percentiles(df, uncertainty_measure, scoring_fn, percentiles, above=True):
    thresholds = np.percentile(df[uncertainty_measure], percentiles)

    def score_(threshold):
        if above:
            df_ = df[df[uncertainty_measure] >= threshold]
        else:
            df_ = df[df[uncertainty_measure] < threshold]
        return scoring_fn(df_)

    return [score_(threshold) for threshold in thresholds]


def plot_percentile_compare(scores_dict, percentiles, title, measure_name, above=True):
    plt.xlabel("k")
    plt.ylabel(measure_name)
    plt.title(title)
    #     plt.logx(steps[len(steps)-len(auc):],auc)
    for k, v in scores_dict.items():
        if above:
            plt.plot(100 - percentiles, v, label=k)
        else:
            plt.plot(percentiles, v, label=k)
    plt.legend()


def get_scores(df, scoring_fn, percentiles, above=True):
    return {'Standard deviation':
                performance_at_percentiles(df, 'std_0', scoring_fn, percentiles, above),
            'Entropy':
                performance_at_percentiles(df, 'entropy_1', scoring_fn, percentiles, above),
            'Entropy of entropies':
                performance_at_percentiles(df, 'entropy_score_1', scoring_fn, percentiles, above)
            # 'Scores':
            # performance_at_percentiles(df,'score1',scoring_fn,percentiles,above)
            #              ,
            #             'hybrid':
            #               performance_at_percentiles(df,'hybrid',scoring_fn,percentiles,above)
            }


def balanced_point(target, scores, sens_at_least=0.50):
    arr = np.asarray(roc_curve(target, scores))
    arr2 = arr[:, arr[1, :] > sens_at_least]
    #     print(arr.shape, arr2.shape)
    min_point = np.argmin(np.square(1 - arr2[1, :]) + np.square(arr2[0, :]))
    #   min_point = np.argmin(arr2[0,:])
    return arr2[1, min_point], 1 - arr2[0, min_point], arr2[2, min_point]


def roc_percentile_above_compare(df):
    percentiles = np.arange(99, 10, -2)
    scores = get_scores(df, lambda df_: roc_auc_score(df_['label'] >= 1, df_['mean_2']), percentiles)
    plot_percentile_compare(scores, percentiles, 'AUC for top k uncertain predictions', 'AUC')


def roc_percentile_below_compare(df):
    percentiles = np.arange(80, 100, 1)
    scores = get_scores(df, lambda df_: roc_auc_score(df_['label'] >= 1, df_['mean_2']), percentiles, above=False)
    plot_percentile_compare(scores, percentiles, 'AUC for bottom k uncertain predictions', 'AUC', above=False)


def balanced_accuracy(df_):
    _, _, threshold = balanced_point(df_['label'] >= 1, df_['mean_2'])
    return accuracy_score(df_['label'] >= 2, df_['mean_1'] >= threshold)


def accuracy_below_compare(df):
    percentiles = np.arange(80, 100, 1)
    scores = get_scores(df, lambda df_: accuracy_score(df_['label'] >= 1, df_['mean_2'] >= 0), percentiles, above=False)
    plot_percentile_compare(scores, percentiles, 'Accuracy for bottom k uncertain predictions', 'Accuracy', above=False)


def accuracy_above_compare(df):
    percentiles = np.arange(99, 10, -2)
    scores = get_scores(df, lambda df_: accuracy_score(df_['label'] >= 1, df_['mean_2'] >= 0), percentiles)
    plot_percentile_compare(scores, percentiles, 'Accuracy for top k uncertain predictions', 'Accuracy')


def label_below_compare(df):
    percentiles = np.arange(80, 100, 1)
    scores = get_scores(df, lambda df_: float((df_['label'] >= 1).sum()) / len(df_), percentiles, above=False)
    plot_percentile_compare(scores, percentiles, 'Ratio of positive label for bottom k uncertain predictions',
                            'Ratio of positive label', above=False)


def label_above_compare(df):
    percentiles = np.arange(99, 10, -2)
    scores = get_scores(df, lambda df_: float((df_['label'] >= 1).sum()) / len(df_), percentiles, above=True)
    plot_percentile_compare(scores, percentiles, 'Ratio of positive label for top k uncertain predictions',
                            'Ratio of positive label')


def plot_all(df):
    '''
    roc_percentile_above_compare(df)
    plt.show()
    roc_percentile_below_compare(df)
    plt.show()
    '''
    accuracy_above_compare(df)
    plt.show()
    accuracy_below_compare(df)
    plt.show()
    label_above_compare(df)
    plt.show()
    label_below_compare(df)

def uncertainty_measure():
    res_names = os.path.join(os.getcwd(), 'Result/data/image.png')

    res = torch.load(os.path.join(os.getcwd(), 'Result/res_norotate'))
    print('RES', res, 'RES[1]: labels', res[1], 'RES[0]: labels', res[0])
    logits = res[2]
    logits = logits.transpose((1,0,2))

    a = int(logits.shape[0]/4)
    b = (logits.shape[1])*4

    logits = logits.reshape(a,b,5)
    logits = logits.transpose((1,0,2))

    std_df  = pd.DataFrame(np.std(sigmoid(logits), axis=0), columns=['std_0', 'std_1', 'std_2', 'std_3', 'std_4'])
    mean_df = pd.DataFrame(np.mean(sigmoid(logits), axis=0),columns=['mean_0', 'mean_1', 'mean_2', 'mean_3', 'mean_4'])

    #df = pd.concat([res_names, mean_df, std_df], axis=1)   print(len(df))    roc_auc_score(df['label'] >= 2, df['mean_1'])

    return std_df