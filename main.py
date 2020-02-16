# usr/bin/env python3
import numpy as np
import pandas as pd
from itertools import permutations, product
import random as rd
from collections import OrderedDict
from time import time
import os
from decision_tree import *
from sklearn.tree import DecisionTreeClassifier


# useful functions

def subseqs_ids(subsequences, sequence):
    """
    for each 1 to 4 length permutation of T,C,G'A array with presence or not of the subsequence,
    :param subsequences:
    :param sequence:
    :return: array like : [1,0,1, 1...] compatible format for ML study
    """
    return [1 if subsequence in sequence else 0 for subsequence in subsequences]


def list_permutations(L, per_size):
    return [''.join(x) for x in list(product(L, repeat=per_size))]


def load_training_data():
    seq_df_0 = pd.DataFrame(pd.read_csv("./data/train_data/sample/Xtr0.csv"))
    bound_df_0 = pd.DataFrame(pd.read_csv("./data/train_data/label/Ytr0.csv"))
    joined_df_0 = seq_df_0.merge(bound_df_0, on='Id')
    seq_df_1 = pd.DataFrame(pd.read_csv("./data/train_data/sample/Xtr1.csv"))
    bound_df_1 = pd.DataFrame(pd.read_csv("./data/train_data/label/Ytr1.csv"))
    joined_df_1 = seq_df_1.merge(bound_df_1, on='Id')
    seq_df_2 = pd.DataFrame(pd.read_csv("./data/train_data/sample/Xtr2.csv"))
    bound_df_2 = pd.DataFrame(pd.read_csv("./data/train_data/label/Ytr2.csv"))
    joined_df_2 = seq_df_2.merge(bound_df_2, on='Id')
    joined_df = pd.concat([joined_df_0, joined_df_1, joined_df_2])

    # fix columns names
    df = pd.DataFrame()
    df["id"] = joined_df["Id"]
    df['subseqs_in_seq'] = subseqids_df(joined_df)
    df['Bound'] = joined_df['Bound']
    return df


def load_test_data():
    test_df_0 = pd.DataFrame(pd.read_csv("./data/test_data/Xte0.csv"))
    test_df_1 = pd.DataFrame(pd.read_csv("./data/test_data/Xte1.csv"))
    test_df_2 = pd.DataFrame(pd.read_csv("./data/test_data/Xte2.csv"))
    return pd.concat([test_df_0, test_df_1, test_df_2])


def subseqids_df(joined_df):
    perms_list = list_permutations(['T', 'C', 'G', 'A'], 3)
    joined_df['subseqs_in_seq'] = [subseqs_ids(perms_list, line[1]['seq']) for
                                   line in
                                   joined_df.iterrows()]
    bounds_subseqids_idf = joined_df['subseqs_in_seq']
    return bounds_subseqids_idf


# pegasos andweight updatesweight updates

def percep_weights_update(input_set, T, eta):
    w = [0.001 for _ in range(65)]  # df.iloc[0]["subseqs_in_seq"]
    t = 0
    while (t < T):
        if len(input_set) != 0:
            i = rd.randint(0, len(input_set) - 1)
            random_sample = input_set[i]
            bb = np.dot(w[1:], random_sample.seq) * random_sample.bound < 0
            if bb:
                w[0] = w[0] + eta * random_sample.bound
                w[1:] = [sum(x) for x in
                         zip(w[1:], list(map(lambda x: x * eta * random_sample.bound, random_sample.seq)))]
            t += 1
    return w


def pegasos_weights_update(df, T, lambda_):
    w = [0.0000001 for _ in range(len(df.iloc[0]["subseqs_in_seq"]))]
    df["Bound"] = df["Bound"].apply(lambda x: -1 if x == 0 else 1)
    set_size = df.shape[0]
    list_of_samples = df.values.tolist()
    for t in range(1, T):
        i_t = rd.randint(1, set_size)
        eta_t = 1.0 / (lambda_ * (t + 1))
        fact_mult = 1 - eta_t * lambda_
        random_sample = list_of_samples[i_t]
        w = list(map(lambda x: fact_mult * x, w))
        if random_sample[1] * np.dot(w, random_sample[0]) < 1:
            tmp = list(map(lambda x: eta_t * random_sample[1] * x, random_sample[0]))
            w = list(map(np.add, w, tmp))
    return w


# prediction
def perceptron_prediction(w, sample):
    # df["percep_prediction"] = df['subseqs_in_seq'].apply(lambda x: 1 if np.dot(w[1:], x) + w[0] > 0 else -1)
    # return df
    return 1 if np.dot(w[1:], sample.seq) + w[0] > 0 else -1


def pegasos(w, sample_data):
    # df2 = df.copy()
    # df2["prediction"] = df2['subseqs_in_seq'].apply(lambda x: 1 if np.dot(w, x) > 1 else -1)
    # return df2
    return 1 if np.dot(w, sample_data) > 1 else -1


def sample_distrib(d_t, data_set, set_size, max_wanted_samples):
    """
    select at most max_wanted_samples wrt distribution at iteration t.
    :return:
    """
    # when distrib is updated it seems some distrib[i] are updated too much so max is really high and
    training_samples = []
    # distrib_mean = np.mean(d_t)
    distrib_max = max(d_t)
    for _ in range(max_wanted_samples):
        id = rd.randint(0, set_size)
        sample = data_set[id]
        # v = rd.uniform(0, distrib_mean)
        v = rd.uniform(0, distrib_max)
        if d_t[id] > v:
            training_samples.append(sample)
    return training_samples


# use of class sample to make calls to subseq, id, and bound more clear
class Sample:
    def __init__(self, id=0, seq=0, bound=-1):
        self.id = id
        self.seq = seq
        self.bound = bound

    def perceptron_prediction(self, w):
        return 1 if np.dot(w[1:], self.seq) + w[0] > 0 else -1


def df_to_sample(df):
    return list(map(lambda x: Sample(id=x[0], seq=x[1], bound=x[2]), df.values.tolist()))


def test_df_to_sample(df):
    return pd.DataFrame(list(map(lambda x: Sample(id=x[0], seq=x[1]), df.values.tolist())))


def ada_scratch(data_set_df, T=10):
    data_set_list = df_to_sample(data_set_df)
    half_set_size = int(len(data_set_list) / 2)
    X = data_set_df["subseqs_in_seq"].values.tolist()
    y = data_set_df["Bound"].values.tolist()
    # samples_weight_distribution = np.array([1.0 / (2 * half_set_size) for _ in range(2 * half_set_size)])
    samples_weight_distribution = np.ones(2 * half_set_size) / (2 * half_set_size)
    alpha_set = []
    predictions_list = []
    epsilon_t_list = []
    tree_list = []
    for t in range(T):
        # Fit decision tree classifier
        tree = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
        tree.fit(X, y, sample_weight=samples_weight_distribution)
        tree_list.append(tree)
        predictions = tree.predict(X)
        predictions_list.append(predictions)
        misclassified_samples = [1 if a == b else 0 for a, b in zip(predictions, y)]

        # misclassified_samples = [1 if val else 0 for val in list(misclassified_samples)]
        # epsilon_t = np.mean(np.average(misclassified_samples, weights=samples_weight_distribution, axis=0))
        # epsilon_t = np.mean(np.average(misclassified_samples, weights=samples_weight_distribution, axis=0))
        epsilon_t = np.average(misclassified_samples, weights=samples_weight_distribution, axis=0)
        epsilon_t_list.append(epsilon_t)
        alpha_t = 0.5 * np.log((1 - epsilon_t) / (epsilon_t + 0.0000000001))
        alpha_set.append(alpha_t)
        # samples_weight_distribution = [val * np.exp(alpha_t * misclassified_samples * ((val > 0) | (alpha_t < 0))) for
        #                                val in samples_weight_distribution]
        alpha_times_misclassified = list(
            map(lambda x: alpha_t * ((epsilon_t > 0) | (alpha_t < 0)) * x, misclassified_samples))
        samples_weight_distribution *= np.exp(alpha_times_misclassified)

        # FUCKING PREDICTIONS ARE FINISHED
    # final_classifier = lambda x: np.sign(sum(alpha_set[k] * x.perceptron_prediction(weights_set[k]) for k in range(T)))
    # final_class = lambda x: np.sign(
    #     np.sum(np.array([alpha_set[n] * predictions_list[n][x.id] for n in
    #             range(T)])))  # x.id is the index of x in the predictions_list[t] variable
    #
    # final_prediction = [final_class(sample) for sample in data_set_list]
    # YYY = [sample.bound for sample in data_set_list]
    # print(score([final_prediction, YYY], len(X)))
    pred_func = lambda sample: - np.sign(
        np.array(list(map(lambda x: x[0] * x[1], zip(func(tree_list, sample), alpha_set)))).sum())
    preds = [pred_func(sample) for
             sample
             in
             data_set_list]
    print(np.sum([x == y for x, y in zip(preds, y)]) / (2 * half_set_size))
    # print('Accuracy = ', np.array(list(map(inverse_pred, [x == y for x, y in zip(preds, y)]))).sum() / (2*half_set_size))
    return pred_func


def func(tree_list, sample):
    return np.array([tree.predict(np.array([sample.seq])) for tree in tree_list])


def zero_to_ones(L):
    """
    algos needs -1 and 1 values
    """
    return [-1 if val == 0 else 1 for val in L]


if __name__ == '__main__':
    test_df = test_df_to_sample(load_test_data())

    training_df = load_training_data()
    training_df["Bound"] = training_df["Bound"].apply(lambda x: -1 if x == 0 else 1)
    training_df["subseqs_in_seq"] = training_df["subseqs_in_seq"].apply(lambda x: zero_to_ones(x))

    t1 = time()
    # compute prediction function , from adaboost algorithm
    pred_func = ada_scratch(training_df, T=10)
    test_df.columns = ["sample"]
    test_df["id"] = [sample.id for sample in test_df["sample"]]
    # compute prediction for each test sample
    test_df["Bound"] = [pred_func(sample) for sample in test_df["sample"]]

    # export needed columns to csv
    test_df = test_df.drop(columns="sample")
    # test_df.to_csv("./adaboost_submission.csv")

    t2 = time()
    print("total time = {}".format(t2 - t1))

    # pegasos prediction
    # w = pegasos_weights_update(df, 1000, 0.01)
    # pred_df = pegasos(w, df)



# Logistic function
def logistic(x):
    return (1.0/(1.0+exp(-x)))

# Gradient vector calculation
def gradient_logistic_surrogate_loss(w, trainingSet):

    df["Bound"] = df["Bound"].apply(lambda x: -1 if x == 0 else 1)
    input_set = df.values.tolist()
    w = [0.0 for _ in range(len(df.iloc[0]["subseqs_in_seq"])+1)]

    for i in range(len(input_set)):
        sample = input_set[i]
        for ps=w[0], j=1 in range(random_sample[0]):
            ps+=w[j]*sample[0]
        g[0] += (logistic(sample[1]*ps)-1.0)*sample[1]
        for j=1 in range(input_set):
            g[j]+=(logistic(sample[1]*ps)-1.0)*sample[1]*sample[0]

    for j=0 in range len(input_set) :
        g[j] /= len(input_set)

    return g

# Logistic cost function calculation
def logistic_surrogate_loss(w, trainingSet):

    input_set = df.values.tolist()

    for i in range(len(input_set)):
        sample = input_set[i]
        for ps=w[0], j=1 in range(len(input_set)):
            ps += w[j]*sample[0]

        S += log(1.0 + exp(-sample[1] * ps))

    S /= len(input_set)

    return S

def logistic_regression(w, trainingSet, params):


