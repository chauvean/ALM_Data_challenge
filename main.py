# usr/bin/env python3
import numpy as np
import pandas as pd
from itertools import permutations, product
import random as rd
from collections import OrderedDict
from time import time


def list_permutations(L, per_size):
    return [''.join(x) for x in list(product(L, repeat=per_size))]


#
# def find_most_common_subseq(seq, set, first_ns):
#     "doesn't count overlapping ex : AA appear stwice in AAA but .count only counts one"
#     dic = dict()
#     for key in set:
#         dic[key] = seq.count(key)
#     d = list({k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}.items())
#     return d[:first_ns]
#
#
def subseqs_ids(subsequences, sequence):
    return [1 if subsequence in sequence else 0 for subsequence in subsequences]


#
#
# def most_common_subseq_df():
#     seq_df = pd.DataFrame(pd.read_csv("./data/Xtr0.csv"))
#     bound_df = pd.DataFrame(pd.read_csv("./data/Ytr0.csv"))
#     joined_df = seq_df.merge(bound_df, on='Id')
#     for i in range(4):
#         perms_list_iter = list_permutations(['T', 'C', 'G', 'A'], i)
#         # look for ma separation in bounds
#     perms_list = list_permutations(['T', 'C', 'G', 'A'], 4)
#     joined_df['BEST_SEQ'] = [
#         find_most_common_subseq(line[1]['seq'], perms_list, 1)[0][0]
#         for
#         line in
#         joined_df.iterrows()]
#     # new_df = joined_df[joined_df['BEST_SEQ'] == 'CTAA']
#     # bound_one_df = new_df[new_df['Bound'] == 1]
#     # bound_two_df = new_df[new_df['Bound'] == 0]
#     # c = list_permutations(['T', 'C', 'G', 'A'],2)
#
#
def subseqids_df(joined_df):
    # seq_df = pd.DataFrame(pd.read_csv("./data/Xtr0.csv"))
    # bound_df = pd.DataFrame(pd.read_csv("./data/Ytr0.csv"))
    # joined_df = seq_df.merge(bound_df, on='Id')
    perms_list = list_permutations(['T', 'C', 'G', 'A'], 3)
    joined_df['subseqs_in_seq'] = [subseqs_ids(perms_list, line[1]['seq']) for
                                   line in
                                   joined_df.iterrows()]
    bounds_subseqids_idf = joined_df['subseqs_in_seq']
    return bounds_subseqids_idf


def percep_weights_update(df):
    df["Bound"] = df["Bound"].apply(lambda x: -1 if x == 0 else 1)
    input_set = df.values.tolist()
    w = [0.001 for _ in range(len(df.iloc[0]["subseqs_in_seq"]) + 1)]
    eta = 0.01
    T = 1000000
    t = 0
    while (t < T):
        random_sample = input_set[rd.randint(0, len(input_set) - 1)]
        random_sample[0] = [-1 if val == 0 else 1 for val in random_sample[0]]
        if np.dot(w[1:], random_sample[0]) * random_sample[1] < 0:
            w[0] = w[0] + eta * random_sample[1]
            w[1:] = [sum(x) for x in zip(w[1:], map(lambda x: x * eta * random_sample[1], random_sample[0]))]
        t += 1
    return w


def pegasos_weights_update(df, T, lambda_):
    df["Bound"] = df["Bound"].apply(lambda x: -1 if x == 0 else 1)
    input_set = df.values.tolist()
    w = [0.001 for _ in range(len(df.iloc[0]["subseqs_in_seq"]))]
    for t in range(T):
        random_sample = input_set[rd.randint(0, len(input_set) - 1)]
        eta_t = 1.0 / (lambda_ * (t + 1))
        w = list(map(lambda x: x * (1 - eta_t * lambda_), w))
        if np.dot(w, random_sample[0]) * random_sample[1] < 1:
            w = [sum(x) for x in zip(w, map(lambda x: x * eta_t * random_sample[1], random_sample[0]))]
    return w


def perceptron(w, df):
    df["percep_prediction"] = df['subseqs_in_seq'].apply(lambda x: 1 if np.dot(w[1:], x) + w[0] > 0 else 0)
    return df


def pegasos(w, df):
    df["Bound"] = df['subseqs_in_seq'].apply(lambda x: 1 if np.dot(w, x) > 1 else 0)
    return df


if __name__ == '__main__':
    seq_df_0 = pd.DataFrame(pd.read_csv("./data/Xtr0.csv"))
    bound_df_0 = pd.DataFrame(pd.read_csv("./data/Ytr0.csv"))
    joined_df_0 = seq_df_0.merge(bound_df_0, on='Id')
    seq_df_1 = pd.DataFrame(pd.read_csv("./data/Xtr1.csv"))
    bound_df_1 = pd.DataFrame(pd.read_csv("./data/Ytr1.csv"))
    joined_df_1 = seq_df_1.merge(bound_df_1, on='Id')
    seq_df_2 = pd.DataFrame(pd.read_csv("./data/Xtr2.csv"))
    bound_df_2 = pd.DataFrame(pd.read_csv("./data/Ytr2.csv"))
    joined_df_2 = seq_df_2.merge(bound_df_2, on='Id')
    joined_df = pd.concat([joined_df_0, joined_df_1, joined_df_2])
    df = pd.DataFrame()
    df['subseqs_in_seq'] = subseqids_df(joined_df)
    df['Bound'] = joined_df['Bound']
    w = pegasos_weights_update(df, 1000, 0.01)
    # df2 = pegasos(w, joined_df)
    # score = sum(1.0 for label, pred in zip(df2["Bound"], df2["pegasos_prediction"]) if label == pred) / len(
    #     df2["Bound"])
    # print(score)
    #
    final_df = pd.DataFrame()
    seq_df_0 = pd.DataFrame(pd.read_csv("./data/Xte0.csv"))
    seq_df_1 = pd.DataFrame(pd.read_csv("./data/Xte1.csv"))
    seq_df_2 = pd.DataFrame(pd.read_csv("./data/Xte2.csv"))
    seq_df_0['subseqs_in_seq'] = subseqids_df(seq_df_0)
    seq_df_1['subseqs_in_seq'] = subseqids_df(seq_df_1)
    seq_df_2['subseqs_in_seq'] = subseqids_df(seq_df_2)
    seq_df = pd.concat([seq_df_0, seq_df_1, seq_df_2])
    df2 = pegasos(w, seq_df)
    df3 = df2.drop(["seq", "subseqs_in_seq"], axis=1)
    df3 = df3.rename({'':'A'})
    # df3.set_index('A').to_csv('submission.csv')
    #
    ...
    import csv
    #
    # input_file = 'submission.csv'
    # output_file = 'submission.csv'
    # cols_to_remove = [0,0]  # Column indexes to be removed (starts at 0)
    #
    # cols_to_remove = sorted(cols_to_remove, reverse=True)  # Reverse so we remove from the end first
    # row_count = 0  # Current amount of rows processed
    #
    # with open(input_file, "r") as source:
    #     reader = csv.reader(source)
    #     with open(output_file, "w", newline='') as result:
    #         writer = csv.writer(result)
    #         for row in reader:
    #             row_count += 1
    #             print('\r{0}'.format(row_count), end='')  # Print rows processed
    #             for col_index in cols_to_remove:
    #                 del row[col_index]
    #             writer.writerow(row)