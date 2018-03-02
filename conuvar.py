"""
Copy Number Variation (CoNuVar)

<Add a short description>

created by
    Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@ucla.edu>
"""

import os, sys
import optparse
import pandas as pd
import numpy as np
import random
import time
import multiprocessing as mp


def load_data(path):
    print("Loading data ...")
    if path[-4:] == ".txt":
        data = pd.read_table(path,
                             sep="\t",
                             header="infer",
                             engine="c")
    elif path[-4:] == ".csv":
        data = pd.read_csv(path,
                           header="infer",
                           engine="c")
    else:
        sys.exit("Cannot identify the input file format!")

    preprocess(data)
    count      = pd.DataFrame.as_matrix(data.TPM)
    conuvar    = pd.DataFrame.as_matrix(data.True_CNV)
    foldchange = pd.DataFrame.as_matrix(data.FoldChange)

    print("Done!")

    return count, conuvar, foldchange


def preprocess(data):
    """Data preprocessing

    Inplace data pre-processing on Pandas level.
    Pre-process data by following rules:

        1 - Remove all entries whose fc < 0.1
        2 - TBC

    Args:
        data - the dataset to work on
    """
    print("=== Preprocess - Under construction ===")
    pass


def initialize(init, n_entry, fc=None):
    """Initialize

    Initialize and return predicted y

    Args:
        init    - indicating how to initialize predicted_y
        n_entry - number of entries in the dataset
        fc      - the array of foldchange,
                  only required when init=="fc"

    Return:
        pred_y  - the initialized predicted y
    """
    print("Initialize y prediction ...")
    if init == "random":
        return np.random.uniform(low=0.0, high=10.0,
                                 size=n_entry)
    elif init == "fc":
        if not fc:
            sys.exit("Please pass in FoldChange params!")
        return fc
    else:
        sys.exit("Cannot identify initialization param!")


def optimize(train_X, pred_y,
             n_iter=1000, interval=5, sr=0.01, lr=0.001, n_proc=4):

    print("Optimizing ...")
    bin_count = {}

    # Creating bin_count for sampling
    for i in range(len(train_X)):
        bin = int(train_X[i]/interval)
        if bin in bin_count:
            bin_count[bin].append(i)
        else:
            bin_count[bin] = [i]

    yi_index = list(range(len(pred_y)))

    for n in range(n_iter):
        # The n-th iteration
        np.random.shuffle(yi_index)
        train_samples = sample(bin_count, sr) # contain samples' indexes
        # print(len(train_samples))

        # Single Process Version
        # start_time = time.time()
        # for i in yi_index:
        #     pred_y[i] = sgd(X=train_X, y=pred_y,
        #                     cur=i, sample_ind=train_samples, lr=lr)
        # end_time = time.time()
        # print("Time used:", end_time - start_time)

        # Multiprocess Version
        yi_index_list = np.array_split(yi_index, n_proc)
        pool = mp.Pool(processes=n_proc)
        for i in range(n_proc):
            pool.apply_async(sgd, (train_X, pred_y, yi_index_list[i], train_samples, lr))
        pool.close()
        pool.join()

        if not n % 10:
            msg = "Optimizing {:.3f}%".format((100 * n/n_iter))
            sys.stdout.write('\r' +msg)



    print("Done!")


def multiproc_worker(X, y, yi_list, samples, lr):
    for i in yi_list:
        y[i] = sgd(X, y, i, samples, lr)


def evaluate(ground_truth, pred_y):
    """Evaluate the predicted result

    Evalute the predicted result

    Args:
        ground_truth - the ground truth of copy number variation
        pred_y       - the predicted value

    Return:
        don't know what to return
    """
    print("=== EVALUATION - Under construction ===")
    pass


def sample(bin_count, sr):
    """Draw samples to optimize

    Proportionally draw samples from the empirical distribution of
        read counts to optimize.

    Args:
        bin_count - the count dictionary of the bins
        sr        - sample rate,
                    the ratio of indexes sampled in each bin

    Return:
        samples - A shuffled list of indexes of samples
    """
    samples = []
    for key in bin_count:
        bin = bin_count[key]
        bin_sample = random.sample(bin, k=int(len(bin) * sr))
        # random.sample is used to sample instead of numpy.random.choice
        #   since random.sample is faster than numpy.random.choice
        # bin_sample = np.random.choice(bin,
        #                               size=int(len(bin) * sr))
        samples += list(bin_sample)
    np.random.shuffle(samples)
    return samples


def sgd(X, y, cur, sample_ind, lr):
    """Stochastic gradient descent

    Using stochastic gradient descent to learn y_i

    Args:
        X          - training data
        y          - the array of predicted y
        cur        - the current y_i to optimize
        sample_ind - the list of sample indexes
        lr         - the learning rate

    Return:
        new_yi     - the optimized y_i
    """
    new_yi = y[cur]
    i = cur
    for j in sample_ind:
        if i != j:
            new_yi = new_yi \
                     + lr * ((X[i] * y[j] - X[j] * y[i]) * X[j]
                             + (y[j] - y[i]) / ((j - i) ** 2))
    return new_yi


def main(path, n_iter, init, lr, sr, interval, n_proc):
    train_X, grdtruth_y, fc_y = load_data(path)
    n_entry = train_X.shape[0]

    pred_y = initialize(init, n_entry, fc_y)

    optimize(train_X, pred_y,
             n_iter=n_iter, interval=interval,
             sr=sr, lr=lr, n_proc=n_proc)

    evaluate(train_X, grdtruth_y)


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-f", "--file-input",
                      type="string",
                      dest="filepath",
                      default="data/exp1-2_cnv.txt")

    parser.add_option("-n", "--num-of-iteration",
                      type="int",
                      dest="num_iter",
                      default=1000)

    parser.add_option("-i", "--initialization",
                      type="string",
                      dest="init",
                      default="random")

    parser.add_option("-r", "--learning-rate",
                      type="float",
                      dest="lr",
                      default=0.001)

    parser.add_option("-t", "--interval",
                      type="int",
                      dest="interval",
                      default=5)

    parser.add_option("-s", "--sample-rate",
                      type="float",
                      dest="sr",
                      default=0.1)

    parser.add_option("-p", "--num-proc",
                      type="int",
                      dest="n_proc",
                      default="4")

    options, args = parser.parse_args()

    main(path=options.filepath,
         n_iter=options.num_iter,
         init=options.init,
         lr=options.lr,
         sr=options.sr,
         interval=options.interval,
         n_proc=options.n_proc)


