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

    data = preprocess(data)
    data.sort_values("StartPos", inplace=True)

    tpm = pd.DataFrame.as_matrix(data.TPM)
    conuvar = pd.DataFrame.as_matrix(data.True_CNV)
    foldchange = pd.DataFrame.as_matrix(data.FoldChange)

    print("Done!")

    return tpm, conuvar, foldchange


def preprocess(data):
    """Data preprocessing

    Inplace data pre-processing on Pandas level.
    Pre-process data by following rules:

        1 - Remove all entries whose fc < 0.1
        2 - Sort by the start position

    Args:
        data - the dataset to work on
    """
    print("\t[Preprocess - Under construction]")

    region = pd.DataFrame.as_matrix(data.Region)
    start_pos = [int(x.split("_")[1]) // 1000 for x in region]
    idx = data.index.tolist()
    df_region = pd.DataFrame(index=idx, data=start_pos,
                             columns=["StartPos"], dtype=np.int64)
    return data.join(df_region)


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
        return np.array(fc)
    elif init == "uni":
        return np.ones(n_entry)
    else:
        sys.exit("Cannot identify initialization param!")


def optimize(train_X, pred_y,
             n_iter=1000, interval=5, sr=0.01,
             lr=0.001, n_proc=4, alpha=0.001):
    print("Optimizing ...")
    bin_count = {}

    # Creating bin_count for sampling
    for i in range(len(train_X)):
        bin = int(train_X[i]/interval)
        if bin in bin_count:
            bin_count[bin].append(i)
        else:
            bin_count[bin] = [i]

    prev_sum_y = sum(pred_y)

    yi_index = list(range(len(pred_y)))  # To be shuffled later

    for n in range(n_iter):
        # The n-th iteration
        np.random.shuffle(yi_index)
        train_samples = sample(bin_count, sr)  # contain samples' indexes

        # yi_index_splits = np.array_split(yi_index, n_proc)  # Split workload
        # pool = mp.Pool(processes=n_proc)

        # Multi-processing
        # for i in range(n_proc):
        #     pool.apply_async(multiproc_worker,
        #                      (train_X, pred_y, yi_index_splits[i],
        #                       train_samples, lr, alpha))
        # pool.close()
        # pool.join()

        # Single-processing
        multiproc_worker(train_X, pred_y, yi_index, train_samples, lr, alpha)

        if not n % 10:
            delta = sum(pred_y) - prev_sum_y
            prev_sum_y = sum(pred_y)
            msg = "\tOptimizing: {:.3f}%," \
                  "\tDelta Sum y: {:.3f}," \
                  "\tSum y: {:3f}." \
                .format(100 * n / n_iter, delta, sum(pred_y))
            # sys.stdout.write('\r' +msg)
            print(msg)

    print("Done!")


def multiproc_worker(X, y, yi_chunk, samples, lr, alpha):
    """Stochastic gradient descent

    Worker of multi-processing using stochastic gradient
        descent to learn yi. yi are modified in place.

    Args:
        X          - training data
        y          - the array of predicted y
        yi_chunk   - the chunk of yi this worker is in charge of
        samples    - the list of sample indexes
        lr         - the learning rate
        alpha      - the scaling factor of 2nd term
    """
    for i in yi_chunk:
        for j in samples:
            if i != j:
                d = lr * (
                    (X[i] * y[j] - X[j] * y[i]) * X[j] / (max(X[i], X[j]) ** 2)
                    + alpha * (y[j] - y[i]) / ((j - i) ** 2)
                )
                y[i] += d


def evaluate(path, ground_truth, pred_y, foldchange):
    """Evaluate the predicted result

    Evalute the predicted result

    Args:
        path         - the output file path
        ground_truth - the ground truth of copy number variation
        pred_y       - the predicted value
        foldchange   - the foldchange metrics

    Return:
        don't know what to return
    """
    # print("=== EVALUATION - Under construction ===")
    print("Printing to {}".format(path))
    with open(path, "w") as fout:
        for i in range(len(ground_truth)):
            print("{:d} {:3f} {:3f}"
                  .format(ground_truth[i], pred_y[i], foldchange[i]),
                  file=fout)
    print("Done!")


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
        # bin_sample = np.random.choice(bin, size=int(len(bin) * sr))
        samples += list(bin_sample)
    np.random.shuffle(samples)  # However, numpy has a better shuffle
    return samples


def main(infile, n_iter, init, lr, sr,
         alpha, interval, n_proc, outfile):
    train_X, grdtruth_y, fc_y = load_data(infile)
    n_entry = train_X.shape[0]

    pred_y = initialize(init, n_entry, fc_y)

    optimize(train_X=train_X, pred_y=pred_y,
             n_iter=n_iter, interval=interval,
             sr=sr, lr=lr, n_proc=n_proc, alpha=alpha)

    evaluate(outfile, grdtruth_y, pred_y, fc_y)


if __name__ == "__main__":
    parser = optparse.OptionParser()

    parser.add_option("-f", "--file-input",
                      type="string",
                      dest="infile",
                      default="data/exp1-2_cnv.txt")

    parser.add_option("-n", "--num-of-iteration",
                      type="int",
                      dest="num_iter",
                      default=1000)

    parser.add_option("-i", "--initialization",
                      type="string",
                      dest="init",
                      default="uni")

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

    parser.add_option("-o", "--file-output",
                      type="string",
                      dest="outfile",
                      default="data/output.txt")

    parser.add_option("-a", "--alpha",
                      type="float",
                      dest="alpha",
                      default="0.001")

    options, args = parser.parse_args()

    main(infile=options.infile,
         n_iter=options.num_iter,
         init=options.init,
         lr=options.lr,
         sr=options.sr,
         alpha=options.alpha,
         interval=options.interval,
         n_proc=options.n_proc,
         outfile=options.outfile)


