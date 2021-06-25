import numpy as np
import pandas as pd
from DSLR.model import LogisticRegressionOVR
import argparse
from DSLR.utils import read_file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file',
                        help="datasets to train")
    parser.add_argument('-v', action='store_true',
                        help="visualize cost/iteration", default=False)
    parser.add_argument('-n', action='store',
                        help="number of iteration", type=int, default=30000)
    args = parser.parse_args()
    # hptrain = pd.read_csv(args.file, index_col="Index")
    hptrain = read_file(args.file)
    logreg = LogisticRegressionOVR(data=hptrain, v=args.v, n_iter=args.n, prediction=False)
    weights = logreg.fit()
    np.save("weights", weights)
    print(f"Weights saved in 'weights.npy',\naccuracy : {'{:.2f}'.format(logreg.score() * 100)} %")
