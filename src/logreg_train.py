import numpy as np
import pandas as pd
import sys
from DSLR.model import LogisticRegressionOVR

if __name__ == "__main__":
    hptrain = pd.read_csv(sys.argv[1], index_col="Index")
    logreg = LogisticRegressionOVR(data=hptrain, v=True, n_iter=30000)
    weights = logreg.fit()
    np.save("weights", weights)
    print(f"Weights saved in 'weights.npy',\naccuracy : {'{:.2f}'.format(logreg.score() * 100)} %")
