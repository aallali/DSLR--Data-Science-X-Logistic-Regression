import numpy as np
import pandas as pd
import sys
from collections import OrderedDict
from DSLR.model import LogisticRegressionOVR


if __name__ == "__main__":
    hptest = pd.read_csv(sys.argv[1], index_col="Index")
    thetas = np.load(sys.argv[2], allow_pickle=True)
    logreg = LogisticRegressionOVR(data=hptest, prediction=True)
    predicts = logreg.predict(thetas)
    houses = pd.DataFrame(OrderedDict({'Index': range(len(predicts)), 'Hogwarts House': predicts}))
    houses.to_csv('houses.csv', index=False)
    print("Predictions saved to 'houses.csv'")

