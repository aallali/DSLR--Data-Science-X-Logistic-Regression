# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_predict.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aallali <hi@allali.me>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2021/06/26 12:48:57 by aallali           #+#    #+#              #
#    Updated: 2021/06/26 12:48:59 by aallali          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import sys
from collections import OrderedDict
from DSLR.model import LogisticRegressionOVR
from DSLR.utils import read_file

if __name__ == "__main__":
    # hptest = pd.read_csv(sys.argv[1], index_col="Index")
    hptest = read_file(sys.argv[1])
    thetas = np.load(sys.argv[2], allow_pickle=True)
    logreg = LogisticRegressionOVR(data=hptest, prediction=True)
    predicts = logreg.predict(thetas)
    houses = pd.DataFrame(OrderedDict({'Index': range(len(predicts)), 'Hogwarts House': predicts}))
    houses.to_csv('houses.csv', index=False)
    print("Predictions saved to 'houses.csv'")
