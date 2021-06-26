# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    utils.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aallali <hi@allali.me>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2021/06/26 12:49:59 by aallali           #+#    #+#              #
#    Updated: 2021/06/26 12:50:00 by aallali          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import csv
import numpy as np
import pandas as pd
import sys

def load_csv(filename):
    dataset = list()
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        try:
            for _ in reader:
                row = list()
                for value in _:
                    try:
                        value = float(value)
                    except:
                        if not value:
                            value = np.nan
                    row.append(value)
                dataset.append(row)
        except csv.Error as e:
            print(f'file {filename}, line {reader.line_num}: {e}')
    return np.array(dataset, dtype=object)


def read_file(filname):
    try:
        data = pd.read_csv(filname)
    except:
        sys.exit("File doesn't exist")
    data = data.drop(['First Name', 'Last Name', 'Birthday', 'Index'], axis=1)
    data['Best Hand'] = data['Best Hand'].map({'Right': 0, 'Left': 1})
    for key in data:
        if (key != "Hogwarts House"):
            data.fillna(value={key: data[key].mean()}, inplace=True)
    return (data)
