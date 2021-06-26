# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    histogram.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aallali <hi@allali.me>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2021/06/26 12:49:08 by aallali           #+#    #+#              #
#    Updated: 2021/06/26 12:49:14 by aallali          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
from DSLR.utils import read_file
import argparse

def split_data_by_house(data, key):
    frst_h = []
    sec_h = []
    th_h = []
    four_h = []
    for i in range(data[key].shape[0]):
        if (pd.notna(data[key][i])):
            if (data['Hogwarts House'][i] == "Gryffindor"):
                frst_h.append(data[key][i])
            elif (data['Hogwarts House'][i] == "Slytherin"):
                sec_h.append(data[key][i])
            elif (data['Hogwarts House'][i] == "Hufflepuff"):
                th_h.append(data[key][i])
            else:
                four_h.append(data[key][i])
    return (frst_h, sec_h, th_h, four_h)


def show_histogramme(data):
    f, axs = plt.subplots(2, 7, figsize=(18, 14))
    i = 0
    j = 0
    for key in data:
        if (j == 7):
            j = 0
            i += 1
        if (key != "Hogwarts House"):
            first_h, sec_h, th_h, four_h = split_data_by_house(data, key)
            axs[i, j].set_title(key)
            axs[i, j].hist(first_h, bins='auto', facecolor='red', alpha=0.4, label='Gryffindor')
            axs[i, j].hist(sec_h, bins='auto', facecolor='green', alpha=0.5, label='Slytherin')
            axs[i, j].hist(th_h, bins='auto', facecolor='yellow', alpha=0.5, label='Hufflepuff')
            axs[i, j].hist(four_h, bins='auto', facecolor='blue', alpha=0.3, label='Ravenclaw')
            axs[i, j].legend(frameon=False)
            j += 1
    plt.show()


def show_most_homogenous_feat(data):
    f, ax = plt.subplots(figsize=(10, 6))
    first_h, sec_h, th_h, four_h = split_data_by_house(data, "Care of Magical Creatures")
    ax.set_title("Most homogenous feature Care of Magical Creatures")
    ax.hist(first_h, bins='auto', facecolor='red', alpha=0.4, label='Gryffindor')
    ax.hist(sec_h, bins='auto', facecolor='green', alpha=0.5, label='Slytherin')
    ax.hist(th_h, bins='auto', facecolor='yellow', alpha=0.5, label='Hufflepuff')
    ax.hist(four_h, bins='auto', facecolor='blue', alpha=0.3, label='Ravenclaw')
    ax.legend(frameon=False)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file',
                        help="datasets to train")
    parser.add_argument('-all', action='store_true',
                        help="plot all histograms ", default=False)
    args = parser.parse_args()
    data = read_file(args.file)
    if args.all == True:
        show_histogramme(data)
    show_most_homogenous_feat(data)


if __name__ == "__main__":
    main()
