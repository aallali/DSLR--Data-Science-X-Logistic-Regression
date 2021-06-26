# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    scatter_plot.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aallali <hi@allali.me>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2021/06/26 12:49:24 by aallali           #+#    #+#              #
#    Updated: 2021/06/26 12:50:42 by aallali          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from DSLR.utils import load_csv
import numpy as np
import matplotlib.pyplot as plt


def scatter_plot(X, y, legend, xlabel, ylabel):
    plt.scatter(X[:327], y[:327], color='red', alpha=0.5)  # Grynffindor House
    plt.scatter(X[327:856], y[327:856], color='yellow', alpha=0.5)  
    # Hufflepuff House
    plt.scatter(X[856:1299], y[856:1299], color='blue', alpha=0.5)  
    # Ravenclaw House
    plt.scatter(X[1299:], y[1299:], color='green', alpha=0.5)  # Slytherin House

    plt.legend(legend, loc='upper right', frameon=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


if __name__ == '__main__':
    """
    after looking at the PairPlot graph you will notice that 
        "Astronomy" and "Defense Against the Dark Arts"
    are the only two smilar features
    """
    dataset = load_csv('./datasets/dataset_train.csv')
    data = dataset[1:, :]
    data = data[data[:, 1].argsort()]

    X = np.array(data[:, 7], dtype=float)  # get the Atronomy row data
    y = np.array(data[:, 9], dtype=float)  # get the "Defennse Again ..." row data
    legend = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']  # set the "Hogwarts House"'s \
    # names manually
    scatter_plot(X, y, legend=legend, xlabel=dataset[0, 7], ylabel=dataset[0, 9])
