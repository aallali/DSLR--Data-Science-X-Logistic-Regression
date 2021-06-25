from DSLR.maths import count_, mean_, std_, min_, percentile_, max_
from DSLR.utils import load_csv
import numpy as np
import argparse


def describe(filename):
    dataset = load_csv(filename)
    features = dataset[0]
    dataset = dataset[1:]
    print(
        f'{"":15} |{"Count":>12} |{"Mean":>12} |{"Std":>12} |{"Min":>12} |{"25%":>12} |{"50%":>12} |{"75%":>12} |{"Max":>12}')
    for i in range(0, len(features)):
        print(f'{features[i]:15.15}', end=' |')
        try:
            data = np.array(dataset[:, i], dtype=float)
            data = data[~np.isnan(data)]
            if not data.any():
                raise Exception()
            print(f'{count_(data):>12.4f}', end=' |')
            print(f'{mean_(data):>12.4f}', end=' |')
            print(f'{std_(data):>12.4f}', end=' |')
            print(f'{min_(data):>12.4f}', end=' |')
            print(f'{percentile_(data, 25):>12.4f}', end=' |')
            print(f'{percentile_(data, 50):>12.4f}', end=' |')
            print(f'{percentile_(data, 75):>12.4f}', end=' |')
            print(f'{max_(data):>12.4f}')
        except:
            print(f'{count_(dataset[:, i]):>12.4f}', end=' |')
            print(f'{"No numerical value to display":>60}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="input dataset")
    args = parser.parse_args()
    describe(args.dataset)
