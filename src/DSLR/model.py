import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionOVR:
    def __init__(self, data, w=[], eta=2e-5, n_iter=3000, v=False, prediction=False):
        self.eta = eta
        self.n_iter = n_iter
        self.w = w
        self.v = v
        self.X = None
        self.y = None
        self._processing(data, prediction)

    def _scaling(self, X):
        for i in range(len(X)):
            X[i] = (X[i] - X.mean()) / X.std()
        return X

    def _processing(self, hptrain, pred=True):
        if pred == False:
            hptrain['Best Hand'] = hptrain['Best Hand'].map({'Right': 0, 'Left': 1})
            hptrain = hptrain.drop(['Astronomy', 'Care of Magical Creatures', 'Charms'], axis=1)
            hptrain = hptrain.dropna()
            hp_features = np.array((hptrain.iloc[:, 4:]))
            hp_labels = np.array(hptrain.loc[:, "Hogwarts House"])
            np.apply_along_axis(self._scaling, 0, hp_features)
            hp_features = np.insert(hp_features, 0, 1, axis=1)
            self.X = hp_features
            self.y = hp_labels
            return hp_features, hp_labels
        else:
            hptrain['Best Hand'] = hptrain['Best Hand'].map({'Right': 0, 'Left': 1})
            hptrain = hptrain.drop(['Astronomy', 'Care of Magical Creatures', 'Charms'], axis=1)

            hptrain = hptrain.iloc[:, 4:]
            hptrain['Flying'] = hptrain['Flying'].fillna(22.0060)
            hptrain['Potions'] = hptrain['Potions'].fillna(5.9529)
            hptrain['Transfiguration'] = hptrain['Transfiguration'].fillna(1030.0939)
            hptrain['History of Magic'] = hptrain['History of Magic'].fillna(2.9604)
            hptrain['Ancient Runes'] = hptrain['Ancient Runes'].fillna(495.8477)
            hptrain['Muggle Studies'] = hptrain['Muggle Studies'].fillna(224.5838)
            hptrain['Divination'] = hptrain['Divination'].fillna(3.1517)
            hptrain['Arithmancy'] = hptrain['Arithmancy'].fillna(49635.2064)

            hptrain = hptrain.fillna(1)

            hp_features = np.array(hptrain)
            np.apply_along_axis(self._scaling, 0, hp_features)
            hp_features = np.insert(hp_features, 0, 1, axis=1)

            self.X = hp_features
            return hp_features, None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self):
        f, axs = plt.subplots(2, 2, figsize=(10, 8))
        X, y = self.X, self.y
        cost = []
        index = []
        for (n, i) in enumerate(np.unique(y)):
            print(f' Training against : {i}')
            y_copy = np.where(y == i, 1, 0)
            w = np.ones(X.shape[1])
            for _ in range(self.n_iter):
                index.append(_)
                output = X.dot(w)
                sig = self._sigmoid(output)
                errors = y_copy - sig
                cost.append(abs(errors.mean()))
                gradient = np.dot(X.T, errors)
                w += self.eta * gradient
                if ((_ * 100) / self.n_iter) % 10 == 0:
                    print(f'training progress {"{:.2f}".format((_ * 100) / self.n_iter)}%', end='\r')

            self.w.append((w, i))
            if self.v == True:
                if n == 0:
                    xx = 0
                    yy = 0
                elif n == 1:
                    xx = 0
                    yy = 1
                elif n == 2:
                    xx = 1
                    yy = 0
                else:
                    xx = 1
                    yy = 1
                axs[xx, yy].set_title(i)
                axs[xx, yy].plot(index, cost, 'b-')
            cost = []
            index = []
        if self.v == True:
            plt.show()

        return self.w

    def _predict_one(self, x, weights=None):
        if weights == None:
            weights = self.w
        return max((x.dot(w), c) for w, c in weights)[1]

    def predict(self, weights=None):

        if weights == None:
            weights = self.w
        X, y = self.X, self.y
        return [self._predict_one(i, weights) for i in X]

    def score(self):
        return sum(self.predict() == self.y) / len(self.y)
