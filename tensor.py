import numpy as np
import tensorly as tl
from scipy import linalg
import pandas as pd
import time
from itertools import product


ALS_MAX_ITERS = 20
ALS_FIT_CHANGE_TOL = 0.01


class Tensor:
    def __init__(self, dimensions: list[int], start_events: pd.DataFrame, rank: int):
        self.dimensions = dimensions
        self.rank = rank

        start_events.columns = list(range(len(start_events.columns)))
        self.X = pd.DataFrame(list(product(*[range(d) for d in dimensions])))
        self.X[len(self.X.columns)] = 0.0
        self.X = pd.concat([self.X, start_events]).groupby(list(range(len(self.X.columns) - 1)), as_index=False).sum()
        self.non_zero_X = start_events

        self.rand_init_A()
        self.ALS()

    def update(self, update_events: pd.DataFrame):
        self.updateX(update_events)
        self.rand_init_A()
        self.ALS()

    def updateX(self, update_events: pd.DataFrame):
        update_events.columns = list(range(len(update_events.columns)))
        self.X = pd.concat([self.X, update_events]).groupby(list(range(len(self.X.columns) - 1)), as_index=False).sum()
        self.non_zero_X = pd.concat([self.non_zero_X, update_events]).groupby(list(range(len(self.X.columns) - 1)), as_index=False).sum()
        self.non_zero_X = self.non_zero_X[self.non_zero_X.iloc[:, -1] != 0.0]

    def ALS(self):
        #self.Ax = tl.decomposition.parafac(self.X, rank=self.rank)
        oldfit = 0.
        for i in range(ALS_MAX_ITERS):
            for m in range(len(self.dimensions)):
                H = np.ones(self.rank**2).reshape(self.rank, self.rank)
                for n in range(len(self.dimensions)):
                    if m == n:
                        continue
                    H *= self.AtA[n]

                XU = np.ones(self.dimensions[m] * self.rank).reshape(self.dimensions[m], self.rank)
                for x in self.non_zero_X.itertuples():
                    xu = np.full(self.rank, x[-1])
                    for n in range(len(self.dimensions)):
                        if m == n:
                            continue
                        xu *= self.A[n][x[n + 1]]
                    XU[x[m + 1]] += xu

                self.A[m] = np.dot(XU, np.linalg.pinv(H))
                self.AtA[m] = np.dot(self.A[m].T, self.A[m])

            newfit = self.fitness()
            if i > 0 and abs(oldfit - newfit) < ALS_FIT_CHANGE_TOL:
                break
            oldfit = newfit

    def rand_init_A(self):
        rng = np.random.default_rng()
        self.A = [rng.random((dimension, self.rank)) for dimension in self.dimensions]
        self.AtA = [np.dot(factor.T, factor) for factor in self.A]

    def rmse(self):
        original_X = self.X.iloc[:, -1].to_numpy().ravel()
        reconst_X = self.reconst()
        return np.sqrt(np.mean(np.square(original_X - reconst_X)))

    def fitness(self):
        original_X = self.X.iloc[:, -1].to_numpy().ravel()
        reconst_X = self.reconst()
        return 1. - np.sqrt(np.sum(np.square(original_X - reconst_X))) / np.sqrt(np.sum(np.square(original_X)))
    
    def reconst(self):
        #return tl.cp_to_tensor(self.Ax)
        U = np.ones(self.rank).reshape(1, self.rank)
        for factor in self.A:
            U = linalg.khatri_rao(U, factor)
        
        return U.sum(axis=1)
    

class Tensor_SNS_MAT(Tensor):

    def update(self, update_events: pd.DataFrame):
        self.updateX(update_events)
        self.ALS()


class Tensor_SNS_VEC(Tensor):
    pass


class TensorStream:
    def __init__(self, dimensions: list[int], T: int, rank: int, start_time: int, algo: str,
                 events: pd.DataFrame, category_labels: list[str], time_label: str, value_label: str = None):
        
        events = events.reindex(columns = [time_label] + category_labels + [value_label])

        self.T = T
        self.W = dimensions[0]
        self.current_time = start_time + T * self.W - 1
        self.events = events
        self.current_events = None
        self.time_label = time_label
        self.value_label = value_label

        start_events = events[(events[time_label] >= start_time) & (events[time_label] <= self.current_time)].copy()
        start_events[time_label] = (start_events[time_label] - start_time) // T

        if algo == "ALS":
            self.tensor = Tensor(dimensions, start_events, rank)
        elif algo == "SNS_MAT":
            self.tensor = Tensor_SNS_MAT(dimensions, start_events, rank)
        else:
            print("¯\_(ツ)_/¯")
            exit(1)
    

    def updateCurrent(self):
        self.current_time += 1

        plus_events = self.events[(self.events[self.time_label] > self.current_time - self.T * self.W) &
                                  (self.events[self.time_label] <= self.current_time)].copy()
        plus_events[self.time_label] += self.T * self.W - 1 - self.current_time
        plus_events = plus_events[plus_events[self.time_label] % self.T == self.T - 1]
        plus_events[self.time_label] = plus_events[self.time_label] // self.T

        minus_events = self.events[(self.events[self.time_label] >= self.current_time - self.T * self.W) &
                                   (self.events[self.time_label] < self.current_time)].copy()
        minus_events[self.time_label] += self.T * self.W - self.current_time
        minus_events = minus_events[minus_events[self.time_label] % self.T == 0]
        minus_events[self.time_label] = minus_events[self.time_label] // self.T
        minus_events[self.value_label] = -minus_events[self.value_label]

        if self.current_events == None:
            self.current_events = pd.concat([plus_events, minus_events])
        else:
            self.current_events = pd.concat([self.current_events, plus_events, minus_events])

    def updateTensor(self):
        self.tensor.update(self.current_events)
        self.current_events = None