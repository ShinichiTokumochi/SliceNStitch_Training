import numpy as np
import tensorly as tl
from scipy import linalg
import pandas as pd
import time
from itertools import product
from sparse import COO
from numba import jit


ALS_MAX_ITERS = 20
ALS_FIT_CHANGE_TOL = 0.01


class Tensor:
    def __init__(self, dimensions: list[int], non_zero_X: COO, rank: int):
        self.dimensions = dimensions
        self.rank = rank

        self.X = non_zero_X.todense()
        self.non_zero_X = non_zero_X

        self.rand_init_A()
        self.ALS()

    def update(self, non_zero_dX: COO):
        self.updateX(non_zero_dX)
        self.rand_init_A()
        self.ALS()

    def updateX(self, non_zero_dX: COO):
        self.X = self.X + non_zero_dX
        self.non_zero_X += non_zero_dX

    def ALS(self):

        @jit(cache=True, nopython=True)
        def ALS_U(dimensions, rank, m, coords, data, A):
            U = np.ones(dimensions[m] * rank).reshape(dimensions[m], rank)
            for x, v in zip(coords, data):
                xu = np.full(rank, v)
                for n, xn in enumerate(x):
                    if m == n:
                        continue
                    xu *= A[n][xn]
                U[x[m]] += xu
            return U

        oldfit = 0.
        for i in range(ALS_MAX_ITERS):
            for m in range(len(self.dimensions)):
                H = np.ones(self.rank**2).reshape(self.rank, self.rank)
                for n in range(len(self.dimensions)):
                    if m == n:
                        continue
                    H *= self.AtA[n]
                U = ALS_U(self.dimensions, self.rank, m, self.non_zero_X.coords.T, self.non_zero_X.data, self.A)
                    
                self.A[m] = np.dot(U, np.linalg.pinv(H))
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
        reconst_X = self.reconst()
        return np.sqrt(np.mean(np.square(self.X.ravel() - reconst_X)))

    def fitness(self):
        reconst_X = self.reconst()
        return 1. - np.sqrt(np.sum(np.square(self.X.ravel() - reconst_X))) / np.sqrt(np.sum(np.square(self.X)))
    
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
        events = events[events[value_label] != 0.0]

        self.T = T
        self.W = dimensions[0]
        self.current_time = start_time + T * self.W - 1
        self.events = events
        self.category_labels = category_labels
        self.time_label = time_label
        self.value_label = value_label
        self.non_zero_dX = COO(coords=[], shape=dimensions)

        start_events = events[(events[time_label] >= start_time) & (events[time_label] <= self.current_time)].copy()
        start_events[time_label] = (start_events[time_label] - start_time) // T
        non_zero_X = COO(coords = start_events[[time_label] + category_labels].to_numpy().T,
                         data = start_events[value_label].to_numpy().ravel(), shape = dimensions)

        if algo == "ALS":
            self.tensor = Tensor(dimensions, non_zero_X, rank)
        elif algo == "SNS_MAT":
            self.tensor = Tensor_SNS_MAT(dimensions, non_zero_X, rank)
        else:
            print("¯\_(ツ)_/¯")
            exit(1)
    

    def updateCurrent(self):
        self.current_time += 1

        p_events = self.events[(self.events[self.time_label] > self.current_time - self.T * self.W) &
                               (self.events[self.time_label] <= self.current_time)].copy()
        p_events[self.time_label] += self.T * self.W - 1 - self.current_time
        p_events = p_events[p_events[self.time_label] % self.T == self.T - 1]
        p_events[self.time_label] = p_events[self.time_label] // self.T
        p_non_zero_dX = COO(coords = p_events[[self.time_label] + self.category_labels].to_numpy().T,
                            data = p_events[self.value_label].to_numpy().ravel(), shape = self.tensor.dimensions)

        m_events = self.events[(self.events[self.time_label] >= self.current_time - self.T * self.W) &
                               (self.events[self.time_label] < self.current_time)].copy()
        m_events[self.time_label] += self.T * self.W - self.current_time
        m_events = m_events[m_events[self.time_label] % self.T == 0]
        m_events[self.time_label] = m_events[self.time_label] // self.T
        m_non_zero_dX = COO(coords = m_events[[self.time_label] + self.category_labels].to_numpy().T,
                            data = m_events[self.value_label].to_numpy().ravel(), shape = self.tensor.dimensions)

        self.non_zero_dX += p_non_zero_dX - m_non_zero_dX


    def updateTensor(self):
        self.tensor.update(self.non_zero_dX)
        self.non_zero_dX = COO(coords=[], shape=self.tensor.dimensions)