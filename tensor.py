import numpy as np
import tensorly as tl
import pandas as pd
from sparse import COO
from numba import jit
from typing import Optional
from sklearn.preprocessing import minmax_scale


ALS_MAX_ITERS = 100
ALS_FIT_CHANGE_TOL = 1e-4
TENSOR_MACHINE_EPSILON = 1e-5


@jit(cache=True, nopython=True)
def ALS_U(dimensions, rank, m, coords, data, A):
    U = np.zeros((dimensions[m], rank))
    for x, v in zip(coords, data):
        xu = np.full(rank, v)
        for n, xn in enumerate(x):
            if m == n:
                continue
            xu *= A[n][xn]
        U[x[m]] += xu
    return U

@jit(cache=True, nopython=True)
def norm_frobenius_X(data):
    nf_X = 0.
    for v in data:
        nf_X += np.square(v)
    return np.sqrt(nf_X)

@jit(cache=True, nopython=True)
def innerprod_X_x_reconst(coords, data, A, _lambda):
    innerprod = 0.
    for x, v in zip(coords, data):
        val_reconst = np.copy(_lambda)
        for n, xn in enumerate(x):
            val_reconst *= A[n][xn]
        innerprod += v * val_reconst.sum()
    return innerprod


class Tensor:
    def __init__(self, dimensions: list[int], non_zero_X: COO, rank: int, non_negative: bool = True):
        self.dimensions = dimensions
        self.mode_num = len(dimensions)
        self.rank = rank
        self.non_zero_X = non_zero_X
        self.non_negative = non_negative

        self.rand_init_A()
        self.ALS()

    def update(self, non_zero_dX: COO):
        self.non_zero_X += non_zero_dX
        self.rand_init_A()
        self.ALS()

    def ALS(self):
        oldfit = 0.
        for i in range(ALS_MAX_ITERS):
            for m in range(self.mode_num):
                H = np.ones(self.rank**2).reshape(self.rank, self.rank)
                for n in range(self.mode_num):
                    if m == n:
                        continue
                    H *= self.AtA[n]
                U = ALS_U(self.dimensions, self.rank, m, self.non_zero_X.coords.T, self.non_zero_X.data, self.A)

                if self.non_negative:
                    self.A[m] *= np.clip(U, a_min=10e-12, a_max=None) / np.clip(np.dot(self.A[m], H), a_min=10e-12, a_max=None)
                else:
                    #self.A[m] = np.linalg.solve(H.T, U.T).T
                    H += np.diag(np.full(self.rank, 1e-10))
                    self.A[m] = np.dot(U, np.linalg.pinv(H))
                
                self._lambda = self.A[m].max(axis=0)
                for r in range(self.rank):
                    if abs(self._lambda[r]) < TENSOR_MACHINE_EPSILON: self._lambda[r] = 1.0
                    else: self.A[m][:, r] /= self._lambda[r]
                self.AtA[m] = np.dot(self.A[m].T, self.A[m])

            newfit = self.fitness()
            if i > 0 and abs(oldfit - newfit) < ALS_FIT_CHANGE_TOL:
                break
            oldfit = newfit

    def SelectiveALS(self, non_zero_dX: COO):
        for m in range(self.mode_num):
            non_zero_indicies = np.unique(non_zero_dX.coords[0] if m == 0 else self.non_zero_X.coords[m])

            H = np.ones(self.rank**2).reshape(self.rank, self.rank)
            for n in range(self.mode_num):
                if m == n:
                    continue
                H *= self.AtA[n]
            if m == 0:
                U = ALS_U(self.dimensions, self.rank, m, non_zero_dX.coords.T, non_zero_dX.data, self.A)
            else:
                U = ALS_U(self.dimensions, self.rank, m, self.non_zero_X.coords.T, self.non_zero_X.data, self.A)

            if self.non_negative:
                Am_new = self.A[m][non_zero_indicies] * np.clip(U[non_zero_indicies], a_min=10e-12, a_max=None) / np.clip(np.dot(self.A[m][non_zero_indicies], H), a_min=10e-12, a_max=None)
            else:
                #self.A[m] = np.linalg.solve(H.T, U.T).T
                H += np.diag(np.full(self.rank, 1e-10))
                Am_new = np.dot(U[non_zero_indicies], np.linalg.pinv(H))

            if m == 0:
                Am_new += self.A[m][non_zero_indicies]
            self.A[m][non_zero_indicies] = Am_new
            
            self._lambda = self.A[m].max(axis=0)
            for r in range(self.rank):
                if abs(self._lambda[r]) < TENSOR_MACHINE_EPSILON: self._lambda[r] = 1.0
                else: self.A[m][:, r] /= self._lambda[r]
            self.AtA[m] = np.dot(self.A[m].T, self.A[m])

    def rand_init_A(self):
        rng = np.random.default_rng()
        self.A = [rng.random((dimension, self.rank)) for dimension in self.dimensions]
        self.AtA = [np.dot(factor.T, factor) for factor in self.A]
        self._lambda = np.ones(self.rank)

    def rmse(self):
        norm_X = norm_frobenius_X(self.non_zero_X.data)
        norm_reconst = self.norm_frobenius_reconst()
        innerprod = innerprod_X_x_reconst(self.non_zero_X.coords.T, self.non_zero_X.data, self.A, self._lambda)

        error_square = np.square(norm_X) + np.square(norm_reconst) - 2. * innerprod
        for d in self.dimensions:
            error_square /= d
        return np.sqrt(error_square)

    def fitness(self):
        norm_X = norm_frobenius_X(self.non_zero_X.data)
        norm_reconst = self.norm_frobenius_reconst()
        innerprod = innerprod_X_x_reconst(self.non_zero_X.coords.T, self.non_zero_X.data, self.A, self._lambda)

        return 1.- np.sqrt(abs(np.square(norm_X) + np.square(norm_reconst) - 2. * innerprod)) / norm_X
    
    def norm_frobenius_reconst(self):
        nf_reconst = np.dot(self._lambda.reshape(self.rank, 1), self._lambda.reshape(1, self.rank))
        for m in range(self.mode_num):
            nf_reconst *= self.AtA[m]
        return np.sqrt(abs(nf_reconst.sum()))
    

class Tensor_SNS_MAT(Tensor):
    def update(self, non_zero_dX: COO):
        self.non_zero_X += non_zero_dX
        self.ALS()


class Tensor_SNS_VEC(Tensor):
    def update(self, non_zero_dX: COO):
        self.non_zero_X += non_zero_dX
        self.SelectiveALS(non_zero_dX)


class TensorStream:
    def __init__(self, dimensions: list[int], T: int, rank: int, start_time: int, algo: str,
                 events: pd.DataFrame, category_labels: list[str], time_label: str, value_label: Optional[str] = None):
        
        if value_label == None:
            events = events.reindex(columns = [time_label] + category_labels)
        else:
            events = events.reindex(columns = [time_label] + category_labels + [value_label])
            events = events[events[value_label] != 0.0]
            events[value_label] = minmax_scale(events[value_label].to_numpy())

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

        if value_label == None:
            non_zero_X = COO(coords = start_events[[time_label] + category_labels].to_numpy().T,
                             data = np.ones(len(start_events)), shape = dimensions)
        else:
            non_zero_X = COO(coords = start_events[[time_label] + category_labels].to_numpy().T,
                             data = start_events[value_label], shape = dimensions)

        if algo == "ALS":
            self.tensor = Tensor(dimensions, non_zero_X, rank)
        elif algo == "SNS_MAT":
            self.tensor = Tensor_SNS_MAT(dimensions, non_zero_X, rank)
        elif algo == "SNS_VEC":
            self.tensor = Tensor_SNS_VEC(dimensions, non_zero_X, rank)
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

        if self.value_label == None:
            p_non_zero_dX = COO(coords = p_events[[self.time_label] + self.category_labels].to_numpy().T,
                                data = np.ones(len(p_events)), shape = self.tensor.dimensions)
        else:
            p_non_zero_dX = COO(coords = p_events[[self.time_label] + self.category_labels].to_numpy().T,
                                data = p_events[self.value_label].to_numpy().ravel(), shape = self.tensor.dimensions)

        m_events = self.events[(self.events[self.time_label] >= self.current_time - self.T * self.W) &
                               (self.events[self.time_label] < self.current_time)].copy()
        m_events[self.time_label] += self.T * self.W - self.current_time
        m_events = m_events[m_events[self.time_label] % self.T == 0]
        m_events[self.time_label] = m_events[self.time_label] // self.T

        if self.value_label == None:
            m_non_zero_dX = COO(coords = m_events[[self.time_label] + self.category_labels].to_numpy().T,
                                data = np.ones(len(m_events)), shape = self.tensor.dimensions)
        else:
            m_non_zero_dX = COO(coords = m_events[[self.time_label] + self.category_labels].to_numpy().T,
                                data = m_events[self.value_label].to_numpy().ravel(), shape = self.tensor.dimensions)
            
        self.non_zero_dX += p_non_zero_dX - m_non_zero_dX


    def updateTensor(self):
        self.tensor.update(self.non_zero_dX)
        self.non_zero_dX = COO(coords=[], shape=self.tensor.dimensions)