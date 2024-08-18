import numpy as np
import tensorly as tl
from scipy import linalg


ALS_MAX_ITERS = 20
ALS_FIT_CHANGE_TOL = 0.01


class Tensor:
    def __init__(self, dimensions: list[int], X: np.ndarray, rank: int):
        self.dimensions = dimensions
        self.X = X
        self.rank = rank

        self.rand_init_A()
        self.ALS()

    def update(self, dX):
        self.X += dX
        self.rand_init_A()
        self.ALS()

    def ALS(self):
        #self.Ax = tl.decomposition.parafac(self.X, rank=self.rank)
        oldfit = 0.
        for i in range(ALS_MAX_ITERS):
            for m in range(len(self.dimensions)):
                U = np.ones(self.rank).reshape(1, self.rank)
                H = np.ones(self.rank**2).reshape(self.rank, self.rank)
                for n in range(len(self.dimensions)):
                    if m == n:
                        continue
                    U = linalg.khatri_rao(U, self.A[n])
                    H *= self.AtA[n]
                self.A[m] = np.dot(np.dot(tl.unfold(self.X, m), U), np.linalg.pinv(H))
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
        return np.sqrt(np.mean(np.square(self.X - reconst_X)))

    def fitness(self):
        reconst_X = self.reconst()
        return 1. - np.sqrt(np.sum(np.square(self.X - reconst_X))) / np.sqrt(np.sum(np.square(self.X)))
    
    def reconst(self):
        #return tl.cp_to_tensor(self.Ax)
        U = np.ones(self.rank).reshape(1, self.rank)
        for factor in self.A:
            U = linalg.khatri_rao(U, factor)
        
        return U.sum(axis=1).reshape(self.dimensions)
    

class Tensor_SNS_MAT(Tensor):

    def update(self, dX):
        self.X += dX
        self.ALS()


class TensorStream:
    def __init__(self, events: list[list[tuple[list[int], float]]], dimensions: list[int], T: int, rank: int, start_time: int, algo: str):
        self.events = events
        self.T = T
        self.W = dimensions[-1]
        self.current_time = start_time + T * self.W - 1

        X = np.zeros(np.prod(dimensions)).reshape(dimensions)
        self.dX = np.zeros(np.prod(dimensions)).reshape(dimensions)

        for time, event in enumerate(events[start_time:self.current_time+1]):
            for ele_indices, value in event:
                ele = X
                for index in ele_indices:
                    ele = ele[index]
                ele[time // T] = np.float64(value)

        if algo == "ALS":
            self.tensor = Tensor(dimensions, X, rank)
        elif algo == "SNS_MAT":
            self.tensor = Tensor_SNS_MAT(dimensions, X, rank)
        else:
            print("¯\_(ツ)_/¯")
            exit(1)
    
    
    def updateCurrent(self):
        self.current_time += 1
        for ele_indices, value in self.events[self.current_time]:
            ele = self.dX
            for index in ele_indices:
                ele = ele[index]
            ele[self.W - 1] += value

        for i, event in enumerate(self.events[self.current_time - self.T : self.current_time - self.T * self.W : -self.T]):
            for ele_indices, value in event:
                ele = self.dX
                for index in ele_indices:
                    ele = ele[index]
                ele[self.W - 1 - i] -= value
                ele[self.W - 2 - i] += value
        
        for ele_indices, value in self.events[self.current_time - self.T * self.W]:
            ele = self.dX
            for index in ele_indices:
                ele = ele[index]
            ele[0] -= value

    def updateTensor(self):
        self.tensor.update(self.dX)
        self.dX = np.zeros(np.prod(self.tensor.dimensions)).reshape(self.tensor.dimensions)