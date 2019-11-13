import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM():
    def __init__(self, kernel=linear_kernel, epochs=10000, C=None):
        self.epochs = epochs
        self.kernel = kernel
        self.C = C

    def fit(self, data, label):
        n, m = data.shape

        # Gram Matrix
        K = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self.kernel(data[i], data[j])

        # Initiate Variables
        P = cvxopt.matrix(np.outer(label, label) * K)
        q = cvxopt.matrix(np.ones(n) * -1)
        A = cvxopt.matrix(label, (1,n))
        b = cvxopt.matrix(0.0)
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n) * -1))
            h = cvxopt.matrix(np.zeros(n))
        else:
            tmp1 = np.diag(np.ones(n) * -1)
            tmp2 = np.identity(n)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n)
            tmp2 = np.ones(n) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(solution['x'])

        # Support Vectors
        sv = a > 0
        ind = np.arange(len(a))[sv]
        self.a_svm = a[sv]
        self.svm = data[sv]
        self.label_svm = label[sv]
        self.b_svm = 0.0

        # Calculate b value
        for n in range(len(self.a_svm)):
            self.b_svm += self.label_svm[n]
            self.b_svm -= np.sum(self.a_svm * self.label_svm * K[ind[n],sv])
        self.b_svm /= len(self.a_svm)

        # Calculate Weight
        if self.kernel == linear_kernel:
            self.w = np.zeros(m)
            for n in range(len(self.a_svm)):
                self.w += self.a_svm[n] * self.label_svm[n] * self.svm[n]
        else:
            self.w = None

    def project(self, data):
        if self.w is not None:
            return np.dot(data, self.w) + self.b_svm
        else:
            label_predict = np.zeros(len(data))
            for i in range(len(data)):
                s = 0
                for a, sv_y, sv in zip(self.a_svm, self.label_svm, self.svm):
                    s += a * sv_y * self.kernel(data[i], sv)
                label_predict[i] = s
            return label_predict + self.b_svm

    def predict(self, data):
        return np.sign(self.project(data))
