import cvxpy as cp
from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import randomized_svd
import numpy as np

def svd_naive(X, rank=1):
    U, S, V = randomized_svd(X, rank)
    L = U @ np.diag(S) @ V
    S = X - L

    error = [np.sqrt(mean_squared_error(X, L + S))]

    return L, S, L + S, error
    
def sdp_naive(X, gamma=1, rank=1, card=None, iterated_power=1, max_iter=100, tol=0.001):
    S = cp.Variable(X.shape)
    L = cp.Variable(X.shape)

    constraints = [L + S == X]

    obj = cp.Minimize(cp.norm(L, "nuc") + gamma * cp.pnorm(S, 1))

    prob = cp.Problem(obj, constraints)

    iter = 1
    RMSE = []

    while True:
        prob.solve(max_iters=iter)

        LS = L.value + S.value

        error = np.sqrt(mean_squared_error(X, LS))
        RMSE.append(error)

        print("iter: ", iter, "error: ", error)
        if (error <= tol) or (iter >= max_iter):
            break
        else:
            iter = iter + 1

    return L.value, S.value, L.value + S.value, RMSE