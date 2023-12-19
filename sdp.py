import cvxpy as cp
from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import randomized_svd
import numpy as np
from godec import godec

# def svd_cardinality(X, rank=1, card=None):
#     card = np.prod(X.shape) if card is None else card

#     U, S, V = randomized_svd(X, rank)
#     L = U @ np.diag(S) @ V
#     S = X - L
#     flat_S = np.abs(S).flatten()
#     largest_card_indices = np.argsort(flat_S)[-card:]
#     S_prime = np.zeros_like(S).flatten()
#     S_prime[largest_card_indices] = S.flatten()[largest_card_indices]
#     S = S_prime.reshape(S.shape)

#     error = [np.sqrt(mean_squared_error(X, L + S))]

#     return L, S, L + S, error

def svd_cardinality(*args, **kwargs):
    return godec.godec(*args, **kwargs, max_iter=1)

    
def sdp_cardinality(X, rank=1, card=None, iterated_power=1, max_iter=100, tol=0.001):
    S = cp.Variable(X.shape)
    L = cp.Variable(X.shape)

    # constraints = [L + S == X]

    # obj = cp.Minimize(cp.norm(L, "nuc") + gamma * cp.pnorm(S, 1))
    obj = cp.Minimize(cp.norm(X - L - S, "fro"))

    constraints = [cp.pnorm(S, 1) <= card, cp.norm(L, "nuc") <= rank]

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