import torch
from torch.linalg import solve
import time
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
from tqdm import tqdm
# from torcheval.metrics.functional import r2_score

def euclidean_distances_M_2(samples, samples_M_applied, centers, centers_M_applied, threshold=1e-3):
    samples_norm = samples_M_applied * samples
    samples_norm = torch.sum(samples_norm, dim=1)

    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = centers_M_applied * centers
        centers_norm = torch.sum(centers_norm, dim=1)

    samples_norm = torch.reshape(samples_norm, (-1, 1))
    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples_M_applied @ centers.T
  
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)

    if samples is centers:
        # print('is centers')
        distances.fill_diagonal_(0)
    else:
        distances = torch.where(abs(distances) < threshold, 0, distances)

    distances.clamp_(min=0)
    distances.sqrt_()

    return distances

def laplacian_M_2(bandwidth, distances):
    assert bandwidth > 0
    kernel_mat = distances
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def laplacian_M_3(X, X_M_applied, x, x_M_applied, L):
    dist = euclidean_distances_M_2(X, X_M_applied, x, x_M_applied)


    K = laplacian_M_2(L, dist.clone())
    K = K/dist

    K[K == float("Inf")] = 0.

    return K


def laplacian_M(X, X_M_applied, x, x_M_applied, L):
    dist = euclidean_distances_M_2(X, X_M_applied, x, x_M_applied)
    K = laplacian_M_2(L, dist.clone())

    K[K == float("Inf")] = 0.

    return K


def get_grads_2(X, x, sol, L, P):

    X_M_applied = X @ P
    x_M_applied = X_M_applied if x is X else x @ P

    K = laplacian_M_3(X, X_M_applied, x, x_M_applied, L)

    n, d = X.shape
    m, d = x.shape

    # step_1
    X1 = X_M_applied.reshape(n,d)
    sol_X1 = sol.reshape(n,1).expand(n,d) * X1 # (n, d)

    step_1 = K.T @ sol_X1 # (m, d)

    # step_2
    sol_K = sol @ K # (m)
    del K

    sol_K = sol_K.reshape(m,1).expand(m,d)
    x1 = x_M_applied.reshape(m,d)
    step_2 = sol_K * x1 # (m, d)
    
    G = (step_1 - step_2)
    G = G - torch.mean(G, axis=0, keepdims=True)

    M = torch.mm(G.T, G)  # (d, d)
    M = M / (m * L * L)

    return M #/ M.max()

def solve_kr(X, X_M_applied, y, L, reg):
    K = laplacian_M(X, X_M_applied, X, X_M_applied, L)    

    try: 
        sol = solve(K + reg * torch.eye(len(K), device=X.device), y).T    
    except torch._C._LinAlgError:
        return None
    return sol

def get_err(sol, X, x, X_M_applied, x_M_applied, y, L):
    K_test = laplacian_M(X, X_M_applied, x, x_M_applied, L)    
    preds = (sol @ K_test).T
    return torch.mean(torch.square(preds - y)), r2_score(preds, y)


def get_top_dir_err(X, y, M):
    epsilon = 1e-6  # Small regularization factor
    max_attempts = 3  # Number of times to try increasing regularization

    # start = time.time()
    for attempt in range(max_attempts):
        try:
            # S, U = torch.linalg.eigh(concept_features)
            s, u = torch.lobpcg(M, k=1)
            break  # If successful, exit the loop
        except torch._C._LinAlgError:
            epsilon *= 10  # Increase regularization
            print(f"Warning: Matrix ill-conditioned. Retrying with epsilon={epsilon}")
            concept_features += epsilon * torch.eye(M.shape[0], device=M.device)
    else:
        raise RuntimeError("linalg.eigh failed to converge even with regularization.")

    # s, u = torch.lobpcg(M, k=1)
    preds = X @ u

    # print("preds, y: ", preds, y)
    # print(preds.shape, y.shape)
    return torch.abs(torch.corrcoef(torch.cat((preds, y), dim=-1).T))[0, 1].item(), u

def rfm(traindata, testdata, L=10, reg=1e-3, num_iters=10, norm=False):
    X_train, y_train = traindata
    X_test, y_test = testdata

    mean = torch.mean(X_train, dim=0, keepdims=True)
    X_train = (X_train - mean)
    X_test = (X_test - mean)

    if norm:
        X_train = X_train / torch.norm(X_train, dim=-1).reshape(-1, 1)
        X_test = X_test / torch.norm(X_test, dim=-1).reshape(-1, 1)

    n, d = X_train.shape
    M = torch.eye(d, device=X_train.device)

    # best_err = float('inf')
    # best_r2 = -float('inf')
    best_r = -float('inf')
    best_M = None
    best_u = None
    for i in range(num_iters):
        X_train_M_applied = X_train @ M
        # X_test_M_applied = X_test @ M
        sol = solve_kr(X_train, X_train_M_applied, y_train, L, reg)
        # print("sol: ", sol)
        if sol is None: 
            break
        
        test_r, u = get_top_dir_err(X_test, y_test, M)
        # print("u: ", u)
        # print("x test: ", X_test)
        # print("RFM Test: ", test_r)
        # return 

        if test_r > best_r: 
            best_M = M.clone()
            best_r = test_r
            best_u = u.clone()

        #test_err, test_r2 = get_err(sol, X_train, X_test, X_train_M_applied, X_test_M_applied, y_test, L)
        # if i > 0 and best_err >= test_err:
        #     best_M = M.clone()
        #     best_err = test_err
        #     best_r2 = test_r2

        M = get_grads_2(X_train, X_train, sol, L, M)
        M /= M.max()

        # if i == 0: 
        #     best_M = M.clone()
        #     best_err = test_err 
        #     best_r2 = test_r2
        # print("Best MSE / Test MSE: ", best_err.item(), test_err.item(), "Best R2 / Test R2: ", best_r2.item(), test_r2.item())
    # return best_M, best_r #best_err.item(), best_r2.item()
    # return M, test_err, test_r2
    
    return best_u, best_r, best_M

def main():

    # create low rank data
    n = 1000
    d = 100
    torch.manual_seed(0)
    X_train = torch.randn(n,d).cuda()
    X_test = torch.randn(n,d).cuda()

    # y_train = torch.where(X_train[:, 0] > 0, 1., 0).reshape(-1, 1)
    # y_test = torch.where(X_test[:, 0] > 0, 1., 0).reshape(-1, 1)

    y_train = ((X_train[:, 0] + X_train[:, 1])).reshape(-1, 1)
    y_test = ((X_test[:, 0] + X_test[:, 1])).reshape(-1, 1)

    print(X_train.shape, y_train.shape)

    start = time.time()
    best_u, best_r = rfm((X_train, y_train), 
                 (X_test, y_test),
                 reg=1e-3,
                 L=10,
                 num_iters=10)
    print(best_u[:3], best_r)


    # best_M, best_err, best_r2 = rfm((X_train, y_train), 
    #              (X_test, y_test),
    #              reg=1e-3,
    #              L=10,
    #              num_iters=10)
    # print(best_M[:3, :3], best_err, best_r2)
    end = time.time()
    print("Training time: ", end - start)    

if __name__ == "__main__":
    main()