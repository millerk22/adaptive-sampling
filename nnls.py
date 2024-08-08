import numpy as np
import scipy.sparse as sp


"""
This is code implemented to mimic the nnls_FPGM method used by Gillis in the nmfbook MATLAB repository.
"""

def projection_cvxhull(Y):
    A = np.maximum(Y, 0)
    mask = A.sum(axis=0) > 1.0
    A[:,mask] = projection_simplex(Y[:,mask], z=1, axis=0)
    return A
    
def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])

    Code taken from https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()

def orth_nnls(M, U, Mn=None):
    if Mn is None:
        norm2m = np.linalg.norm(M, axis=0)
        Mn = M /(norm2m + 1e-16)
    m, n = Mn.shape
    m, r = U.shape
    
    # Normalize columns of U
    norm2u = np.linalg.norm(U, axis=0)
    Un = U /(norm2u + 1e-16)
    
    A = Mn.T @ Un  # n by r matrix of angles between the columns of U and M
    b = np.argmax(A, axis=1)  # best column of U to approximate each column of M
    
    # Calculate the optimal weights using vectorized operations
    V = np.zeros((r, n))
    V[b, np.arange(n)] = (M * U[:,b]).sum(axis=0) / (norm2u[b] ** 2)
    return V


def nnls_init(X, W, WtW, WtX):
    if np.linalg.cond(W) > 1e6:
        # Assign each column of X to the closest column of W
        # in terms of angle: this is the optimal solution with
        # V having a single non-zero per column.
        H = orth_nnls(X, W)
    else:
        # Projected LS solution + scaling
        if sp.issparse(X):
            H = np.maximum(0, np.linalg.pinv(W) @ X)
        else:
            H = np.maximum(0, np.linalg.lstsq(W, X, rcond=None)[0])
        
        # Scale
        alpha = np.sum(H * WtX) / np.sum(WtW * (H @ H.T))
        H *= alpha
    
    # Check that no rows of H is zeros
    # If it is the case, set them to small random numbers
    zerow = np.where(np.sum(H, axis=1) == 0)[0]
    if len(zerow) > 0:
        H[zerow, :] = 0.001 * np.max(H) * np.random.rand(len(zerow), H.shape[1])
    
    return H

def nnls_FPGM(X, W, delta=1e-6, inneriter=500, alpha0=0.05, H0=None, returnH=True, verbose=True):
    W = np.array(W) if sp.issparse(W) else np.asarray(W)
    m, n = X.shape
    m, r = W.shape
    WtW = W.T @ W
    WtX = W.T @ X

    normX2 = np.linalg.norm(X, axis=0)**2.
    
    if H0 is None:
        H = nnls_init(X, W, WtW, WtX)
    else:
        H = H0
    
    L = np.linalg.norm(WtW, 2)
    WtX = W.T @ X
    alpha = alpha0

    H = projection_cvxhull(H)  
    Y = H
    i = 1
    eps0 = 0.0
    eps = 1.0
    
    while i <= inneriter and eps >= delta * eps0:
        Hp = H.copy()
        alpha_i = (np.sqrt(alpha**4 + 4. * alpha**2) - alpha**2) / 2.
        beta = alpha * (1.0 - alpha) / (alpha**2 + alpha_i)
        alpha = alpha_i
        
        H = Y - (WtW @ Y - WtX) / L
        
        H = projection_cvxhull(H)
        
        Y = H + beta * (H - Hp)
        
        if i == 1:
            eps0 = np.linalg.norm(H - Hp, 'fro')
        eps = np.linalg.norm(H - Hp, 'fro')
        i += 1
    
    energy_vals = normX2 - 2*(WtX * H).sum(axis=0) + (H*(WtW @ H)).sum(axis=0)
    if verbose:
        # print(i, inneriter, L)
        return H, {"energy_vals":energy_vals, "iters":i, "eps":eps }
    if returnH:
        return H, energy_vals
    return None, energy_vals




def nnls_OGM(X, W, delta=1e-3, maxiter=1000, lam=1.0, H0=None, returnH=True, verbose=False):
    W = np.array(W) if sp.issparse(W) else np.asarray(W)
    m, n = X.shape
    m, r = W.shape
    WtW = W.T @ W
    WtX = W.T @ X
    normX2 = np.linalg.norm(X, axis=0)**2.
    
    if H0 is None:
        H = nnls_init(X, W, WtW, WtX)
    else:
        H = H0
    
    L = np.linalg.norm(WtW, 2)

    H = projection_cvxhull(H)  
    Y = H
    Hinit = H.copy()
    i = 0
    eps0 = 0.0
    eps = 1.0

    #other = {}
    while i <= maxiter and eps >= delta * eps0:
        Hp = H.copy()
        lam_ = 0.5*(1. + np.sqrt(1. + 4.*lam**2.))
        beta = (lam - 1.0)/lam_ 
        H = projection_cvxhull(Y - (WtW @ Y - WtX) / L)
        Y = H + beta * (H - Hp)
        
        if i == 0:
            eps0 = np.linalg.norm(H - Hp, 'fro')
        eps = np.linalg.norm(H - Hp, 'fro')
        i += 1
        lam = lam_

    
    energy_vals = normX2 - 2*(WtX * H).sum(axis=0) + (H*(WtW @ H)).sum(axis=0)
    if verbose:
        return H, {"energy_vals":energy_vals, "iters":i, "eps":eps }

    if returnH:
        return H, energy_vals
    return None, energy_vals



def nnls_OGM_gram(G_S, S_ind, G_diag, delta=1e-3, maxiter=500, lam=1.0, returnH=True, verbose=False, term_cond=1, X=None, hull=True):
    """
    G_S = |S_ind| x n  numpy array Gram submatrix
    """
    if term_cond == 1:
        assert X is not None 


    G_SS = G_S[:,S_ind]
    L = np.linalg.norm(G_SS, 2)
    if hull:
        H = projection_cvxhull(np.linalg.pinv(G_SS)@ G_S)
    else:
        H = np.maximum(0.0, np.linalg.pinv(G_SS)@ G_S)
    Z = H.copy()

    i = 0
    continue_flag = True
    if verbose:
        EPS = []
    while i <= maxiter and continue_flag:
        Hp = H.copy()
        if hull:
            H = projection_cvxhull(Z - (G_SS @ Z - G_S)/L)
        else:
            H = np.maximum(0.0, Z - (G_SS @ Z - G_S)/L)
        lam_ = 0.5*(1. + np.sqrt(1. + 4.*lam**2.))  
        beta = (lam - 1.0)/lam_ 
        Z = H + beta*(H - Hp)

        i += 1
        lam = lam_
        

        if term_cond == 0:
            if i == 1:
                eps0 = np.linalg.norm(H - Hp, 'fro')
                eps = eps0
            else:
                eps = np.linalg.norm(H-Hp, 'fro')
            continue_flag = eps >= delta*eps0
        elif term_cond == 1:
            gradient_proj = G_SS @ H - G_S
            mask = H <= 1e-6
            gradient_proj[mask] = np.minimum(0.0, gradient_proj[mask])
            if hull:
                mask1 = H >= (1.0 -1e-6)
                gradient_proj[mask1] = np.maximum(1.0, gradient_proj[mask1])
            eps = np.linalg.norm(gradient_proj, ord='fro')
            if i == 1:
                Mat = H@(H.T @ X[:,S_ind].T - X.T) # don't need to mask out,because we know that W = X[:,S_ind] is non-negative
                eps0 = np.sqrt(eps**2. + np.linalg.norm(Mat, ord='fro')**2.) 
            else:
                
                if eps < delta*eps0:
                    if i <= 10:
                        delta *= 0.1
                    else:
                        continue_flag = False 
        else:
            raise ValueError(f"term_cond = {term_cond} not recognized...")
        
        if verbose:
            EPS.append(eps)
         
            
    energy_vals = G_diag - 2.*(G_S * H).sum(axis=0) + ((G_SS @ H) * H).sum(axis=0)
    energy_vals[energy_vals < 0] = 0.0
    if verbose:
        return H, {"energy_vals":energy_vals, "iters":i, "eps":eps, "eps0":eps0, "EPS":EPS }

    if returnH:
        return H, energy_vals
    return None, energy_vals




def nnls_OGM_gram_sub(G_S, S_ind, G_diag, delta=1e-3, maxiter=500, lam=1.0, returnH=True, verbose=False):
    """
    G_S = |S_ind| x n  numpy array Gram submatrix
    """
    G_SS = G_S[:,S_ind]
    L = np.linalg.norm(G_SS, 2)
    H = projection_cvxhull(np.linalg.pinv(G_SS)@ G_S)
    Z = H.copy()

    i = 0
    eps0 = 0.0
    eps = 1.0
    active_set = np.ones(G_diag.size, dtype=bool)

    while i <= maxiter and eps >= delta*eps0*active_set.sum()/active_set.size:
        Hp = H.copy()
        H[:,active_set] = projection_cvxhull(Z[:,active_set] - (G_SS @ Z[:,active_set] - G_S[:,active_set])/L)
        lam_ = 0.5*(1. + np.sqrt(1. + 4.*lam**2.))  
        beta = (lam - 1.0)/lam_ 
        Z[:,active_set] = H[:,active_set] + beta*(H[:,active_set] - Hp[:,active_set])


        if i == 0:
            eps0 = np.linalg.norm(H - Hp, 'fro')
            eps = eps0
        else:
            eps_ = np.zeros_like(active_set)
            eps_[active_set] = np.linalg.norm((H-Hp)[:,active_set], axis=0)
            active_set = eps_ > delta*eps0/active_set.size # restrict to only stepping on those points that are contributing more than what we'd expect
            eps = eps_[active_set].sum()


        i += 1
        lam = lam_
    energy_vals = G_diag - 2.*(G_S * H).sum(axis=0) + ((G_SS @ H) * H).sum(axis=0)
    energy_vals[energy_vals < 0] = 0.0
    if verbose:
        return H, {"energy_vals":energy_vals, "iters":i, "eps":eps }

    if returnH:
        return H, energy_vals
    return None, energy_vals