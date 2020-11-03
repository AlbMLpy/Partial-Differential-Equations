import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicg, gmres, spsolve, bicgstab

import matplotlib.pyplot as plt
from matplotlib import ticker, cm

import SDE as sde



def f(x, y, eps):
    """ Right side """
    return np.pi**2 * np.cos(np.pi * y) * np.sin(np.pi * x) * (1 + eps)

def g_d(x, y):
    """ Dirichlet condition function """
    if y == 0:
        return np.sin(np.pi * x)
    if x == 0:
        return 0.0
    assert False # catch problems
    
def g_n(x, y):
    """ Newman condition function """
    if x == 1:
        return np.pi * np.cos(np.pi * y)
    if y == 1:
        return 0.0
    assert False # catch problems
    
    
    
sde.plot(f, g_d, g_n, eps=1, solver=sde.SDE2, mode=2, n=[32, 64, 128], to="Report/pictures/plot(1).png")
sde.plot(f, g_d, g_n, eps=10, solver=sde.SDE2, mode=2, n=[32, 64, 128], to="Report/pictures/plot(10).png")
sde.plot(f, g_d, g_n, eps=100, solver=sde.SDE2, mode=2, n=[32, 64, 128], to="Report/pictures/plot(100).png")
sde.plot(f, g_d, g_n, eps=100, solver=sde.SDE, mode=2, n=[32, 64, 128], to="Report/pictures/plot(100_1).png")

actor = sde.Act(f, g_d, g_n)

actor.act_norms(
    sde.SDE2, eps=1,
    n=32, mode=2, 
    tol=1e-08, maxiter=532
)

actor.act_norms(
    sde.SDE2, eps=1,
    n=64, mode=2, 
    tol=1e-08, maxiter=1325
)

actor.act_norms(
    sde.SDE2, eps=1,
    n=128, mode=2, 
    tol=1e-08, maxiter=3020
)

actor.act_norms(
    sde.SDE2, eps=10,
    n=32, mode=2, 
    tol=1e-08, maxiter=846
)

actor.act_norms(
    sde.SDE2, eps=10,
    n=64, mode=2, 
    tol=1e-08, maxiter=2610
)

actor.act_norms(
    sde.SDE2, eps=10,
    n=128, mode=2, 
    tol=1e-08, maxiter=8100
)

actor.act_norms(
    sde.SDE2, eps=100,
    n=32, mode=2, 
    tol=1e-08, maxiter=1470
)

actor.act_norms(
    sde.SDE2, eps=100,
    n=64, mode=2, 
    tol=1e-08, maxiter=5391
)

actor.act_norms(
    sde.SDE2, eps=100,
    n=128, mode=2, 
    tol=1e-08, maxiter=21200
)

actor.act_norms(
    sde.SDE, eps=100,
    n=32, mode=2, 
    tol=1e-08, maxiter= 819
)

actor.act_norms(
    sde.SDE, eps=100,
    n=64, mode=2, 
    tol=1e-08, maxiter=2090
)

actor.act_norms(
    sde.SDE, eps=100,
    n=128, mode=2, 
    tol=1e-08, maxiter=6505
)