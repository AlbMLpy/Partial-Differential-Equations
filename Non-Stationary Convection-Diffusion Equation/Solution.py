import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicg, gmres

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import NCDE as ncde

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=100,
    n=200, 
    dx=1.0,
    supg=False, 
    save=True, 
    video=False
)

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=100,
    n=400, 
    dx=1.0,
    supg=False, 
    save=True, 
    video=False
)

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=100,
    n=800, 
    dx=1.0,
    supg=False, 
    save=True, 
    video=False
)

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=100,
    n=200, 
    dx=0.0001,
    supg=False, 
    save=True, 
    video=False
)

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=100,
    n=200, 
    dx=0.0001,
    supg=True, 
    save=True, 
    video=False
)

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=100,
    n=400, 
    dx=0.0001,
    supg=False, 
    save=True, 
    video=False
)

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=100,
    n=400, 
    dx=0.0001,
    supg=True, 
    save=True, 
    video=False
)

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=100,
    n=800, 
    dx=0.0001,
    supg=False, 
    save=True, 
    video=False
)

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=100,
    n=800, 
    dx=0.0001,
    supg=True, 
    save=True, 
    video=False
)

T = 100

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=T,
    n=400, 
    dx=0.0001,
    supg=False, 
    save=True, 
    video=False
)

T = 200

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=T,
    n=400, 
    dx=0.0001,
    supg=False, 
    save=True, 
    video=False
)

T = 400

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=T,
    n=400, 
    dx=0.0001,
    supg=False, 
    save=True, 
    video=False
)

T = 800

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=T,
    n=400, 
    dx=0.0001,
    supg=False, 
    save=True, 
    video=False
)

T = 1600

actor = ncde.Act()
actor.act_plot(
    solver=ncde.NCDE,
    T=T,
    n=400, 
    dx=0.0001,
    supg=False, 
    save=True, 
    video=False
)

