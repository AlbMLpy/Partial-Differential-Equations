import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicg, gmres, spsolve, bicgstab

import matplotlib.pyplot as plt
from matplotlib import ticker, cm


class SDE:
    """ First order solver """
    def __init__(self, right_side, dirichlet, newman):
        """
            Initialize functions for the right side
            and boundary conditions.
        """
        
        self.f = right_side
        self.g_d = dirichlet
        self.g_n = newman
        
    
    def _newman_cond(
            self, i, j, 
            where_put, num_i, mode):
        
        """
            Used to fill in Newton boundary conditions.
        """
        
        x_i = i * self.h
        y_j = j * self.h
        
        if (i == self.n - 1) and (j == self.n - 1):
            
            if mode == 1:
                self.fg[self.n*j + i] = self.g_n(x_i, y_j) / self.h
                
                self.a[where_put] = -self.C
                self.a[where_put + 1] = -self.A
                self.a[where_put + 2] = self.A + self.C
                
                self.ja[where_put] = num_i - self.n
                self.ja[where_put + 1] = num_i - 1
                self.ja[where_put + 2] = num_i
                
                self.ia[num_i + 1] = self.ia[num_i] + 3
                where_put += 3
                
            if mode == 2:
                self.fg[self.n*j + i] = self.A * self.h * np.pi
            
                self.a[where_put] = self.C
                self.a[where_put + 1] = self.A
                self.a[where_put + 2] = self.B / 2
                
                self.ja[where_put] = num_i - self.n
                self.ja[where_put + 1] = num_i - 1
                self.ja[where_put + 2] = num_i
                
                self.ia[num_i + 1] = self.ia[num_i] + 3
                where_put += 3
            
        else:
            if i == self.n - 1:
                self.fg[self.n*j + i] = self.g_n(x_i, y_j) / self.h
                self.a[where_put] = self.A
                self.a[where_put + 1] = -self.A
                    
                self.ja[where_put] = num_i 
                self.ja[where_put + 1] = num_i - 1
                    
                self.ia[num_i + 1] = self.ia[num_i] + 2
                where_put += 2
                    
            else:
                self.fg[self.n*j + i] = self.g_n(x_i, y_j) / self.h
                self.a[where_put] = self.C
                self.a[where_put + 1] = -self.C
                    
                self.ja[where_put] = num_i 
                self.ja[where_put + 1] = num_i - self.n
                    
                self.ia[num_i + 1] = self.ia[num_i] + 2
                where_put += 2        
        return where_put            
        
    def _nnz(self):
        """
            Count the number of nonzero enrties in a matrix
        """
        
        return 5 * self.n**2 - 14*self.n + 14
    
    def construct_system(self, eps=1, n=32, mode=1):
        """
            Create sparse matrix for discretization, right side
            and true solution
            Input parameters:
                - "eps" - 1 or 10 or 100
                - "n" - size of 1d mesh
                - "mode" - 1 or 2 order in n*n point
        """
        
        self.n = n
        self.eps = eps
        d_x = 1 # coef for C second derivative by x
        d_y = eps # coef for C second derivative by y
        self.h = 1 / (n - 1) # step through mesh
        self.A = -d_x / self.h**2 
        self.C = -d_y / self.h**2
        self.B = 2 * (d_x + d_y) / self.h**2
        nn = n**2 
        nnz = self._nnz()
        
        self.fg = np.empty(nn) # right side
        self.x_true = np.empty(nn) # actual solution
        self.a = np.empty(nnz) # values for sparse matrix
        self.ja = np.empty(nnz) # column indices for sparse matrix
        self.ia = np.empty(nn + 1) # row indices for sparse matrix 
        self.ia[0] = 0
        
        where_put = 0 # to iterate over arrays with values of matrix
        num_i = 0 # to moderate row/equation of a matrix
        for j in range(n):
            y_j = j * self.h
            for i in range(n):
                x_i = i * self.h
                if (i == 0) or (j == 0):
                    # Dirichlet conditions:
                    self.fg[n*j + i] = self.g_d(x_i, y_j) # fill the right side
            
                    self.a[where_put] = 1.0
                    self.ja[where_put] = num_i
                    self.ia[num_i + 1] = self.ia[num_i] + 1
                    where_put += 1
            
                elif (i == n - 1) or (j == n - 1):
                    # Newman conditions:
                    where_put = self._newman_cond(
                        i,
                        j,
                        where_put,
                        num_i,
                        mode=mode
                    )
                else:
                    # Equations from discretization:
                    self.fg[n*j + i] = self.f(x_i, y_j, eps) # fill the right side
            
                    self.a[where_put] = self.C
                    self.a[where_put + 1] = self.A
                    self.a[where_put + 2] = self.B
                    self.a[where_put + 3] = self.A
                    self.a[where_put + 4] = self.C
            
                    self.ja[where_put] = num_i - n
                    self.ja[where_put + 1] = num_i - 1
                    self.ja[where_put + 2] = num_i
                    self.ja[where_put + 3] = num_i + 1
                    self.ja[where_put + 4] = num_i + n
            
                    self.ia[num_i + 1] = self.ia[num_i] + 5
                    where_put += 5
                num_i += 1
                self.x_true[n*j + i] = np.cos(np.pi * y_j) * np.sin(np.pi * x_i)      

    
    def solve(self, tol=1e-08, maxiter=20000):
        """
            Solve discretized system with BiCG.
            Input parameters:
                - "tol" - tolerence for the error
                - "maxiter" - needed to tune iterations properly
        """
        
        nn = self.n**2
        mtx = csr_matrix((self.a, self.ja, self.ia), shape=(nn, nn))
        self.x_appr, info = bicg(mtx, self.fg, x0=np.zeros(nn), tol=tol, maxiter=maxiter)
        if info > 0:
            print("Iterations number: ", info)
            self.iter = info
        return info
    
    def plot_err(self):
        """
            Plot error between our and actual solution
        """
        
        #  Define the number of regions for smoothness:
        lev_region = 10

        z = np.abs(self.x_appr - self.x_true).reshape(self.n, self.n)
        fig, ax = plt.subplots()
        cs = ax.contourf(
            z,
            levels = lev_region,
        )
        cbar = fig.colorbar(cs)
        plt.show()
        
    def c_h_norm(self):
        """ Max norm error of true and approximate solution """
        return np.max(np.abs(self.x_appr - self.x_true))
    
    def l_2h_norm(self):
        """ Max norm error of true and approximate solution """
        sq_err = (self.x_appr - self.x_true)**2
        integ = Integrate(self.n, sq_err)
        x = np.array([i * self.h for i in range(self.n)] * (self.n))
        y = np.sort([x * self.h for x in list(range(self.n)) * (self.n)])
        xyc = np.vstack((x, y, sq_err)).T
        integ.int_all(xyc)
        return np.sqrt(integ.int_value)
    

class SDE2(SDE):
    """ Second order solver """
    def _newman_cond(
            self, i, j, 
            where_put, num_i, mode):
        
        """
            Used to fill in Newton boundary conditions. 
        """
        
        x_i = i * self.h
        y_j = j * self.h
        
        if (i == self.n - 1) and (j == self.n - 1):
            if mode == 1:
                self.fg[self.n*j + i] = self.g_n(x_i, y_j) / self.h
                
                self.a[where_put] = -self.C
                self.a[where_put + 1] = -self.A
                self.a[where_put + 2] = self.A + self.C
                
                self.ja[where_put] = num_i - self.n
                self.ja[where_put + 1] = num_i - 1
                self.ja[where_put + 2] = num_i
                
                self.ia[num_i + 1] = self.ia[num_i] + 3
                where_put += 3
                
            if mode == 2:
                self.fg[self.n*j + i] = (
                    self.A * self.h * np.pi
                    + self.f(x_i, y_j, self.eps) * 0.5
                )
            
                self.a[where_put] = self.C
                self.a[where_put + 1] = self.A
                self.a[where_put + 2] = self.B / 2
                
                self.ja[where_put] = num_i - self.n
                self.ja[where_put + 1] = num_i - 1
                self.ja[where_put + 2] = num_i
                
                self.ia[num_i + 1] = self.ia[num_i] + 3
                where_put += 3
            
        else:
            if i == self.n - 1:
                self.fg[self.n*j + i] = (
                    self.g_n(x_i, y_j) / self.h 
                    - self.f(x_i, y_j, self.eps) / 2
                )
                
                self.a[where_put] = self.A + self.C
                self.a[where_put + 1] = -self.A
                self.a[where_put + 2] = -self.C / 2
                self.a[where_put + 3] = -self.C / 2
                    
                self.ja[where_put] = num_i 
                self.ja[where_put + 1] = num_i - 1
                self.ja[where_put + 2] = num_i - self.n
                self.ja[where_put + 3] = num_i + self.n
                    
                self.ia[num_i + 1] = self.ia[num_i] + 4
                where_put += 4
                    
            else:
                self.fg[self.n*j + i] = (
                    self.g_n(x_i, y_j) / self.h 
                    - self.f(x_i, y_j, self.eps) / 2
                )
                
                self.a[where_put] = self.A + self.C
                self.a[where_put + 1] = -self.C
                self.a[where_put + 2] = -self.A / 2
                self.a[where_put + 3] = -self.A / 2
                    
                self.ja[where_put] = num_i 
                self.ja[where_put + 1] = num_i - self.n
                self.ja[where_put + 2] = num_i - 1
                self.ja[where_put + 3] = num_i + 1
                    
                self.ia[num_i + 1] = self.ia[num_i] + 4
                where_put += 4
                
        return where_put
    
    def _nnz(self):
        return 5 * self.n**2 - 10*self.n + 6
    
    
class Act:
    """ Better interface solver class """
    def __init__(self, f, g_d, g_n):
        """
            Initialize functions for the right side
            and boundary conditions.
        """
        self.f = f
        self.g_d = g_d
        self.g_n = g_n
        
    def act(self,
            solver, eps=1,
            n=128, mode=1, 
            tol=1e-08, maxiter=100000):
        
        """
            Use chosen solver class for SD problem.
            Input parameters:
                - "solver" - SDE/SDE2
                - "eps" - used for d_y setting
                - "n"- number of 1d points
                - "mode" - Newman conditions 
                        edge point 1/2 order approximation
                - "tol" - used for solving SLE
                - "maxiter" - bound on iterations made(need for the task) 
        """
        
        self.sde = solver(self.f, self.g_d, self.g_n)
        self.sde.construct_system(eps=eps, n=n, mode=mode)
        self.sde.solve(tol=tol, maxiter=maxiter)
        
    def act_norms(self,
            solver, eps=1,
            n=128, mode=1, 
            tol=1e-08, maxiter=100000):
        
        """
            Use chosen solver class for SD problem.
            Input parameters:
                - "solver" - SDE/SDE2
                - "eps" - used for d_y setting
                - "n"- number of 1d points
                - "mode" - Newman conditions 
                        edge point 1/2 order approximation
                - "tol" - used for solving SLE
                - "maxiter" - bound on iterations made(need for the task) 
        """
        
        self.sde = solver(self.f, self.g_d, self.g_n)
        self.sde.construct_system(eps=eps, n=n, mode=mode)
        self.sde.solve(tol=tol, maxiter=maxiter)
        print(f"Ch norm = {self.sde.c_h_norm()};\n" 
                + f"L2h norm = {self.sde.l_2h_norm()}")
        
    def act_plot(self,
            solver, eps=1,
            n=128, mode=1, 
            tol=1e-08, maxiter=100000):
        
        """
            Use chosen solver class for SD problem.
            Input parameters:
                - "solver" - SDE/SDE2
                - "eps" - used for d_y setting
                - "n"- number of 1d points
                - "mode" - Newman conditions 
                        edge point 1/2 order approximation
                - "tol" - used for solving SLE
                - "maxiter" - bound on iterations made(need for the task) 
        """
        
        self.sde = solver(self.f, self.g_d, self.g_n)
        self.sde.construct_system(eps=eps, n=n, mode=mode)
        self.sde.solve(tol=tol, maxiter=maxiter)    
        self.sde.plot_err()

class Integrate:
    """ Integration through triangles with bilinear functions """
    def __init__(self, n, vals):
        """
            Capture baricenter coordinates and weights
        """
        
        self.n = n
        self.vals = vals
        self.h = 1 / (n - 1)
        
        # MAGIC NUMBERS!!!:

        w1 = 0.205950504760887
        w2 = 0.063691414286223

        a11 = 0.124949503233232
        a12 = 0.437525248383384
        a13 = a12

        a21 = 0.797112651860071
        a22 = 0.165409927389841
        a23 = 0.037477420750088

        self.w = np.array(3*[w1] + 6*[w2])
        self.a = np.array(
            [
                [a11, a12, a13],
                [a12, a11, a13],
                [a12, a13, a11],
            
                [a21, a22, a23],
                [a21, a23, a22],
                [a22, a21, a23],
                [a22, a23, a21],
                [a23, a22, a21],
                [a23, a21, a22],
            ]
        )
    
    def _phi(self, points_sq):
        """
            Used to produce bilinear approximation 
            on a square.
        """
        
        bones = []
        for i in points_sq:
            bones.append([i[0]*i[1], i[0], i[1], 1])
        A = np.array(bones)
        coefs = np.linalg.solve(A, points_sq[:, 2])
        def hi(x, y):
            return np.array([x*y, x, y, 1]).dot(coefs)   
        return hi

    def _n_xy(self, points_tri):
        """ Give x, y coordinates by baricenters """  
        #xy = np.array([[i[0] for i in points_tri], [i[1] for i in points_tri]])
        return points_tri[:, [0, 1]].T @ self.a.T

    def _int_tri(self, points_tri, elem):
        """ Integrate over triangle """
        xy = self._n_xy(points_tri)
        add_them = 0.0
        for i in range(9):
            add_them = self.w[i] * elem(xy[0, i], xy[1, i])
            
        return add_them * 0.5 * self.h**2
    
    def _int_sq(self, points_sq):
        """ Integrate over square """
        elem = self._phi(points_sq)
        summation = 0.0
        summation += self._int_tri(points_sq[[0,1,2]], elem)
        summation += self._int_tri(points_sq[[1,2,3]], elem)
        return summation
    
    def int_all(self, xyc):
        """
            Integrate mesh.
            Input parameters:
                - "xyc" - numpy.ndarray [x, y, z], 
                    where x, y, z define columns of
                    x, y coordinates and z values in (x, y) points
                    Like: [
                            [0, 0, 1.25],
                            [1, 0, 2.34],
                            [0, 1, 4.3],
                            [1, 1, 5.12]
                          ] - [0, 1]*[0, 1] square.
        """
        
        self.int_value = 0.0
        for j in range(self.n):
            for i in range(self.n):
                if (i <= self.n - 2) and (j <= self.n - 2):
                    self.int_value += self._int_sq(
                        xyc[[i + j*self.n,
                             i + 1 + j*self.n,
                             i + self.n + j*self.n,
                             i + self.n + 1 + j*self.n]]
                    )


def plot(f, g_d, g_n, eps=1, solver=SDE, mode=1, n=[32, 64, 128], to="Report/pictures/new.png"):
    #  Define the number of regions for smoothness:
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'Let d_y = eps = {eps}', fontsize=16)
    actor = Act(f, g_d, g_n)
    
    # First graph
    actor.act(
        solver=solver, eps=eps,
        n=n[0], mode=mode, 
        tol=1e-08,
    )
    z = np.abs(actor.sde.x_appr - actor.sde.x_true).reshape(actor.sde.n, actor.sde.n)
    cs = axs[0, 0].contourf(
            z,
            levels = 20,
    )
    fig.colorbar(cs, ax=axs[0,0])
    axs[0, 0].set_title(f'Error plot(n={n[0]})')

    # Second graph
    actor.act(
        solver=solver, eps=eps,
        n=n[1], mode=mode, 
        tol=1e-08,
    )
    z = np.abs(actor.sde.x_appr - actor.sde.x_true).reshape(actor.sde.n, actor.sde.n)
    cs = axs[0, 1].contourf(
            z,
            levels = 20,
    )
    fig.colorbar(cs, ax=axs[0,1])
    axs[0, 1].set_title(f'Error plot(n={n[1]})')

    # Third graph
    actor.act(
        solver=solver, eps=eps,
        n=n[2], mode=mode, 
        tol=1e-08,
    )
    z = np.abs(actor.sde.x_appr - actor.sde.x_true).reshape(actor.sde.n, actor.sde.n)
    cs = axs[1, 0].contourf(
            z,
            levels = 20,
    )
    fig.colorbar(cs, ax=axs[1,0])
    axs[1, 0].set_title(f'Error plot(n={n[2]})')

    #Forth graph
    solution = actor.sde.x_appr.reshape(actor.sde.n, actor.sde.n)
    cs = axs[1, 1].contourf(
            solution,
            levels = 100,
    )
    fig.colorbar(cs, ax=axs[1,1])
    axs[1, 1].set_title('Approximate solution (n=128)')
    
    fig.savefig(to)                    