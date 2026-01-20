import numpy as np
from abc import ABC, abstractmethod
from scipy.ndimage import gaussian_filter


class Force(ABC):
    def __init__(self, simulator):
        self.simulator = simulator

    @abstractmethod
    def __call__(self, **args):
        pass


class Simulator:
    """
    Class that store all important information necessary to the simulation, including the grid and step parameters.
    """
    def __init__(
            self, 
            x_limit: float, 
            y_limit: float, 
            grid_size: int,
            step_size: float,
            n_steps: int,
        ):
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.n_steps = n_steps
        self.step_size = step_size

        self.x, self.y = np.linspace(0, x_limit, grid_size, endpoint=False), np.linspace(0, y_limit, grid_size, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij') # 
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]

        p = 2*np.pi * np.fft.fftfreq(grid_size, d=self.dx)  #
        q = 2*np.pi * np.fft.fftfreq(grid_size, d=self.dy)  #

        self.P, self.Q = np.meshgrid(p, q, indexing='ij')
        self.K2 = self.P**2 + self.Q**2

    def bilinear_interpolator(self, x, field):
        """
        Calculates the field at any position x by bilinear interpolation
        """
        dx, dy = self.dx, self.dy
        posx, posy = x[:, 0]/dx, x[:, 1]/dy

        # Calculates 4 closest points
        i0, j0 = np.floor(posx).astype(int), np.floor(posy).astype(int)
        i1, j1 = (i0 + 1) % len(self.x), (j0 + 1) % len(self.y)

        sx = (posx - np.floor(posx))[:, np.newaxis]
        sy = (posy - np.floor(posy))[:, np.newaxis]

        u_grid = field.transpose(1, 2, 0)

        w00, w10, w01, w11 = (1-sx)*(1-sy), sx*(1-sy), (1-sx)*sy, sx*sy

        u_p = (u_grid[i0, j0] * w00 + u_grid[i1, j0] * w10 + 
               u_grid[i0, j1] * w01 + u_grid[i1, j1] * w11)        

        return u_p, (i0, j0, i1, j1), (w00, w10, w01, w11)


class SpectralNavierStokes:
    def __init__(
            self, 
            simulator: Simulator,
            external_force: Force,
            initial_vorticity: np.ndarray,
            nu: float,
        ):
        self.simulator = simulator
        self.dx = simulator.dx
        self.dy = simulator.dy
        self.step_size = simulator.step_size
        self.n_steps = simulator.n_steps

        self.initial_vorticity = initial_vorticity
        self.nu = nu
        self.external_force = external_force

        self.P = simulator.P
        self.Q = simulator.Q
        self.K2 = simulator.K2 
        self.iP = 1j * self.P
        self.iQ = 1j * self.Q

        cut = int(self.P.shape[0]*2/3)
        self.dealias_mask = np.ones_like(self.K2, dtype=bool)
        self.dealias_mask[cut:-cut, :] = False
        self.dealias_mask[:, cut:-cut] = False

    def fourier_curl(self, vector_field):
        """
        Calculates the Fourier curl of a vector field.

        args:
            - vector_field: 2D grid evaluations of the vector field.

        returns:
            - curl_field: 2D grid evaluation of the curl.
        """
        fx_hat = np.fft.fft2(vector_field[0])
        fy_hat = np.fft.fft2(vector_field[1])
        force_curl_hat = self.iP*fy_hat - self.iQ*fx_hat

        return force_curl_hat

    def vorticity_to_velocity(self, w_hat):
        """
        Transform the Fourier vorticity function at a time t to the Fourier velocity through Poisson's Equation. 
        (k1^2 + k2^2)ψ^ = ω^

        args:
            - w_hat: 2D grid Fourier evaluations of the vorticity.

        returns:
            - ux, uy: 2D grid of the two components of the velocity in normal variables.
        """
        psi_hat = np.zeros_like(w_hat)
        psi_hat[self.K2 != 0] = w_hat[self.K2 != 0] / self.K2[self.K2 != 0]
        psi_hat[0,0] = 0.0        
        ux_hat = self.iQ*psi_hat
        uy_hat = -self.iP*psi_hat

        ux = np.fft.ifft2(ux_hat).real
        uy = np.fft.ifft2(uy_hat).real

        return np.array([ux, uy])

    def nonlinear(self, ux, uy, w_hat):
        """
        Calculate the Fourier non-linear term from Navier-Stokes.
        """
        wx = np.fft.ifft2(self.iP*w_hat).real
        wy = np.fft.ifft2(self.iQ*w_hat).real

        nonlinear = np.fft.fft2(ux*wx + uy*wy)
        return nonlinear


    def step(self, w_hat, external_force):
        """
        Does a single step of the Navier simulation
        """
        dt = self.step_size

        fourier_force = self.fourier_curl(external_force)

        ux, uy = self.vorticity_to_velocity(w_hat)

        nonlinear = self.nonlinear(ux, uy, w_hat)
        nonlinear *= self.dealias_mask

        # Crank-Nicolson step
        visc_term = 0.5 * dt * self.nu * self.K2
        w_hat = (w_hat * (1 - visc_term) - dt * nonlinear + dt * fourier_force) / (1 + visc_term) 

        return w_hat, ux, uy

    def solve(self):
        """
        Calculates the velocity at each grid point for all times between t0=0 and t0=step_size*n_steps
        
        returns:
            - u_cache: all the values of u for all t 
            - w_cache: all the values of 
        """
        u_cache, w_cache = [], []
        w = self.initial_vorticity
        w_hat = np.fft.fft2(w)

        for step in range(self.n_steps):
            t = step*self.step_size
            w_hat, ux, uy = self.step(w_hat, self.external_force(t))

            u_cache.append([ux.copy(), uy.copy()])
            w_cache.append(np.fft.ifft2(w_hat).real.copy())

        return np.array(u_cache), np.array(w_cache)


class Particle:
    def __init__(
            self,
            simulator: Simulator,
            n_particles: int,
            x0: np.ndarray, 
            v0: np.ndarray, 
            field: np.ndarray,
            mass: float,
            sigma=0.001,
        ):
        self.simulator = simulator
        self.step_size = simulator.step_size
        self.n_steps = simulator.n_steps
        self.limits = np.array([simulator.x_limit, simulator.y_limit])

        self.n_particles = n_particles
        self.x0 = x0
        self.v0 = v0
        self.field = field
        self.mass = mass
        self.sigma = sigma

    def step(self, x, v, u):
        """
        Does a single simulation step for a single particle.

        args:
            - x : position of the particles 
            - v : velocity of the particles

        """
        dt = self.step_size
        x = (x + dt*v) % self.limits
        g = np.random.randn(*v.shape)
        u, _, _ = self.simulator.bilinear_interpolator(x, u)
        v = v + dt*(u - v)/self.mass + np.sqrt(dt)*self.sigma*g

        return x, v

    def move(self):
        """
        """
        x, v = self.x0, self.v0
        path = np.zeros((self.n_steps, self.n_particles, 2))

        for step in range(self.n_steps):
            x, v = self.step(x, v, self.field[step])
            path[step] = x.copy()

        return path


class CoupledNavier:
    def __init__(
            self,
            navier: SpectralNavierStokes,
            particle: Particle,
        ):
        self.navier = navier
        self.n_steps = navier.n_steps

        self.particle = particle
        self.coupled_term = CoupledForce(navier.simulator)

        self.step_size = navier.step_size
        self.initial_vorticity = navier.initial_vorticity

        self.x0 = self.particle.x0
        self.v0 = self.particle.v0

    def step(self, u, w_hat, xi, vi, coupled_term):
        w_hat, ux, uy = self.navier.step(w_hat, coupled_term)
        u = np.array([ux, uy])

        xi, vi = self.particle.step(xi, vi, u)

        return w_hat, u, xi, vi

    def solve(self):
        """
        """
        u_cache, w_cache, path, force_cache = [], [], [], []
        w = self.navier.initial_vorticity
        w_hat = np.fft.fft2(w)
        u = self.navier.vorticity_to_velocity(w_hat)

        xi, vi = self.x0, self.v0

        for step in range(self.n_steps):
            coupled_term = self.coupled_term(u, xi, vi)
            w_hat, u, xi, vi = self.step(u, w_hat, xi, vi, coupled_term)

            u_cache.append(u.copy())
            w_cache.append(np.fft.ifft2(w_hat).real.copy())
            path.append(xi.copy())
            force_cache.append(coupled_term.copy())

        return np.array(u_cache), np.array(w_cache), np.array(path), np.array(force_cache)


class ZeroForce(Force):
    def __call__(self):
        return np.array([0.0*self.simulator.X, 0.0*self.simulator.Y])


class CoupledForce(Force):
    def __call__(self, u, xi, vi):
        """
        Calculates the coupling term from Navier-Stokes.

        args:
            - u: 2D grid representing vector field
            - xi: array of particle positions
            - vi: array of particle positions
        """
        Np = len(xi)
        Nx, Ny = self.simulator.X.shape
        Lx, Ly = self.simulator.x_limit, self.simulator.y_limit

        u_p, _, _ = self.simulator.bilinear_interpolator(xi, u)
        alpha = 1.0
        dv = vi - u_p  # shape (Np,2)
        dv = alpha * dv

        xi_x = xi[:,0][:, None, None]  # (Np,1,1)
        xi_y = xi[:,1][:, None, None]

        dx = self.simulator.X[None,:,:] - xi_x  # (Np, Nx, Ny)
        dy = self.simulator.Y[None,:,:] - xi_y

        dx = (dx + Lx/2) % Lx - Lx/2
        dy = (dy + Ly/2) % Ly - Ly/2

        epsilon = 3*self.simulator.dx
        kernel = np.exp(-(dx**2 + dy**2)/(2*epsilon**2))
        kernel /= 2*np.pi*epsilon**2

        fx = np.sum(dv[:,0][:,None,None] * kernel, axis=0)
        fy = np.sum(dv[:,1][:,None,None] * kernel, axis=0)

        return np.array([fx, fy])





