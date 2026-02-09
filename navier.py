import jax.numpy as jnp
import jax.random as jr

from jax import jit, lax
from functools import partial
from collections import namedtuple


SimulationConfig = namedtuple("SimulationConfig", ["nx", "ny", "x_limit", "y_limit",
                                                   "mass", "dt", "coupling_strength",
                                                   "sigma", "nu", "save_every",
                                                   "n_steps", "n_particles", "speed"])

SimulationState = namedtuple("SimulationState", ["omega_hat", "x", "v", "key"])

SpectralOperators = namedtuple("SpectralOperators", ["Kx", "Ky", "K2", "mask", "dx", "dy"])


def generate_grid(config):
    x_coords = jnp.linspace(0, config.x_limit, config.nx, endpoint=False)
    y_coords = jnp.linspace(0, config.y_limit, config.ny, endpoint=False)
    X, Y = jnp.meshgrid(x_coords, y_coords, indexing='ij')
    return X, Y


def dealias_mask(nx, ny):
    kx = jnp.abs(jnp.fft.fftfreq(nx))
    ky = jnp.abs(jnp.fft.fftfreq(ny))
    mask = (kx[:, None] < 1/3) & (ky[None, :] < 1/3)
    return mask.astype(jnp.float32)


def build_operators(config: SimulationConfig):
    dx = config.x_limit / config.nx
    dy = config.y_limit / config.ny

    kx = 2*jnp.pi*jnp.fft.fftfreq(config.nx, dx)
    ky = 2*jnp.pi*jnp.fft.fftfreq(config.ny, dy)

    Kx, Ky = jnp.meshgrid(kx, ky, indexing="ij")
    K2 = Kx**2 + Ky**2

    mask = dealias_mask(config.nx, config.ny)

    return SpectralOperators(Kx, Ky, K2, mask, dx, dy)


@partial(jit, static_argnums=(2,))
def bilinear_weights(x, y, config, operator):
    pos_x, pos_y = x / operator.dx, y / operator.dy
    i0, j0 = jnp.floor(pos_x).astype(jnp.int32) % config.nx, jnp.floor(pos_y).astype(jnp.int32) % config.ny
    i1, j1 = (i0 + 1) % config.nx, (j0 + 1) % config.ny
    sx, sy = pos_x - i0, pos_y - j0
    weights = jnp.stack([(1 - sx) * (1 - sy),
                         sx * (1 - sy),
                         (1 - sx) * sy,
                         sx * sy], axis=-1)
    return (i0, j0, i1, j1), weights


@jit
def bilinear_interpolation(indices, weights, u):
    i0, j0, i1, j1 = indices
    v00, v10, v01, v11 = u[:, i0, j0].T, u[:, i1, j0].T, u[:, i0, j1].T, u[:, i1, j1].T
    u_p = (v00 * weights[:, 0:1] + v10 * weights[:, 1:2] +
            v01 * weights[:, 2:3] + v11 * weights[:, 3:4])
    return u_p


@partial(jit, static_argnums=(3,))
def scatter_force(dv, indices, weights, config):
    i0, j0, i1, j1 = indices
    f = jnp.zeros((2, config.nx, config.ny))
    f = f.at[:, i0, j0].add(dv.T * weights[:, 0])
    f = f.at[:, i1, j0].add(dv.T * weights[:, 1])
    f = f.at[:, i0, j1].add(dv.T * weights[:, 2])
    f = f.at[:, i1, j1].add(dv.T * weights[:, 3])
    return f


@jit
def vorticity_to_velocity(omega_hat, operator):
    psi_hat = -omega_hat / jnp.where(operator.K2 > 0, operator.K2, 1.0)
    ux = jnp.fft.ifft2(1j * operator.Ky * psi_hat).real
    uy = jnp.fft.ifft2(-1j * operator.Kx * psi_hat).real
    return jnp.stack([ux, uy])


@jit
def compute_advection(u, omega_hat, operator):
    w_grad_hat = jnp.stack([1j * operator.Kx * omega_hat, 1j * operator.Ky * omega_hat])
    w_grad = jnp.fft.ifft2(w_grad_hat).real
    adv = u[0] * w_grad[0] + u[1] * w_grad[1]
    return jnp.fft.fft2(adv) * operator.mask


@jit
def compute_curl_fourier(f, operator):
    f_hat = jnp.fft.fft2(f)
    return 1j * operator.Kx * f_hat[1] - 1j * operator.Ky * f_hat[0]


@partial(jit, static_argnums=(3,))
def coupling_force(u, x: jnp.ndarray, v:jnp.ndarray, config: SimulationConfig, operator: SpectralOperators):
    indices, weights = bilinear_weights(x[:, 0], x[:, 1], config, operator)
    u_p = bilinear_interpolation(indices, weights, u) 
    dv = v - u_p
    f_coupling = scatter_force(-dv * config.coupling_strength, indices, weights, config)
    return f_coupling, dv


@partial(jit, static_argnums=(3,))
def update_omega(omega_hat: jnp.ndarray, f_coupling: jnp.ndarray, adv_hat: jnp.ndarray, config: SimulationConfig, operator: SpectralOperators):
    f_curl_hat = compute_curl_fourier(f_coupling, operator)
    visc = 0.5 * config.dt * config.nu * operator.K2
    omega_hat_new = (omega_hat * (1 - visc) - config.dt * adv_hat + config.dt * f_curl_hat) / (1 + visc)
    return omega_hat_new


@partial(jit, static_argnums=(3,))
def update_particles(x: jnp.ndarray, v: jnp.ndarray, dv: jnp.ndarray, config: SimulationConfig, key):
    key, subkey = jr.split(key)
    v_new = v - config.dt * dv / config.mass + config.sigma * jnp.sqrt(config.dt) * jr.normal(subkey, v.shape)
    x_new = (x + config.dt * v_new) % jnp.array([config.x_limit, config.y_limit])
    return x_new, v_new, key


@partial(jit, static_argnums=(2,))
def calculate_energy(u: jnp.ndarray, v: jnp.ndarray, config: SimulationConfig, operator: SpectralOperators):
    particle_energy = jnp.sum(v**2)*config.mass/2.0
    field_energy = jnp.sum(u[0]**2 + u[1]**2) * (operator.dx*operator.dy)/2.0
    return particle_energy + field_energy


@partial(jit, static_argnums=(1,))
def calculate_energy_spectral(state: SimulationState, config: SimulationConfig, operator: SpectralOperators):
    particle_energy = 0.5 * config.mass * jnp.sum(state.v**2)
    mag_sq = jnp.abs(state.omega_hat)**2
    inv_k2 = jnp.where(operator.K2 > 0, 1.0 / operator.K2, 0.0)
    normalization = (config.nx * config.ny)
    field_energy = 0.5 * jnp.sum(mag_sq * inv_k2) / normalization
    return particle_energy + field_energy


@partial(jit, static_argnums=(1,))
def step(state: SimulationState, config: SimulationConfig, operator: SpectralOperators):
    omega_hat, x, v, key = state
    
    u = vorticity_to_velocity(omega_hat, operator)
    f_coupling, dv = coupling_force(u, x, v, config, operator)
    adv_hat = compute_advection(u, omega_hat, operator)

    omega_hat_new = update_omega(omega_hat, f_coupling, adv_hat, config, operator)
    x_new, v_new, key_new = update_particles(x, v, dv, config, key)

    #kinetic_energy = calculate_energy_spectral(state, config, operator)
    kinetic_energy = calculate_energy(u, v, config, operator)
    new_state = SimulationState(omega_hat_new, x_new, v_new, key_new)

    return new_state, kinetic_energy


def simulate(w0: jnp.ndarray, x0: jnp.ndarray, v0: jnp.ndarray, config: SimulationConfig, key):
    operators = build_operators(config)
    omega_hat0 = jnp.fft.fft2(w0)
    state0 = SimulationState(omega_hat0, x0, v0, key)
    
    num_blocks = config.n_steps // config.save_every
    
    def outer_step(carry, _):
        def inner_step(c, _):
            return step(c, config, operators)
        
        final_carry, ke = lax.scan(inner_step, carry, None, length=config.save_every)
        w = jnp.fft.ifft2(final_carry.omega_hat).real
        u = vorticity_to_velocity(final_carry.omega_hat, operators)
        snapshot = (final_carry.x, u, ke, w)
        
        return final_carry, snapshot

    _, (x_hist, u_hist, ke_history_blocks, w_cache) = lax.scan(outer_step, state0, jnp.arange(num_blocks))
    
    return x_hist, u_hist, ke_history_blocks.flatten(), w_cache
