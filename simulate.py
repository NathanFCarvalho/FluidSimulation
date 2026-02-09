import jax
import yaml
import time

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from navier import generate_grid, simulate, SimulationConfig

if __name__ == "__main__":
    with open("config.yml", "r") as file:
        params = yaml.safe_load(file)

key = jr.PRNGKey(42)
k1, k2, k3 = jr.split(key, 3)

config = SimulationConfig(**params)
X, Y = generate_grid(config)

w0 = -2 * jnp.sin(X) * jnp.sin(Y)
x0 = jr.uniform(k1, (config.n_particles, 2)) * config.x_limit
v0 = jr.normal(k2, (config.n_particles, 2)) * config.speed

print("Simulating...")

start = time.time()
x_hist, u_hist, e_full, w_cache = simulate(w0, x0, v0, config, key)

w_plot = jax.device_get(w_cache)
u_plot = jax.device_get(u_hist)
x_plot = jax.device_get(x_hist)
end = time.time()

print(f"Simulation finished in {end-start:.2f} seconds")

np.savez_compressed(
    "experiments/simulation_data.npz",
    x_hist=jax.device_get(x_hist),
    u_hist=jax.device_get(u_hist),
    w_cache=jax.device_get(w_cache),
    e_full=jax.device_get(e_full)
)
