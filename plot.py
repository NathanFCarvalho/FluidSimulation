import time
import numpy as np
import yaml

from animation import animate_simulation
from navier import SimulationConfig

if __name__ == "__main__":
    with open("config.yml", "r") as file:
        params = yaml.safe_load(file)


data = np.load("experiments/simulation_data.npz")
x_plot = data["x_hist"]
u_plot = data["u_hist"]
w_plot = data["w_cache"]
e_full = data["e_full"]

config = SimulationConfig(**params)

print("Plotting...")

start = time.time()
animate_simulation(x_plot, u_plot, w_plot, e_full, config)
end = time.time()

print(f"Plotting finished in {end-start:.2f} seconds")