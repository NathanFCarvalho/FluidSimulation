import numpy as np

# Grid constants
GRID_SIZE = 64
X_LIMIT = 2*np.pi
Y_LIMIT = 2*np.pi

# Navier-Stokes constants
NU = 0.001            
STEP_SIZE = 0.0001    
N_STEPS = 10000

# Animation constants
SAVE_EVERY = 100

# Particle constants
N_PARTICLES = 4
PARTICLE_SIZE = 20
MASS = 1000