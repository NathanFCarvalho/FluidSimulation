import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from navier import *
from constants import *

simulator = Simulator(
    x_limit=X_LIMIT,
    y_limit=Y_LIMIT,
    grid_size=GRID_SIZE,
    step_size=STEP_SIZE,
    n_steps=N_STEPS,
)

w0 = 2 * np.sin(simulator.X) * np.sin(simulator.Y)
zero_force = ZeroForce(simulator)

ns = SpectralNavierStokes(
    simulator=simulator,
    external_force=zero_force,
    initial_vorticity=w0,
    nu=NU,
)

particle_init = np.random.rand(N_PARTICLES, 2) * [X_LIMIT, Y_LIMIT]
v0 = np.random.rand(N_PARTICLES, 2) * 100.0
part = Particle(
    simulator=simulator,
    n_particles=N_PARTICLES,
    x0=particle_init,
    v0=v0,
    field=None,
    mass=MASS,
)

coupled = CoupledNavier(navier=ns, particle=part)
velocity, vorticity, paths, force_cache = coupled.solve()

velocity_anim = velocity[::SAVE_EVERY]
paths_anim = np.array(paths[::SAVE_EVERY])
force_anim = force_cache[::SAVE_EVERY]
n_frames = len(velocity_anim)
decimate = 4  # quiver decimation

fig, (ax_particles, ax_force) = plt.subplots(1, 2, figsize=(12, 6))

Q = ax_particles.quiver(
    simulator.X[::decimate, ::decimate],
    simulator.Y[::decimate, ::decimate],
    velocity_anim[0][0][::decimate, ::decimate],
    velocity_anim[0][1][::decimate, ::decimate],
    color='white', scale=50
)
sc = ax_particles.scatter(paths_anim[0,:,0], paths_anim[0,:,1], c='cyan', s=PARTICLE_SIZE, edgecolor='none')
ax_particles.set_title("Particles + Velocity")
ax_particles.set_xlabel("x"); ax_particles.set_ylabel("y")
ax_particles.set_xlim(0, X_LIMIT)
ax_particles.set_ylim(0, Y_LIMIT)
ax_particles.set_facecolor('k')
ax_particles.set_xticks([]); ax_particles.set_yticks([])

Fmag = np.sqrt(force_anim[0,0]**2 + force_anim[0,1]**2)
pc = ax_force.pcolormesh(simulator.X, simulator.Y, Fmag, shading='gouraud', cmap='plasma')
Qf = ax_force.quiver(
    simulator.X[::decimate, ::decimate],
    simulator.Y[::decimate, ::decimate],
    force_anim[0,0][::decimate, ::decimate],
    force_anim[0,1][::decimate, ::decimate],
    color='lime', scale=50
)
scf = ax_force.scatter(paths_anim[0,:,0], paths_anim[0,:,1], color='cyan', s=PARTICLE_SIZE, edgecolor='none')
ax_force.set_title("Coupling Force")
ax_force.set_xlabel("x"); ax_force.set_ylabel("y")
ax_force.set_xlim(0, X_LIMIT)
ax_force.set_ylim(0, Y_LIMIT)
ax_force.set_xticks([]); ax_force.set_yticks([])
fig.colorbar(pc, ax=ax_force, label="|F|")

def update(frame):
    # Left panel
    sc.set_offsets(paths_anim[frame,:,:])
    Q.set_UVC(
        velocity_anim[frame][0][::decimate, ::decimate],
        velocity_anim[frame][1][::decimate, ::decimate]
    )
    ax_particles.set_title(f"Particles + Velocity - Frame {frame}")

    # Right panel
    Fmag = np.sqrt(force_anim[frame,0]**2 + force_anim[frame,1]**2)
    pc.set_array(Fmag.ravel())
    Qf.set_UVC(
        force_anim[frame,0][::decimate, ::decimate],
        force_anim[frame,1][::decimate, ::decimate]
    )
    scf.set_offsets(paths_anim[frame,:,:])
    ax_force.set_title(f"Coupling Force - Frame {frame}")

    return Q, sc, pc, Qf, scf

ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)

ani.save("particles_simulation.gif", writer='pillow', fps=20)

plt.show()
