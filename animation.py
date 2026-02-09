import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def animate_simulation(x_hist, u_hist, w_cache, e_full, config, fps=30, nframes=500, min_energy=1e-6):
    stride = max(1, x_hist.shape[0] // nframes)
    x_hist, u_hist, w_cache = x_hist[::stride], u_hist[::stride], w_cache[::stride]

    sim_len = len(x_hist)
    u_ref = np.max(np.sqrt(u_hist[0,0]**2 + u_hist[0,1]**2)) + 1e-9

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 5), dpi=90)
    canvas = FigureCanvasAgg(fig)

    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.25)
    ax1 = fig.add_subplot(gs[0])

    img = ax1.imshow(
        w_cache[0].T,
        origin='lower',
        extent=[0, config.x_limit, 0, config.y_limit],
        cmap='magma',
        alpha=0.5,
        interpolation='bilinear'
    )

    sub = max(1, config.nx // 30)
    Y, X = np.mgrid[0:config.ny:sub, 0:config.nx:sub]
    x_pos = X * (config.x_limit / config.nx)
    y_pos = Y * (config.y_limit / config.ny)
    quiv = ax1.quiver(
        x_pos, y_pos,
        np.zeros_like(x_pos), np.zeros_like(y_pos),
        color='white', alpha=0.6, scale=u_ref*25, width=0.0025, headwidth=4, headlength=5
    )

    scat = ax1.scatter([], [], s=8, c='cyan', edgecolors='none', alpha=0.6)
    ax1.set_title("Particle Dynamics", fontsize=16, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1])
    line, = ax2.plot([], [], color='cyan', lw=2)
    ax2.set_title("Kinetic Energy (log scale)", fontsize=18, fontweight='bold')
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Energy", fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle=':', alpha=0.3)

    t = np.linspace(0, 1, nframes)
    frame_indices = (t**1.5 * (sim_len - 1)).astype(int) 

    global_vmax = np.max(np.abs(w_cache)) + 1e-9

    with imageio.get_writer('simulation.gif', mode='I', fps=fps) as writer:
        for i in range(nframes):
            idx = frame_indices[i]

            vorticity = w_cache[idx].T
            v_norm = np.abs(vorticity) / global_vmax
            v_scaled = np.sign(vorticity) * v_norm**0.4
            img.set_data(v_scaled)
            img.set_cmap('magma')
            img.set_norm(mcolors.Normalize(vmin=-1, vmax=1))
            alpha = 0.3 + 0.7 * np.abs(v_scaled)
            img.set_alpha(alpha)

            ux = u_hist[idx,0][X,Y]
            uy = u_hist[idx,1][X,Y]
            quiv.set_UVC(ux, uy)

            scat.set_offsets(x_hist[idx])

            start_idx = min(50, len(e_full)-1)
            curr_e_idx = int((idx / sim_len) * len(e_full))
            if curr_e_idx > start_idx:
                t_plot = np.arange(start_idx, curr_e_idx) * config.dt
                e_plot = e_full[start_idx:curr_e_idx]
                e_safe = np.maximum(e_plot, min_energy)

                line.set_data(t_plot, e_safe)
                ax2.set_xlim(t_plot[0], t_plot[-1])
                ax2.set_ylim(np.min(e_safe)*0.8, np.max(e_safe)*1.05)

            canvas.draw()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)

    plt.close(fig)
