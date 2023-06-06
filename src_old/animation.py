import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation
import numpy as np


def animate_spacecraft(t, s):
    """
    Example usage:
        import matplotlib.pyplot as plt
        from animations import animate_cartpole
        fig, ani = animate_cartpole(t, x, θ)
        ani.save('cartpole.mp4', writer='ffmpeg')
        plt.show()
    """
    # Geometry
    chaser_side_length = 1
    target_side_length = 1
    chaser_color = 'tab:blue'
    target_color = 'tab:orange'

    # cart_width = 2.
    # cart_height = 1.
    # wheel_radius = 0.3
    # wheel_sep = 1.
    # pole_length = 5.
    # mass_radius = 0.25

    # Figure and axis
    fig, ax = plt.subplots(dpi=100)
    # x_min, x_max = np.min(x) - 1.1*pole_length, np.max(x) + 1.1*pole_length
    x_min, x_max = -100, 100
    y_min, y_max = -100, 100
    # z_min, z_max = -100, 100
    # y_min = -pole_length
    # y_max = 1.1*(wheel_radius + cart_height + pole_length)
    ax.plot([x_min, x_max], [0., 0.], '-', linewidth=1, color='k')[0]
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_yticks([])
    ax.set_aspect(1.)

    # Artists
    cart = mpatches.FancyBboxPatch((0., 0.), cart_width, cart_height,
                                   facecolor='tab:blue', edgecolor='k',
                                   boxstyle='Round,pad=0.,rounding_size=0.05')
    # wheel_left = mpatches.Circle((0., 0.), wheel_radius, color='k')
    # wheel_right = mpatches.Circle((0., 0.), wheel_radius, color='k')
    mass = mpatches.Circle((0., 0.), mass_radius, color='k')
    pole = ax.plot([], [], '-', linewidth=3, color='k')[0]
    trace = ax.plot([], [], '--', linewidth=2, color='tab:orange')[0]
    timestamp = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    ax.add_patch(cart)
    ax.add_patch(wheel_left)
    ax.add_patch(wheel_right)
    ax.add_patch(mass)