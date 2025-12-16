import os
import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import imageio
from matplotlib.animation import FuncAnimation
class RipsFiltrationAnimator:
    def __init__(
        self,
        points,
        max_edge_length=1.5,
        max_dimension=2,
        out_dir="frames",
        figsize=(5, 5),
    ):
        """
        Parameters
        ----------
        points : np.ndarray, shape (n_points, 2)
            Point cloud (2D)
        max_edge_length : float
            Maximum Rips edge length
        max_dimension : int
            Maximum simplex dimension (2 for triangles)
        out_dir : str
            Directory to store frames
        figsize : tuple
            Matplotlib figure size
        """
        self.points = np.asarray(points)
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension
        self.out_dir = out_dir
        self.figsize = figsize

        os.makedirs(self.out_dir, exist_ok=True)

        self._build_complex()
        self._extract_filtration_values()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_complex(self):
        self.rips = gd.RipsComplex(
            points=self.points,
            max_edge_length=self.max_edge_length
        )
        self.simplex_tree = self.rips.create_simplex_tree(
            max_dimension=self.max_dimension
        )

    def _extract_filtration_values(self):
        self.filtration_values = sorted(
            {f for _, f in self.simplex_tree.get_filtration()}
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_at_epsilon(self, epsilon, ax):
        """
        Plot Rips complex at filtration value epsilon.
        """
        ax.clear()
        ax.set_aspect("equal")
        ax.set_title(f"Vietoris–Rips filtration  ε = {epsilon:.2f}")

        # Plot points
        ax.scatter(
            self.points[:, 0],
            self.points[:, 1],
            s=40,
            zorder=3
        )

        # Plot simplices
        for simplex, filt in self.simplex_tree.get_skeleton(self.max_dimension):
            if filt <= epsilon:
                if len(simplex) == 2:
                    i, j = simplex
                    ax.plot(
                        [self.points[i, 0], self.points[j, 0]],
                        [self.points[i, 1], self.points[j, 1]],
                        linewidth=1,
                        zorder=1
                    )
                elif len(simplex) == 3:
                    tri = self.points[list(simplex)]
                    ax.fill(
                        tri[:, 0],
                        tri[:, 1],
                        alpha=0.25,
                        zorder=0
                    )

        pad = 0.5
        ax.set_xlim(
            self.points[:, 0].min() - pad,
            self.points[:, 0].max() + pad
        )
        ax.set_ylim(
            self.points[:, 1].min() - pad,
            self.points[:, 1].max() + pad
        )

    # ------------------------------------------------------------------
    # Frame generation
    # ------------------------------------------------------------------

    def save_frames(self, dpi=150):
        """
        Save one image per filtration value.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for k, eps in enumerate(self.filtration_values):
            self.plot_at_epsilon(eps, ax)
            plt.savefig(
                os.path.join(self.out_dir, f"frame_{k:03d}.png"),
                dpi=dpi
            )

        plt.close(fig)

    # ------------------------------------------------------------------
    # Animations
    # ------------------------------------------------------------------

    def save_gif(self, filename="rips_filtration.gif", duration=0.4):
        """
        Save GIF animation from generated frames.
        """
        if not os.listdir(self.out_dir):
            self.save_frames()

        with imageio.get_writer(filename, mode="I", duration=duration) as writer:
            for k in range(len(self.filtration_values)):
                frame_path = os.path.join(self.out_dir, f"frame_{k:03d}.png")
                image = imageio.imread(frame_path)
                writer.append_data(image)

    def save_mp4(self, filename="rips_filtration.mp4", fps=3):
        """
        Save MP4 animation (requires ffmpeg).
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        def update(frame):
            self.plot_at_epsilon(self.filtration_values[frame], ax)

        ani = FuncAnimation(
            fig,
            update,
            frames=len(self.filtration_values),
            interval=1000 // fps
        )

        ani.save(filename, fps=fps)
        plt.close(fig)