"""visualize.py — 3D scatter + edge plot using matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def visualize(points: np.ndarray, edges: np.ndarray, title: str = "Point Cloud Connector") -> None:
    """
    Render a 3D scatter of *points* with lines for every edge in *edges*.

    Parameters
    ----------
    points : (N, 3) float array
    edges  : (E, 2) int array — index pairs
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter points
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=2, c="steelblue", depthshade=True, zorder=2,
    )

    # Build Line3DCollection for all edges at once (much faster than per-edge plotting)
    if len(edges) > 0:
        segs = np.stack(
            [points[edges[:, 0]], points[edges[:, 1]]], axis=1
        )  # shape (E, 2, 3)
        lc = Line3DCollection(segs, linewidths=0.4, alpha=0.25, colors="tomato", zorder=1)
        ax.add_collection3d(lc)

    ax.set_title(f"{title}\n{len(points):,} points · {len(edges):,} edges")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)

    plt.tight_layout()
    plt.show()
