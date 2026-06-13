"""main.py — entry point for the GPU Point Cloud Connector demo."""

from generate import generate_points
from connect import connect_points
from visualize import visualize

N = 5000
K = 3
RADIUS = 10.0


def main() -> None:
    # 1. Generate
    points = generate_points(n=N)
    print(f"Points: {len(points):,}")

    # 2. Connect
    edges, elapsed = connect_points(points, k=K, radius=RADIUS)
    print(f"Edges:  {len(edges):,}")
    print(f"Connect time: {elapsed:.3f}s")

    # 3. Visualize
    visualize(points, edges)


if __name__ == "__main__":
    main()
