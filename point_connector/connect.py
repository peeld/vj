"""connect.py - Warp HashGrid k-nearest-neighbor edge builder (k=3)."""

import time
import numpy as np
import warp as wp


@wp.kernel
def find_knn_edges_k3(
    points:     wp.array(dtype=wp.vec3),
    grid:       wp.uint64,
    radius:     float,
    edges:      wp.array(dtype=wp.int32),
    edge_count: wp.array(dtype=wp.int32),
    max_edges:  int,
):
    tid = wp.tid()
    pi  = points[tid]

    b0 = float(1.0e30)
    i0 = int(-1)
    b1 = float(1.0e30)
    i1 = int(-1)
    b2 = float(1.0e30)
    i2 = int(-1)

    query    = wp.hash_grid_query(grid, pi, radius)
    neighbor = int(0)
    while wp.hash_grid_query_next(query, neighbor):
        if neighbor == tid:
            continue
        pj = points[neighbor]
        d  = wp.length(pi - pj)
        if d >= radius:
            continue

        worst = b0
        if b1 > worst:
            worst = b1
        if b2 > worst:
            worst = b2

        if d >= worst:
            continue

        if b0 == worst:
            b0 = d
            i0 = neighbor
        elif b1 == worst:
            b1 = d
            i1 = neighbor
        else:
            b2 = d
            i2 = neighbor

    if i0 >= int(0):
        if i0 > tid:
            idx = wp.atomic_add(edge_count, 0, 1)
            if idx < max_edges:
                edges[idx * int(2)]           = tid
                edges[idx * int(2) + int(1)]  = i0

    if i1 >= int(0):
        if i1 > tid:
            idx = wp.atomic_add(edge_count, 0, 1)
            if idx < max_edges:
                edges[idx * int(2)]           = tid
                edges[idx * int(2) + int(1)]  = i1

    if i2 >= int(0):
        if i2 > tid:
            idx = wp.atomic_add(edge_count, 0, 1)
            if idx < max_edges:
                edges[idx * int(2)]           = tid
                edges[idx * int(2) + int(1)]  = i2


def connect_points(
    points_np: np.ndarray,
    k: int = 3,
    radius: float = 10.0,
    device: str | None = None,
) -> tuple[np.ndarray, float]:
    """Build k-nearest-neighbor edges via a Warp HashGrid.

    Returns edges as (E, 2) int32 array (i < j pairs) and kernel elapsed time.
    """
    if k != 3:
        raise ValueError("connect_points requires k=3 (kernel uses 3 fixed slots).")

    if device is None:
        device = "cuda" if wp.is_cuda_available() else "cpu"

    wp.init()

    n         = len(points_np)
    max_edges = n * k * 2

    pts_wp = wp.array(points_np, dtype=wp.vec3, device=device)

    grid = wp.HashGrid(128, 128, 128, device=device)
    grid.build(pts_wp, radius)

    edges_wp = wp.zeros(max_edges * 2, dtype=wp.int32, device=device)
    count_wp = wp.zeros(1,             dtype=wp.int32, device=device)

    t0 = time.perf_counter()
    wp.launch(
        kernel=find_knn_edges_k3,
        dim=n,
        inputs=[pts_wp, grid.id, radius, edges_wp, count_wp, max_edges],
        device=device,
    )
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - t0

    edge_count = int(count_wp.numpy()[0])
    edges      = edges_wp.numpy()[: edge_count * 2].reshape(-1, 2)
    return edges, elapsed
