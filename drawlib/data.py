"""
data.py
PointCloudData -- owns all Warp arrays and point-cloud configuration.
BallData       -- 5 bouncing balls with positions, velocities, and colors.
"""

import numpy as np
import warp as wp


class PointCloudData:
    NUM_POINTS    = 1_000_000
    CUBE_HALF     = 1.0
    INITIAL_SPEED = 0.05   # small random starting velocity magnitude

    def __init__(self, seed: int = 42):
        self.num_points = self.NUM_POINTS
        self.cube_half  = self.CUBE_HALF

        self.wp_pos: wp.array = None
        self.wp_vel: wp.array = None
        self.wp_col: wp.array = None

        self._init_arrays(seed)

    def _init_arrays(self, seed: int):
        rng = np.random.default_rng(seed)
        pos = rng.uniform(
            -self.cube_half, self.cube_half,
            (self.num_points, 3)
        ).astype(np.float32)
        vel = rng.uniform(
            -self.INITIAL_SPEED, self.INITIAL_SPEED,
            (self.num_points, 3)
        ).astype(np.float32)

        # grey = np.full((self.num_points, 4), [0.5, 0.5, 0.5, 1.0], dtype=np.float32)
        black = np.full((self.num_points, 4), [0.05, 0.05, 0.05, 0.01], dtype=np.float32)

        self.wp_pos = wp.array(pos, dtype=wp.vec3)
        self.wp_vel = wp.array(vel, dtype=wp.vec3)
        self.wp_col = wp.array(black, dtype=wp.vec4)

    def randomize(self):
        """Reset points to a fresh random distribution inside the cube."""
        rng = np.random.default_rng()
        pos = rng.uniform(
            -self.cube_half, self.cube_half,
            (self.num_points, 3)
        ).astype(np.float32)
        vel = rng.uniform(
            -self.INITIAL_SPEED, self.INITIAL_SPEED,
            (self.num_points, 3)
        ).astype(np.float32)
        grey = np.full((self.num_points, 4), [0.5, 0.5, 0.5, 0.2], dtype=np.float32)
        self.wp_pos = wp.array(pos, dtype=wp.vec3)
        self.wp_vel = wp.array(vel, dtype=wp.vec3)
        self.wp_col = wp.array(grey, dtype=wp.vec4)

    def positions_numpy(self) -> np.ndarray:
        return self.wp_pos.numpy()

    def colors_numpy(self) -> np.ndarray:
        return self.wp_col.numpy()


class BallData:
    """5 bouncing balls -- positions, velocities, and fixed per-ball colors."""

    NUM_BALLS = 5
    COLORS = np.array([
        [1.0, 0.30, 0.30, 1.0],
        [0.30, 1.0, 0.30, 1.0],
        [0.35, 0.55, 1.0, 1.0],
        [1.0, 0.90, 0.20, 1.0],
        [1.0, 0.40, 1.0, 1.0],
    ], dtype=np.float32)

    def __init__(self, cube_half: float = 1.0, speed: float = 0.8, seed: int = 7):
        self.num_balls = self.NUM_BALLS
        self.cube_half = cube_half

        rng = np.random.default_rng(seed)
        pos = rng.uniform(-cube_half * 0.5, cube_half * 0.5,
                          (self.num_balls, 3)).astype(np.float32)
        vel = rng.uniform(-speed, speed,
                          (self.num_balls, 3)).astype(np.float32)

        self.wp_pos    = wp.array(pos, dtype=wp.vec3)
        self.wp_vel    = wp.array(vel, dtype=wp.vec3)
        self.wp_colors = wp.array(self.COLORS, dtype=wp.vec4)

    def positions_numpy(self) -> np.ndarray:
        return self.wp_pos.numpy()
