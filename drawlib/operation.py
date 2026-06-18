"""
operation.py
PointCloudOperation — applies Warp kernels to a PointCloudData instance.
BallOperation       — steps bouncing-ball physics.
"""

import warp as wp

from drawlib.data import BallData, PointCloudData
from drawlib.kernels import (
    apply_alive_mask, bounce_balls, color_by_interaction,
    influence_points, kill_batch, respawn_escaped, spawn_batch, step_points,
)


class PointCloudOperation:
    VEL_THRESH         = 0.005  # below this speed a point is considered stopped
    SPAWN_PROB = 0.05   # probability per dead particle per frame to activate
    KILL_PROB  = 0.008  # probability per live particle per frame to deactivate

    def __init__(self, data: PointCloudData):
        self.data   = data
        self._frame = 0
        self.influence_radius = 0.35  # world-space units
        self.influence_strength = 6.0    # velocity impulse scale
        self.damping = 0.99  # per-frame velocity multiplier

    def apply_ball_influence(self, ball_data: BallData, dt: float):
        """Push points away from each ball within INFLUENCE_RADIUS."""
        wp.launch(
            influence_points,
            dim=self.data.num_points,
            inputs=[
                self.data.wp_pos,
                self.data.wp_vel,
                ball_data.wp_pos,
                self.influence_radius,
                self.influence_strength,
                dt,
            ],
        )

    def step_positions(self, dt: float):
        """Integrate velocity into position with damping."""
        wp.launch(
            step_points,
            dim=self.data.num_points,
            inputs=[
                self.data.wp_pos,
                self.data.wp_vel,
                dt,
                self.damping,
            ],
        )

    def apply_interaction_colors(self, ball_data: BallData):
        """Blend points toward the color of any ball within influence radius."""
        wp.launch(
            color_by_interaction,
            dim=self.data.num_points,
            inputs=[
                self.data.wp_pos,
                self.data.wp_col,
                ball_data.wp_pos,
                ball_data.wp_colors,
                self.influence_radius,
            ],
        )

    def respawn_escaped(self):
        """Teleport escaped, stopped points back inside the cube as grey."""
        wp.launch(
            respawn_escaped,
            dim=self.data.num_points,
            inputs=[
                self.data.wp_pos,
                self.data.wp_vel,
                self.data.wp_col,
                self.data.cube_half,
                self.VEL_THRESH,
                self._frame,
            ],
        )

    def spawn_particles(self) -> None:
        """Probabilistically activate dead particles with random positions/colours."""
        wp.launch(
            spawn_batch,
            dim=self.data.num_points,
            inputs=[
                self.data.wp_pos, self.data.wp_vel, self.data.wp_col,
                self.data.wp_alive,
                self.data.cube_half,
                self.SPAWN_PROB,
                self._frame,
            ],
        )

    def kill_particles(self) -> None:
        """Probabilistically deactivate live particles."""
        wp.launch(
            kill_batch,
            dim=self.data.num_points,
            inputs=[
                self.data.wp_col, self.data.wp_alive,
                self.KILL_PROB,
                self._frame,
            ],
        )

    def step(self, time: float, dt: float, ball_data: BallData):
        """Full frame: influence -> integrate -> respawn -> color by interaction -> mask dead."""
        self.apply_ball_influence(ball_data, dt)
        self.step_positions(dt)
        self.respawn_escaped()
        self.apply_interaction_colors(ball_data)
        wp.launch(
            apply_alive_mask,
            dim=self.data.num_points,
            inputs=[self.data.wp_col, self.data.wp_alive],
        )
        self._frame += 1
        wp.synchronize()


class BallOperation:
    """Steps bouncing-ball physics via the bounce_balls kernel."""

    def __init__(self, data: BallData):
        self.data = data

    def step(self, dt: float):
        wp.launch(
            bounce_balls,
            dim=self.data.num_balls,
            inputs=[self.data.wp_pos, self.data.wp_vel,
                    dt, self.data.cube_half],
        )
        wp.synchronize()
        