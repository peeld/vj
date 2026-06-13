"""
operation.py
PointCloudOperation — applies Warp kernels to a PointCloudData instance.
BallOperation       — steps bouncing-ball physics.
"""

import warp as wp

from drawlib.data import BallData, PointCloudData
from drawlib.kernels import bounce_balls, color_by_interaction, influence_points, respawn_escaped, step_points


class PointCloudOperation:
    INFLUENCE_RADIUS   = 0.35   # world-space units
    INFLUENCE_STRENGTH = 6.0    # velocity impulse scale
    DAMPING            = 0.99   # per-frame velocity multiplier
    VEL_THRESH         = 0.005  # below this speed a point is considered stopped

    def __init__(self, data: PointCloudData):
        self.data   = data
        self._frame = 0

    def apply_ball_influence(self, ball_data: BallData, dt: float):
        """Push points away from each ball within INFLUENCE_RADIUS."""
        wp.launch(
            influence_points,
            dim=self.data.num_points,
            inputs=[
                self.data.wp_pos,
                self.data.wp_vel,
                ball_data.wp_pos,
                self.INFLUENCE_RADIUS,
                self.INFLUENCE_STRENGTH,
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
                self.DAMPING,
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
                self.INFLUENCE_RADIUS,
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

    def step(self, time: float, dt: float, ball_data: BallData):
        """Full frame: influence -> integrate -> respawn -> color by interaction."""
        self.apply_ball_influence(ball_data, dt)
        self.step_positions(dt)
        self.respawn_escaped()
        self.apply_interaction_colors(ball_data)
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
        