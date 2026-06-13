"""
elements/cloud.py
CloudElement -- encapsulates the point-cloud + bouncing-ball simulation.

Owns:
  - PointCloudData / BallData (Warp arrays)
  - PointCloudOperation / BallOperation (kernel drivers)
  - PointsDrawable / LinesDrawable / ShapeDrawable (GL renderables)

Usage::

    cloud = CloudElement(ctx)

    # each frame
    cloud.step(time, frame_time)   # run Warp kernels + upload to GL
    cloud.draw(mvp)                # issue draw calls

    # on R-key
    cloud.randomize()
"""

import moderngl

from drawlib.data import BallData, PointCloudData
from drawlib.operation import BallOperation, PointCloudOperation
from drawlib.drawable import LinesDrawable, PointsDrawable, ShapeDrawable
from drawlib.helpers import build_wireframe


class CloudElement:
    """Point-cloud + bouncing-ball scene element.

    Parameters
    ----------
    ctx:
        Active ModernGL context.
    """

    def __init__(self, ctx: moderngl.Context):
        self.cloud_data = PointCloudData()
        self.ball_data  = BallData(cube_half=self.cloud_data.cube_half)
        self.cloud_op   = PointCloudOperation(self.cloud_data)
        self.ball_op    = BallOperation(self.ball_data)

        self.cloud_draw = PointsDrawable(ctx)
        self.cloud_draw.setup(
            self.cloud_data.positions_numpy(),
            self.cloud_data.colors_numpy(),
        )

        self.wire_draw = LinesDrawable(ctx)
        self.wire_draw.setup(*build_wireframe(self.cloud_data.cube_half))

        self.ball_draw = ShapeDrawable(ctx)
        self.ball_draw.setup(self.ball_data.positions_numpy(), BallData.COLORS)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, time: float, frame_time: float) -> None:
        """Run physics kernels and upload results to GL buffers."""
        self.ball_op.step(frame_time)
        self.cloud_op.step(time, frame_time, self.ball_data)
        self.cloud_draw.write_warp(self.cloud_data.wp_pos, self.cloud_data.wp_col)
        self.ball_draw.write_warp(self.ball_data.wp_pos)

    def draw(self, mvp) -> None:
        """Issue draw calls for the cloud, wireframe, and balls."""
        self.cloud_draw.draw(mvp)
        self.wire_draw.draw(mvp)
        # self.ball_draw.draw(mvp, point_size=80.0)

    def randomize(self) -> None:
        """Reset cloud points to a fresh random distribution."""
        self.cloud_data.randomize()
        self.cloud_draw.update(
            self.cloud_data.positions_numpy(),
            self.cloud_data.colors_numpy(),
        )
