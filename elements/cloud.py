"""
elements/cloud.py
CloudElement -- encapsulates the point-cloud + bouncing-ball simulation.

Owns:
  - PointCloudData / BallData (Warp arrays)
  - PointCloudOperation / BallOperation (kernel drivers)
  - PointsDrawable / LinesDrawable / ShapeDrawable (GL renderables)

Implements the DrawingElement interface (elements/base.py) directly so
MergedGUI.elements can drive it the same as every other scene element.

Usage::

    cloud = CloudElement(ctx)

    # each frame
    cloud.step(ctx)        # run Warp kernels + upload to GL
    cloud.draw(mvp, ctx)   # issue draw calls

    # on R-key
    cloud.regen()
"""

import moderngl

from .base import DrawingElement, FrameContext, register_element_type, Prop
from drawlib.data import BallData, PointCloudData
from drawlib.operation import BallOperation, PointCloudOperation
from drawlib.drawable import LinesDrawable, PointsDrawable, ShapeDrawable
from drawlib.helpers import build_wireframe


class CloudElement(DrawingElement, section="cloud"):
    """Point-cloud + bouncing-ball scene element.

    Parameters
    ----------
    ctx:
        Active ModernGL context.
    """
    kind = "cloud"
    ball_size = Prop("Ball Size", float, 1.0, 0.01, 1.0, 0.05,
                     description="World space radius of the influence ball")

    def __init__(self, ctx: moderngl.Context, device=None, **kwargs):
        super().__init__()
        self.cloud_data = PointCloudData()
        self.ball_data  = BallData(cube_half=self.cloud_data.cube_half)
        self.cloud_op   = PointCloudOperation(self.cloud_data)
        self.ball_op    = BallOperation(self.ball_data)

        self.ball_size = 1.0

        self.cloud_draw = PointsDrawable(ctx)
        self.cloud_draw.setup(
            self.cloud_data.positions_numpy(),
            self.cloud_data.colors_numpy(),
        )

        #self.wire_draw = LinesDrawable(ctx)
        #self.wire_draw.setup(*build_wireframe(self.cloud_data.cube_half))

        self.ball_draw = ShapeDrawable(ctx)
        self.ball_draw.setup(self.ball_data.positions_numpy(), BallData.COLORS)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, ctx: FrameContext) -> None:
        """Run physics kernels and upload results to GL buffers.

        While visible, dead particles are probabilistically spawned in.
        While inactive, live particles are probabilistically killed off.
        """
        self.ball_op.step(ctx.frame_time)

        if self.active:
            self.cloud_op.spawn_particles()
        else:
            self.cloud_op.kill_particles()

        self.cloud_op.influence_radius = self.ball_size

        self.cloud_op.step(ctx.time, ctx.frame_time, self.ball_data)
        self.cloud_draw.write_warp(self.cloud_data.wp_pos, self.cloud_data.wp_col)
        self.ball_draw.write_warp(self.ball_data.wp_pos)

    def draw(self, mvp, ctx: FrameContext) -> None:
        """Issue draw calls for the cloud, wireframe, and balls."""
        self.cloud_draw.draw(mvp)
        # self.wire_draw.draw(mvp)
        # self.ball_draw.draw(mvp, point_size=80.0)

    def regen(self) -> None:
        """Reset cloud points to a fresh random distribution."""
        self.cloud_data.randomize()
        self.cloud_draw.update(
            self.cloud_data.positions_numpy(),
            self.cloud_data.colors_numpy(),
        )


register_element_type("cloud", CloudElement)
