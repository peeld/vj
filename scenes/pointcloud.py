"""
scenes/pointcloud.py
The original point-cloud + bouncing-balls experiment, ported to the Scene API.

Run with:
    python -c "from scenes.pointcloud import PointCloudScene; from viewport import run; run(PointCloudScene)"

Or create a one-line launcher (recommended):
    # run_pointcloud.py
    from viewport import run
    from scenes.pointcloud import PointCloudScene
    run(PointCloudScene)
"""

from __future__ import annotations

import sys
import os

# Allow running from the project root (warp/)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import moderngl

from drawlib.scene import Scene
from drawlib.post_effect import FeedbackPostEffect
from drawlib.drawable import LinesDrawable, PointsDrawable, ShapeDrawable
from drawlib.helpers import wireframe_cube
from drawlib.data import BallData, PointCloudData
from drawlib.operation import BallOperation, PointCloudOperation
from drawlib.warp_feedback import FeedbackParams


class PointCloudScene(Scene):
    """
    250 k GPU particles inside a wireframe cube, disturbed by 5 bouncing balls.
    Points colour-bleed toward whichever ball is closest.
    """

    title       = "Warp — Point Cloud"
    auto_rotate = True

    # Post-effect — comment out or set to None to render direct
    post_effect = FeedbackPostEffect(
        params=FeedbackParams(
            base_zoom        = 1.008,
            zoom_sensitivity = 0.0,
            base_rot         = 0.003,
            rot_sensitivity  = 0.0,
            decay            = 0.985,
            ripple_strength  = 8.0,
            ripple_freq      = 10.0,
            hue_shift        = 0.018,
            chroma_offset    = 0.012,
            sat_boost        = 1.15,
        ),
        scene_alpha=0.13,
    )

    # ── Scene interface ───────────────────────────────────────────────────────

    def setup(self, ctx: moderngl.Context) -> None:
        # Data
        self._cloud = PointCloudData()
        self._balls = BallData(cube_half=self._cloud.cube_half)

        # Operations
        self._cloud_op = PointCloudOperation(self._cloud)
        self._ball_op  = BallOperation(self._balls)

        # Drawables
        self._cloud_draw = PointsDrawable(ctx)
        self._cloud_draw.setup(
            self._cloud.positions_numpy(),
            self._cloud.colors_numpy(),
        )

        self._wire_draw = LinesDrawable(ctx)
        self._wire_draw.setup(*wireframe_cube(self._cloud.cube_half))

        self._ball_draw = ShapeDrawable(ctx)
        self._ball_draw.setup(
            self._balls.positions_numpy(),
            BallData.COLORS,
        )

    def step(self, t: float, dt: float) -> None:
        self._ball_op.step(dt)
        self._cloud_op.step(t, dt, self._balls)

        # GPU → GPU upload via CUDA-GL interop
        self._cloud_draw.write_warp(self._cloud.wp_pos, self._cloud.wp_col)
        self._ball_draw.write_warp(self._balls.wp_pos)

    def draw(self, mvp: np.ndarray) -> None:
        self._cloud_draw.draw(mvp)
        self._wire_draw.draw(mvp)
        # Balls are currently hidden — uncomment to show:
        # self._ball_draw.draw(mvp, point_size=80.0)

    def on_key(self, key, action, keys) -> None:
        if action != keys.ACTION_PRESS:
            return
        if key == keys.R:
            self._cloud.randomize()
            self._cloud_draw.update(
                self._cloud.positions_numpy(),
                self._cloud.colors_numpy(),
            )
            print("[scene] point cloud randomised")
