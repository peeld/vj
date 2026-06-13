"""
scene.py
Base class for all Warp visualisation scenes.

To create a new scene, subclass Scene and implement the three abstract methods.
Optionally declare a `post_effect` class attribute to enable post-processing.

Example
-------
    from scene       import Scene
    from post_effect import FeedbackPostEffect
    from warp_feedback import FeedbackParams

    class MyScene(Scene):
        title       = "My Scene"
        post_effect = FeedbackPostEffect(FeedbackParams(decay=0.97, hue_shift=0.015))

        def setup(self, ctx):
            self.draw_ = PointsDrawable(ctx)
            self.draw_.setup(...)

        def step(self, t, dt):
            wp.launch(my_kernel, dim=N, inputs=[...])
            self.draw_.write_warp(...)

        def draw(self, mvp):
            self.draw_.draw(mvp)

        def on_key(self, key, action, keys):
            if key == keys.R:
                self.reset()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import moderngl


class Scene(ABC):
    """
    Contract between a visualisation scene and Viewport3D.

    Lifecycle
    ---------
    1. Viewport3D instantiates the scene and calls ``setup(ctx)``.
    2. Each frame: ``step(t, dt)`` then ``draw(mvp)``.
    3. Key presses not consumed by the viewport are forwarded to ``on_key``.

    Post-processing
    ---------------
    Declare a ``post_effect`` class attribute (a PostEffect instance, or None).
    Viewport3D checks this once at init and wires up the FBO pipeline if set.
    You can swap it out at runtime by assigning a new PostEffect to the instance.

    Auto-rotate
    -----------
    The viewport auto-rotates the camera (cam_yaw += dt) by default.
    Set ``auto_rotate = False`` on your subclass to disable.
    """

    # Override in subclass ────────────────────────────────────────────────────
    title:       str          = "Warp Viewport"
    post_effect               = None   # PostEffect instance or None
    auto_rotate: bool         = True
    cam_yaw:     float        = 35.0   # initial camera yaw   (degrees)
    cam_pitch:   float        = -25.0  # initial camera pitch  (degrees)
    cam_dist:    float        = 3.8    # initial camera distance

    # Abstract interface ──────────────────────────────────────────────────────

    @abstractmethod
    def setup(self, ctx: moderngl.Context) -> None:
        """
        Create drawables, allocate Warp arrays, compile shaders.
        Called once after the GL context is ready.
        ``ctx`` is the ModernGL context — pass it to every Drawable constructor.
        """

    @abstractmethod
    def step(self, t: float, dt: float) -> None:
        """
        Advance simulation by ``dt`` seconds; ``t`` is total elapsed time.
        Also call ``drawable.write_warp(...)`` here to push data to GL.
        """

    @abstractmethod
    def draw(self, mvp: np.ndarray) -> None:
        """
        Issue draw calls. The correct framebuffer is already bound by
        Viewport3D (screen or off-screen FBO depending on post_effect).
        ``mvp`` is a column-major float32 (4,4) array ready for
        ``prog["mvp"].write(mvp.tobytes())``.
        """

    # Optional overrides ──────────────────────────────────────────────────────

    def on_key(self, key, action, keys) -> None:
        """
        Handle keyboard events not consumed by the viewport.
        ``keys`` is ``self.wnd.keys`` from moderngl-window, e.g. ``keys.R``.
        ``action`` values: ``keys.ACTION_PRESS``, ``keys.ACTION_RELEASE``.
        """
