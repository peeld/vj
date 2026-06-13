"""
drawlib/post_effect.py — backward-compatibility shim.

All post-processing code now lives in the ``post`` package.
This file re-exports everything so that existing code (gui.py, gui2.py,
gui3.py, gui4.py) keeps working without modification.

New code should import directly from ``post``:

    from post import FeedbackPostEffect, BLEND_MODES, PostEffect
"""

from post.base import PostEffect, DEVICE                        # noqa: F401
from post.feedback import FeedbackPostEffect, BLEND_MODES      # noqa: F401

__all__ = ["PostEffect", "DEVICE", "FeedbackPostEffect", "BLEND_MODES"]
