"""
drawlib/warp_feedback.py — backward-compatibility shim.

All feedback-loop code now lives in ``post.warp_feedback``.
This file re-exports everything so that existing code (gui.py, gui2.py,
gui3.py, gui4.py, audio_fft_warp.py) keeps working without modification.

New code should import directly from ``post``:

    from post import FeedbackLoop, FeedbackParams, SMEAR_PATTERNS
"""

from post.warp_feedback import (        # noqa: F401
    FeedbackParams,
    FeedbackLoop,
    SMEAR_PATTERNS,
)

__all__ = ["FeedbackParams", "FeedbackLoop", "SMEAR_PATTERNS"]
