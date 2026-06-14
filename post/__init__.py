"""
post — post-processing effects package.

Drop-in import for gui_merged and any new GUI:

    from post import FeedbackPostEffect, FeedbackParams, BLEND_MODES, SMEAR_PATTERNS
    from post import PassThroughEffect
    from post import GlitchEffect
    from post import BokehEffect
    from post import PRESETS, EffectPreset
    from post import PostEffect          # base class for custom effects

Contents
--------
  base.py           PostEffect (ABC), shared quad shaders, DEVICE
  warp_feedback.py  FeedbackLoop, FeedbackParams, SMEAR_PATTERNS
  feedback.py       FeedbackPostEffect, BLEND_MODES, EffectPreset, PRESETS
  pass_through.py   PassThroughEffect
  glitch.py         GlitchEffect
  bokeh.py          BokehEffect
"""

from post.base          import PostEffect, DEVICE
from post.warp_feedback import FeedbackLoop, FeedbackParams, SMEAR_PATTERNS
from post.feedback      import FeedbackPostEffect, BLEND_MODES, EffectPreset, PRESETS
from post.pass_through  import PassThroughEffect
from post.glitch        import GlitchEffect
from post.bokeh         import BokehEffect

__all__ = [
    "PostEffect",
    "DEVICE",
    "FeedbackLoop",
    "FeedbackParams",
    "SMEAR_PATTERNS",
    "FeedbackPostEffect",
    "BLEND_MODES",
    "EffectPreset",
    "PRESETS",
    "PassThroughEffect",
    "GlitchEffect",
    "BokehEffect",
]
