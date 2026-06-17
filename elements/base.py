"""
elements/base.py - Common interface for dynamically-managed scene elements.

DrawingElement is the contract a host's render loop, regen, and palette
pushes rely on so every element can be treated identically instead of
branching per concrete class.

See elements/readme.md for the full guide to implementing a new element
type, including the FrameContext fields, rendering/compute conventions,
and a worked example.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


@dataclass
class FrameContext:
    """Per-frame inputs handed to every DrawingElement.step() / draw().

    Carries the union of what the current element types need (scene clock,
    camera basis) so step()/draw() take one argument regardless of which
    underlying values a given element actually reads.
    """
    time: float
    current_time: float
    frame_time: float
    cam_eye: object
    cam_fwd: object
    cam_right: object
    cam_up: object


class DrawingElement(ABC):
    """Base class for a dynamically-managed scene element.

    `kind` is the registry key for this type (see ELEMENT_TYPES below).
    `name` defaults to `kind` -- at most one live instance per kind is ever
    permitted, so `kind` alone is already a unique label for removal and UI
    display.
    """
    kind: str = "element"

    def __init__(self) -> None:
        self.name: str = self.kind
        self.visible: bool = True   # show/hide rendering only; does not affect simulation
        self.active: bool = True    # when False, stop spawning and let existing particles die off

    @abstractmethod
    def step(self, ctx: FrameContext) -> None:
        """Advance simulation for one frame. Called every frame regardless
        of visible/active -- elements check self.active internally to gate spawning."""

    @abstractmethod
    def draw(self, mvp, ctx: FrameContext) -> None:
        """Issue draw calls. Only called by the owner when visible is True."""

    def regen(self) -> None:
        """Reseed/rebuild this element. No-op for elements that don't support it."""

    def set_palette(self, palette: list) -> None:
        """Push a colour palette. No-op for elements that don't support it."""


# ── Type registry ────────────────────────────────────────────────────────
# Maps a stable "kind" string -> factory(ctx, device, **kwargs) -> DrawingElement.
# Populated by each element module at import time via register_element_type(),
# so a host can discover and construct registered types without hardcoding
# concrete classes. See elements/readme.md for the registration pattern.

ElementFactory = Callable[..., DrawingElement]
ELEMENT_TYPES: dict[str, ElementFactory] = {}


def register_element_type(kind: str, factory: ElementFactory) -> None:
    ELEMENT_TYPES[kind] = factory
