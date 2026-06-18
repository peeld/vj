from __future__ import annotations

from typing import Any


class Prop:
    """Metadata declaration for one PM-managed property.

    Declared as a class variable on a Node subclass.  Metadata-only: no
    __get__/__set__, so instance attribute access is never intercepted.
    register_node() reads Prop instances at registration time to build the
    PropDef objects that PropertyManager stores internally.

    Fields mirror PropDef minus key/section/name, which are derived at
    registration time from the class attribute name and _node_section.
    """

    __slots__ = (
        "label", "type", "default",
        "min_val", "max_val", "step",
        "choices", "widget_hint", "description",
        "attr",
    )

    def __init__(
        self,
        label: str,
        type: type,
        default: Any,
        min_val: float | None = None,
        max_val: float | None = None,
        step: float | None = None,
        choices: list[str] | None = None,
        widget_hint: str | None = None,
        description: str = "",
        attr: str | None = None,
    ) -> None:
        self.label = label
        self.type = type
        self.default = default
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.choices = choices
        self.widget_hint = widget_hint
        self.description = description
        self.attr = attr

    def __repr__(self) -> str:
        return (
            f"Prop(label={self.label!r}, type={self.type.__name__}, "
            f"default={self.default!r})"
        )


class Node:
    """Mixin/base for objects that expose PM-managed properties.

    Declare the PM section with a keyword argument on the class definition:

        class FeedbackParams(Node, section="feedback"):
            ...

    For non-dataclass nodes, Prop instances are declared with the same name
    as the instance attribute they describe.  The class-level Prop is shadowed
    at runtime by the instance attribute set in __init__, so normal attribute
    access is unaffected:

        class CloudElement(DrawingElement, section="cloud"):
            ball_size = Prop("Ball Size", float, 1.0, 0.0, 5.0, 0.1)

            def __init__(self):
                self.ball_size = 1.0   # shadows the class-level Prop

    For @dataclass nodes, declare Prop instances as ClassVar so @dataclass
    ignores them as fields.  The attr name uses a distinct name (e.g. a
    leading underscore) to avoid colliding with the dataclass field:

        @dataclass
        class Params(Node, section="feedback"):
            zoom: float = 1.0
            _zoom_prop: ClassVar[Prop] = Prop("Zoom", float, 1.0, 0.95, 1.05)
    """

    _node_section: str = ""

    def __init_subclass__(cls, section: str = "", **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if section:
            cls._node_section = section

    @classmethod
    def node_props(cls) -> dict[str, "Prop"]:
        """Return {attr_name: Prop} for all Props declared on this class or
        any base (MRO order; subclass declarations shadow base declarations)."""
        props: dict[str, Prop] = {}
        for klass in reversed(cls.__mro__):
            for attr, val in klass.__dict__.items():
                if isinstance(val, Prop):
                    props[attr] = val
        return props
