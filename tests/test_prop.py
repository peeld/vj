"""
test_prop.py — Unit tests for prop.py (Phase 1b) and PM integration (Phase 2b).

Run with:
    python -m pytest test_prop.py -v
or:
    python test_prop.py
"""
import dataclasses
import unittest
from typing import ClassVar

from prop import Node, Prop
from property_manager import PropertyManager


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level classes used in dataclass co-existence tests
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _DataParams(Node, section="feedback"):
    zoom: float = 1.0
    decay: float = 0.993
    _zoom_prop: ClassVar[Prop] = Prop("Zoom", float, 1.0, 0.95, 1.05, 0.001)
    _decay_prop: ClassVar[Prop] = Prop("Decay", float, 0.993, 0.80, 0.999, 0.010)


# ─────────────────────────────────────────────────────────────────────────────
#  Prop
# ─────────────────────────────────────────────────────────────────────────────

class TestProp(unittest.TestCase):

    def test_required_fields(self):
        p = Prop("Decay", float, 0.993)
        self.assertEqual(p.label, "Decay")
        self.assertIs(p.type, float)
        self.assertEqual(p.default, 0.993)

    def test_optional_fields_default_to_none_or_empty(self):
        p = Prop("Flag", bool, True)
        self.assertIsNone(p.min_val)
        self.assertIsNone(p.max_val)
        self.assertIsNone(p.step)
        self.assertIsNone(p.choices)
        self.assertIsNone(p.widget_hint)
        self.assertEqual(p.description, "")

    def test_optional_fields_set(self):
        p = Prop("Speed", float, 1.0,
                 min_val=0.0, max_val=10.0, step=0.1,
                 widget_hint="slider", description="Move speed")
        self.assertEqual(p.min_val, 0.0)
        self.assertEqual(p.max_val, 10.0)
        self.assertEqual(p.step, 0.1)
        self.assertEqual(p.widget_hint, "slider")
        self.assertEqual(p.description, "Move speed")

    def test_choices_field(self):
        p = Prop("Mode", str, "add", choices=["add", "blend", "sub"])
        self.assertEqual(p.choices, ["add", "blend", "sub"])

    def test_repr_contains_key_info(self):
        p = Prop("Decay", float, 0.993)
        r = repr(p)
        self.assertIn("Decay", r)
        self.assertIn("float", r)
        self.assertIn("0.993", r)

    def test_metadata_only_does_not_intercept_instance_attr(self):
        """Instance attribute set in __init__ shadows the class-level Prop."""
        class Obj:
            speed = Prop("Speed", float, 1.0)
            def __init__(self):
                self.speed = 5.0

        obj = Obj()
        self.assertEqual(obj.speed, 5.0)
        self.assertIsInstance(type(obj).__dict__["speed"], Prop)

    def test_slots_prevents_arbitrary_attributes(self):
        p = Prop("X", float, 0.0)
        with self.assertRaises(AttributeError):
            p.nonexistent = 99


# ─────────────────────────────────────────────────────────────────────────────
#  Node — section declaration
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeSection(unittest.TestCase):

    def test_section_kwarg_stored(self):
        class MyNode(Node, section="feedback"):
            pass
        self.assertEqual(MyNode._node_section, "feedback")

    def test_section_default_empty(self):
        class MyNode(Node):
            pass
        self.assertEqual(MyNode._node_section, "")

    def test_sibling_sections_independent(self):
        class A(Node, section="aaa"):
            pass
        class B(Node, section="bbb"):
            pass
        self.assertEqual(A._node_section, "aaa")
        self.assertEqual(B._node_section, "bbb")

    def test_subclass_inherits_section_when_not_declared(self):
        class Base(Node, section="base"):
            pass
        class Child(Base):
            pass
        self.assertEqual(Child._node_section, "base")

    def test_subclass_overrides_section(self):
        class Base(Node, section="base"):
            pass
        class Child(Base, section="child"):
            pass
        self.assertEqual(Child._node_section, "child")
        self.assertEqual(Base._node_section, "base")


# ─────────────────────────────────────────────────────────────────────────────
#  Node — node_props()
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeProps(unittest.TestCase):

    def test_no_props_returns_empty(self):
        class MyNode(Node, section="x"):
            pass
        self.assertEqual(MyNode.node_props(), {})

    def test_props_collected_by_attr_name(self):
        class MyNode(Node, section="x"):
            speed = Prop("Speed", float, 1.0)
            visible = Prop("Visible", bool, True)

        props = MyNode.node_props()
        self.assertIn("speed", props)
        self.assertIn("visible", props)
        self.assertIsInstance(props["speed"], Prop)

    def test_non_prop_class_vars_excluded(self):
        class MyNode(Node, section="x"):
            speed = Prop("Speed", float, 1.0)
            label = "not a prop"
            count = 42

        self.assertEqual(set(MyNode.node_props()), {"speed"})

    def test_inherited_props_visible(self):
        class Base(Node, section="base"):
            visible = Prop("Visible", bool, True)
        class Child(Base, section="child"):
            speed = Prop("Speed", float, 1.0)

        props = Child.node_props()
        self.assertIn("visible", props)
        self.assertIn("speed", props)

    def test_child_prop_shadows_base_prop(self):
        class Base(Node, section="base"):
            visible = Prop("Visible", bool, True)
        class Child(Base, section="child"):
            visible = Prop("Shown", bool, False)

        self.assertEqual(Child.node_props()["visible"].label, "Shown")
        self.assertEqual(Base.node_props()["visible"].label, "Visible")

    def test_callable_on_instance(self):
        """node_props() is a classmethod; instance state doesn't affect it."""
        class MyNode(Node, section="x"):
            speed = Prop("Speed", float, 1.0)
            def __init__(self):
                self.speed = 99.0

        obj = MyNode()
        props = obj.node_props()
        self.assertIn("speed", props)
        self.assertIsInstance(props["speed"], Prop)
        self.assertEqual(obj.speed, 99.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Node + @dataclass co-existence
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeDataclassCoexistence(unittest.TestCase):

    def test_dataclass_init_works(self):
        p = _DataParams()
        self.assertEqual(p.zoom, 1.0)
        self.assertEqual(p.decay, 0.993)

    def test_dataclass_field_mutation_works(self):
        p = _DataParams()
        p.zoom = 1.005
        self.assertEqual(p.zoom, 1.005)

    def test_dataclass_fields_unaffected_by_node(self):
        fields = {f.name for f in dataclasses.fields(_DataParams)}
        self.assertEqual(fields, {"zoom", "decay"})

    def test_node_props_finds_classvar_props(self):
        props = _DataParams.node_props()
        self.assertIn("_zoom_prop", props)
        self.assertIn("_decay_prop", props)

    def test_classvar_props_not_in_dataclass_fields(self):
        field_names = {f.name for f in dataclasses.fields(_DataParams)}
        self.assertNotIn("_zoom_prop", field_names)
        self.assertNotIn("_decay_prop", field_names)

    def test_section_correct(self):
        self.assertEqual(_DataParams._node_section, "feedback")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2 — PropertyManager.register_node / pre_register_node_class /
#             unregister_node
# ─────────────────────────────────────────────────────────────────────────────

# Fixtures used across Phase 2 tests.

class _PlainNode(Node, section="plain"):
    speed  = Prop("Speed",   float, 1.0, 0.0, 10.0, 0.1)
    active = Prop("Active",  bool,  True)


class _MultiPropNode(Node, section="multi"):
    alpha = Prop("Alpha", float, 0.5, 0.0, 1.0)
    count = Prop("Count", int,   3,   1,   10)


@dataclasses.dataclass
class _DCNode(Node, section="dc"):
    zoom:  float = 1.0
    decay: float = 0.993
    _zoom_prop:  ClassVar[Prop] = Prop("Zoom",  float, 1.0,   0.95, 1.05, 0.001, attr="zoom")
    _decay_prop: ClassVar[Prop] = Prop("Decay", float, 0.993, 0.80, 0.999, 0.010, attr="decay")


class TestRegisterNode(unittest.TestCase):

    def _pm(self):
        return PropertyManager()

    # ── basic registration ────────────────────────────────────────────────────

    def test_register_node_creates_keys(self):
        pm = self._pm()
        node = _PlainNode()
        node.speed = 3.0
        pm.register_node(node)
        self.assertIn("plain.speed",  pm._defs)
        self.assertIn("plain.active", pm._defs)

    def test_register_node_get_reads_instance(self):
        pm = self._pm()
        node = _PlainNode()
        node.speed = 7.5
        pm.register_node(node)
        self.assertEqual(pm.get("plain.speed"), 7.5)

    def test_register_node_set_writes_instance(self):
        pm = self._pm()
        node = _PlainNode()
        node.speed = 1.0
        pm.register_node(node)
        pm.set("plain.speed", 4.0)
        self.assertEqual(node.speed, 4.0)

    def test_register_node_multiple_props(self):
        pm = self._pm()
        node = _MultiPropNode()
        node.alpha = 0.8
        node.count = 5
        pm.register_node(node)
        self.assertAlmostEqual(pm.get("multi.alpha"), 0.8)
        self.assertEqual(pm.get("multi.count"), 5)

    def test_register_node_returns_node(self):
        pm = self._pm()
        node = _PlainNode()
        result = pm.register_node(node)
        self.assertIs(result, node)

    def test_register_node_one_liner(self):
        pm = self._pm()
        node = pm.register_node(_PlainNode())
        pm.set("plain.speed", 2.0)
        self.assertEqual(node.speed, 2.0)

    def test_register_node_no_section_raises(self):
        class _Bare(Node):
            x = Prop("X", float, 0.0)
        pm = self._pm()
        with self.assertRaises(ValueError):
            pm.register_node(_Bare())

    # ── idempotency ───────────────────────────────────────────────────────────

    def test_register_node_idempotent_second_call(self):
        pm = self._pm()
        node = _PlainNode()
        pm.register_node(node)
        pm.register_node(node)  # must not raise
        self.assertEqual(len([k for k in pm._defs if k.startswith("plain.")]), 2)

    # ── @dataclass co-existence with attr= ───────────────────────────────────

    def test_register_node_dataclass_keys_use_field_name(self):
        pm = self._pm()
        node = _DCNode()
        pm.register_node(node)
        self.assertIn("dc.zoom",  pm._defs)
        self.assertIn("dc.decay", pm._defs)
        self.assertNotIn("dc._zoom_prop",  pm._defs)
        self.assertNotIn("dc._decay_prop", pm._defs)

    def test_register_node_dataclass_get_reads_field(self):
        pm = self._pm()
        node = _DCNode(zoom=1.02, decay=0.99)
        pm.register_node(node)
        self.assertAlmostEqual(pm.get("dc.zoom"),  1.02)
        self.assertAlmostEqual(pm.get("dc.decay"), 0.99)

    def test_register_node_dataclass_set_writes_field(self):
        pm = self._pm()
        node = _DCNode()
        pm.register_node(node)
        pm.set("dc.zoom", 1.01)
        self.assertAlmostEqual(node.zoom, 1.01)


class TestPreRegisterNodeClass(unittest.TestCase):

    def _pm(self):
        return PropertyManager()

    def test_pre_register_creates_keys_with_defaults(self):
        pm = self._pm()
        pm.pre_register_node_class(_PlainNode, "plain")
        self.assertIn("plain.speed",  pm._defs)
        self.assertIn("plain.active", pm._defs)
        self.assertEqual(pm.get("plain.speed"), 1.0)
        self.assertEqual(pm.get("plain.active"), True)

    def test_pre_register_no_binding(self):
        pm = self._pm()
        pm.pre_register_node_class(_PlainNode, "plain")
        self.assertNotIn("plain.speed",  pm._bindings)
        self.assertNotIn("plain.active", pm._bindings)

    def test_pre_register_idempotent(self):
        pm = self._pm()
        pm.pre_register_node_class(_PlainNode, "plain")
        pm.pre_register_node_class(_PlainNode, "plain")  # must not raise
        self.assertEqual(len([k for k in pm._defs if k.startswith("plain.")]), 2)

    def test_pre_register_then_register_node_upgrades_binding(self):
        """register_node on a pre-registered key upgrades to live binding.
        The stored default latches onto the instance (PM is source of truth)."""
        pm = self._pm()
        pm.pre_register_node_class(_PlainNode, "plain")
        node = _PlainNode()
        node.speed = 5.0  # instance initial — overwritten by default latch
        pm.register_node(node)
        # stored default (1.0) latched onto instance; key is now bound
        self.assertEqual(pm.get("plain.speed"), 1.0)
        self.assertEqual(node.speed, 1.0)
        self.assertIn("plain.speed", pm._bindings)

    def test_pre_register_set_then_register_latches_value(self):
        """Value written before the instance exists is latched onto it on bind."""
        pm = self._pm()
        pm.pre_register_node_class(_PlainNode, "plain")
        pm.set("plain.speed", 9.0)
        node = _PlainNode()
        node.speed = 1.0  # initial instance value (will be overwritten by latch)
        pm.register_node(node)
        self.assertEqual(node.speed, 9.0)
        self.assertEqual(pm.get("plain.speed"), 9.0)

    def test_pre_register_different_section_same_class(self):
        """pre_register_node_class can stamp the same Prop schema under any section."""
        pm = self._pm()
        pm.pre_register_node_class(_PlainNode, "cloud")
        pm.pre_register_node_class(_PlainNode, "lasers")
        self.assertIn("cloud.speed",   pm._defs)
        self.assertIn("lasers.speed",  pm._defs)


class TestUnregisterNode(unittest.TestCase):

    def _pm(self):
        return PropertyManager()

    def test_unregister_unbinds_all_keys(self):
        pm = self._pm()
        node = _PlainNode()
        pm.register_node(node)
        pm.unregister_node(node)
        self.assertNotIn("plain.speed",  pm._bindings)
        self.assertNotIn("plain.active", pm._bindings)

    def test_unregister_preserves_values(self):
        pm = self._pm()
        node = _PlainNode()
        node.speed = 6.0
        pm.register_node(node)
        pm.unregister_node(node)
        # key still registered, value preserved
        self.assertIn("plain.speed", pm._defs)
        self.assertEqual(pm.get("plain.speed"), 6.0)

    def test_unregister_then_register_node_rebinds(self):
        pm = self._pm()
        node = _PlainNode()
        node.speed = 6.0
        pm.register_node(node)
        pm.unregister_node(node)

        node2 = _PlainNode()
        node2.speed = 1.0
        pm.register_node(node2)
        # preserved value (6.0) latched onto new instance
        self.assertEqual(node2.speed, 6.0)
        self.assertEqual(pm.get("plain.speed"), 6.0)

    def test_unregister_noop_on_unbound_key(self):
        """unregister_node is safe even when a key was never bound."""
        pm = self._pm()
        pm.pre_register_node_class(_PlainNode, "plain")
        node = _PlainNode()
        node._node_section = "plain"  # section matches pre-registered keys
        pm.unregister_node(node)  # nothing bound yet — must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
