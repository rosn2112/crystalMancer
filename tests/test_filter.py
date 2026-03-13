"""Unit tests for CIF filtering logic."""

import pytest
from unittest.mock import MagicMock, patch

from crystalmancer.cif.filter import is_perovskite_spacegroup, is_perovskite_composition


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_structure_mock(sg_number: int, composition_dict: dict[str, float]):
    """Create a mock pymatgen Structure with the given space group and composition."""
    structure = MagicMock()

    # Mock composition
    comp = MagicMock()
    comp.elements = [MagicMock(__str__=MagicMock(return_value=el)) for el in composition_dict]
    comp.get_el_amt_dict.return_value = composition_dict
    comp.reduced_formula = "".join(
        f"{el}{int(amt) if amt != 1 else ''}"
        for el, amt in composition_dict.items()
    )
    structure.composition = comp

    return structure, sg_number


# ── Space Group Tests ─────────────────────────────────────────────────────────

class TestIsPerovskiteSpacegroup:
    def test_cubic_pm3m(self):
        """Pm-3m (#221) — ideal cubic perovskite."""
        structure = MagicMock()
        with patch("crystalmancer.cif.filter.SpacegroupAnalyzer") as MockSGA:
            MockSGA.return_value.get_space_group_number.return_value = 221
            assert is_perovskite_spacegroup(structure) is True

    def test_rhombohedral_r3c(self):
        """R-3c (#167) — LaAlO₃ / LaCoO₃ type."""
        structure = MagicMock()
        with patch("crystalmancer.cif.filter.SpacegroupAnalyzer") as MockSGA:
            MockSGA.return_value.get_space_group_number.return_value = 167
            assert is_perovskite_spacegroup(structure) is True

    def test_orthorhombic_pbnm(self):
        """Pbnm (#62) — GdFeO₃ type distortion."""
        structure = MagicMock()
        with patch("crystalmancer.cif.filter.SpacegroupAnalyzer") as MockSGA:
            MockSGA.return_value.get_space_group_number.return_value = 62
            assert is_perovskite_spacegroup(structure) is True

    def test_non_perovskite_fd3m(self):
        """Fd-3m (#227) — spinel, not perovskite."""
        structure = MagicMock()
        with patch("crystalmancer.cif.filter.SpacegroupAnalyzer") as MockSGA:
            MockSGA.return_value.get_space_group_number.return_value = 227
            assert is_perovskite_spacegroup(structure) is False

    def test_analyzer_exception(self):
        """Gracefully handle pymatgen errors."""
        structure = MagicMock()
        with patch("crystalmancer.cif.filter.SpacegroupAnalyzer") as MockSGA:
            MockSGA.side_effect = RuntimeError("bad CIF")
            assert is_perovskite_spacegroup(structure) is False


# ── Composition Tests ─────────────────────────────────────────────────────────

class TestIsPerovskiteComposition:
    def test_srtio3(self):
        """SrTiO₃ — classic perovskite."""
        struct, _ = _make_structure_mock(221, {"Sr": 1, "Ti": 1, "O": 3})
        assert is_perovskite_composition(struct) is True

    def test_lacoo3(self):
        """LaCoO₃ — perovskite."""
        struct, _ = _make_structure_mock(167, {"La": 1, "Co": 1, "O": 3})
        assert is_perovskite_composition(struct) is True

    def test_sio2_rejected(self):
        """SiO₂ — no A/B site elements → rejected."""
        struct, _ = _make_structure_mock(0, {"Si": 1, "O": 2})
        assert is_perovskite_composition(struct) is False

    def test_no_oxygen_rejected(self):
        """LaCo — no oxygen → rejected."""
        struct, _ = _make_structure_mock(0, {"La": 1, "Co": 1})
        assert is_perovskite_composition(struct) is False

    def test_batio3(self):
        """BaTiO₃ — perovskite."""
        struct, _ = _make_structure_mock(221, {"Ba": 1, "Ti": 1, "O": 3})
        assert is_perovskite_composition(struct) is True
