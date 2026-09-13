"""Board capabilities, as declared in pyproject.toml."""

from __future__ import annotations

import pytest

from pipeline.core.boards import Accelerator, BoardCapabilities, BoardRegistry

REGISTRY_TOML = """
[tool.zephyr-ml-forge.boards.frdm_mcxn947]
accelerator = "neutron"
neutron_target = "mcxn94x"

[tool.zephyr-ml-forge.boards.qemu_cortex_m3]
accelerator = "none"
"""


@pytest.fixture
def registry(tmp_path):
    """A registry read from a pyproject.toml in a temporary directory."""
    (tmp_path / "pyproject.toml").write_text(REGISTRY_TOML)
    return BoardRegistry.discover(tmp_path)


def test_a_board_with_an_accelerator_declares_its_target(registry):
    board = registry.for_board("frdm_mcxn947")

    assert board.has_npu is True
    assert board.accelerator is Accelerator.NEUTRON
    assert board.neutron_target == "mcxn94x"


def test_zephyr_board_qualifiers_do_not_lose_the_accelerator(registry):
    # Zephyr 4.x names this board 'frdm_mcxn947/mcxn947/cpu0'.
    assert registry.for_board("frdm_mcxn947/mcxn947/cpu0").has_npu is True


def test_a_board_declared_without_one_has_none(registry):
    assert registry.for_board("qemu_cortex_m3").has_npu is False


def test_an_undeclared_board_is_assumed_to_have_none(registry):
    # Safe default: an unknown board runs on its CPU rather than failing.
    assert registry.for_board("nucleo_f446re").has_npu is False


def test_boards_with_an_accelerator_can_be_listed(registry):
    assert registry.boards_with_npu() == ["frdm_mcxn947"]


def test_a_neutron_board_must_name_its_compiler_target():
    with pytest.raises(ValueError, match="neutron_target"):
        BoardCapabilities(accelerator="neutron")


def test_a_registry_is_empty_when_no_pyproject_declares_boards(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "unrelated"\n')

    assert BoardRegistry.discover(tmp_path).boards == {}


def test_the_projects_own_board_is_declared():
    # The board this project targets must be in the shipped registry.
    assert BoardRegistry.discover().for_board("frdm_mcxn947/mcxn947/cpu0").has_npu is True
