# _____________________________________________________________________________
#
# @file boards.py
# @brief What each supported board can do, beyond its memory sizes
# @version 0.1
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
What each supported board can do, beyond its memory sizes.

Whether a board carries a neural accelerator is a fact about the hardware, not a
choice for a run, so it is declared once in pyproject.toml rather than repeated in
every configuration. A run asks for the accelerator with ``execution.npu``; the
registry decides whether that board has one to ask for.

A board absent from the registry is treated as having no accelerator, so an unlisted
board runs on its CPU rather than failing.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import tomllib
from pydantic import BaseModel, Field, model_validator


class Accelerator(str, Enum):
    """Neural accelerators the pipeline knows how to target."""

    NONE = "none"
    NEUTRON = "neutron"  # NXP eIQ Neutron NPU


class BoardCapabilities(BaseModel):
    """What one board offers."""

    accelerator: Accelerator = Field(
        default=Accelerator.NONE, description="Neural accelerator fitted to this board"
    )
    neutron_target: str | None = Field(
        default=None,
        description="Target name passed to neutron_compiler --target, e.g. mcxn94x",
    )

    @model_validator(mode="after")
    def validate_accelerator_details(self) -> "BoardCapabilities":
        """A Neutron board must say which target the converter should build for."""
        if self.accelerator is Accelerator.NEUTRON and not self.neutron_target:
            raise ValueError("A board with accelerator 'neutron' must set neutron_target")
        return self

    @property
    def has_npu(self) -> bool:
        """True when the board carries an accelerator the pipeline can use."""
        return self.accelerator is not Accelerator.NONE


class BoardRegistry(BaseModel):
    """Capabilities of every board declared in pyproject.toml."""

    boards: dict[str, BoardCapabilities] = Field(default_factory=dict)

    @classmethod
    def discover(cls, start: Path | None = None) -> "BoardRegistry":
        """Search upward from ``start`` for the pyproject.toml declaring the boards."""
        current = (start or Path.cwd()).resolve()

        for candidate in [current, *current.parents]:
            pyproject_path = candidate / "pyproject.toml"
            if not pyproject_path.exists():
                continue

            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)

            table = data.get("tool", {}).get("zephyr-ml-forge", {}).get("boards")
            if table is not None:
                return cls(
                    boards={
                        name: BoardCapabilities(**entry) for name, entry in table.items()
                    }
                )

        return cls()

    def for_board(self, board: str) -> BoardCapabilities:
        """Capabilities of a board named as Zephyr names it.

        Zephyr 4.x board identifiers carry qualifiers - 'frdm_mcxn947/mcxn947/cpu0' -
        and the registry is keyed by the board alone, so that a change of core or
        revision does not silently lose the accelerator.
        """
        return self.boards.get(board.split("/")[0], BoardCapabilities())

    def boards_with_npu(self) -> list[str]:
        """Names of the declared boards that carry an accelerator."""
        return sorted(name for name, board in self.boards.items() if board.has_npu)
