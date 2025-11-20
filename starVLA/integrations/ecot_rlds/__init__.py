# SPDX-License-Identifier: MIT
"""
ECOT RLDS integration package.

This module provides public entry points for building datasets and dataloaders
that wrap Prismatic/ECoT RLDS pipelines without modifying the upstream code.
"""

from .builder import get_vla_dataset_ecot, make_dataloader_ecot

__all__ = ["get_vla_dataset_ecot", "make_dataloader_ecot"]

