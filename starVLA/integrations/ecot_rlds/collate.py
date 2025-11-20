# SPDX-License-Identifier: MIT
"""
Collate utilities for ECOT RLDS batches.

The adapter keeps batches as lists-of-dicts to remain compatible with the
existing ``QwenGR00T.forward`` signature.
"""

from __future__ import annotations

from typing import List, MutableSequence


def collate_fn_ecot(samples: MutableSequence[dict]) -> List[dict]:
    """
    Identity collate function for ECOT RLDS samples.

    Parameters
    ----------
    samples:
        Sequence of dicts produced by the dataset.

    Returns
    -------
    list[dict]
        The same sequence converted to a list, ensuring downstream code can
        rely on list semantics.
    """

    return list(samples)

