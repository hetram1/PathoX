try:
    from .pathox_native import (
        max_threads,
        score_tiles,
        score_tissue_tiles,
    )
except ImportError:
    max_threads = None
    score_tiles = None
    score_tissue_tiles = None

__all__ = [
    "score_tiles",
    "score_tissue_tiles",
    "max_threads",
]
