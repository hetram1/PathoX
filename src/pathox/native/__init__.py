try:
    from .pathox_native import max_threads, score_tiles
except ImportError:
    max_threads = None
    score_tiles = None

__all__ = ["score_tiles", "max_threads"]
