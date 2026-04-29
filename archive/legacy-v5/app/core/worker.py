"""
Thread helpers for fire-and-forget background work.
"""

import threading
from typing import Callable


def run_in_thread(target: Callable, *args, daemon: bool = True, **kwargs) -> threading.Thread:
    """
    Run *target* in a daemon thread, passing *args* and **kwargs*.
    Returns the thread (already started).

    Using daemon=True means threads are automatically killed when the
    main process exits — appropriate for background I/O tasks.
    """
    t = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=daemon)
    t.start()
    return t
