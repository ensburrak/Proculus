# -*- coding: utf-8 -*-
import time
import functools
import random
from typing import Tuple, Type

def retry(
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    tries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    jitter: float = 0.2
):
    """
    Basit retry decorator'u: exp backoff + jitter.
    """
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for i in range(tries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if i == tries - 1:
                        raise
                    time.sleep(delay + random.uniform(0, jitter))
                    delay = min(delay * 2.0, max_delay)
            # normalde buraya dusmez
        return wrapper
    return deco
