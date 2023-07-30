"""

Author: Sa1g
Github: https://github.com/sa1g

Date: 2023.07.30

"""
import time
import logging
from functools import wraps


def exec_time(func):  # pylint: disable = missing-function-docstring
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logging.debug(f"Function %s Took %f seconds", func.__name__, total_time)
        # print(f"Function {func.__name__} Took {total_time} seconds")
        return result

    return timeit_wrapper
