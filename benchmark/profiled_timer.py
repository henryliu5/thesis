''' Evaluate the overhead of using a @contextmanager timer vs. class based '''
import time
from contextlib import contextmanager
TRACES = {}

@contextmanager
def Timer(name, track_cuda = False):
    try:
        start_time = time.perf_counter()
        yield
    finally:
        if name not in TRACES:
            TRACES[name] = []
        end_time = time.perf_counter()
        TRACES[name].append(end_time - start_time)
    
class TimerClass:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start_time = time.perf_counter()
    def __exit__(self, type, value, traceback):
        if self.name not in TRACES:
            TRACES[self.name] = []
        end_time = time.perf_counter()
        TRACES[self.name].append(end_time - self.start_time)

if __name__ == '__main__':
    x = 0
    n = 1_000_000
    dec_start = time.perf_counter()
    for i in range(n):
        with Timer('decorated'):
            x += 1
    dec_end = time.perf_counter()

    cls_start = time.perf_counter()
    for i in range(n):
        with TimerClass('class'):
            x += 1
    cls_end = time.perf_counter()

    dec_time = dec_end - dec_start
    cls_time = cls_end - cls_start

    print('decorated', dec_time / n, 's')
    print('class', cls_time / n, 's')