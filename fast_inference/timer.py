from contextlib import contextmanager
import time

TRACE_ENABLED = False
TRACES = {}
def enable_timers():
    global TRACE_ENABLED
    TRACE_ENABLED = True

@contextmanager
def Timer(name):
    global TRACE_ENABLED
    try:
        if TRACE_ENABLED:
            start_time = time.time()
        yield
    finally:
        if TRACE_ENABLED:
            end_time = time.time()
            if name not in TRACES:
                TRACES[name] = []
            TRACES[name].append(end_time - start_time)

def print_timer_info():
    print('Timing summary: ')
    for k, v in TRACES.items():
        print('\tTotal', k, 'time:', sum(v))
        print('\tAvg', k, 'time:', sum(v) / len(v))