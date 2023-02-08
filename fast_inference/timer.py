from contextlib import contextmanager
import time
import torch

TRACE_ENABLED = False
TRACES = {}
EVENT_CLOSURES = {}
def enable_timers():
    global TRACE_ENABLED
    TRACE_ENABLED = True

@contextmanager
def Timer(name, track_cuda = False):
    global TRACE_ENABLED
    try:
        if TRACE_ENABLED:
            if track_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                start_time = time.time()
        yield
    finally:
        if TRACE_ENABLED:
            if name not in TRACES:
                TRACES[name] = []
            if track_cuda:
                end.record()

                def event_end_closure():
                    # Use a closure here that will be called later record event times after sync
                    TRACES[name].append(start.elapsed_time(end) / 1000.0)

                if name not in EVENT_CLOSURES:
                    EVENT_CLOSURES[name] = []
                EVENT_CLOSURES[name].append(event_end_closure)
            else:
                end_time = time.time()
                TRACES[name].append(end_time - start_time)

def print_timer_info():
    ''' Print global timing info collected through Timer 

    WARNING: Performs CUDA syncrhonization for accurate events
    '''
    torch.cuda.synchronize()
    global EVENT_CLOSURES
    for v in EVENT_CLOSURES.values():
        for f in v:
            f()

    EVENT_CLOSURES = {}
    print('Timing summary: ')
    for k, v in TRACES.items():
        print('\tTotal', k, 'time:', round(sum(v), 4))
        print('\t\tAvg:', round(sum(v) / len(v), 6))