from contextlib import contextmanager
import time
import torch
import pandas as pd
from typing import Dict, Any
import os
from pathlib import Path
from torch.profiler import record_function

TRACE_ENABLED = False
TRACES = {}
EVENT_CLOSURES = {}
def enable_timers():
    global TRACE_ENABLED
    TRACE_ENABLED = True

@contextmanager
def Timer(name, track_cuda = True):
    global TRACE_ENABLED
    if TRACE_ENABLED:
        with record_function(name):
            try:
                if track_cuda:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                else:
                    start_time = time.perf_counter()
                yield
            finally:
                if name not in TRACES:
                    TRACES[name] = []
                if track_cuda:
                    end.record()
                    # torch.cuda.synchronize()
                    # TRACES[name].append(start.elapsed_time(end) / 1000.0)
                    def event_end_closure():
                        # Use a closure here that will be called later record event times after sync
                        TRACES[name].append(start.elapsed_time(end) / 1000.0)

                    if name not in EVENT_CLOSURES:
                        EVENT_CLOSURES[name] = []
                    EVENT_CLOSURES[name].append(event_end_closure)
                else:
                    end_time = time.perf_counter()
                    TRACES[name].append(end_time - start_time)
    else:
        yield

def clear_timers():
    global TRACES
    TRACES = {}

def sync_timers():
    ''' Synchronize and capture cuda event timers '''
    torch.cuda.synchronize()
    global EVENT_CLOSURES
    for v in EVENT_CLOSURES.values():
        for f in v:
            f()
    EVENT_CLOSURES = {}

def print_timer_info(ignore_first_n = 2):
    ''' Print global timing info collected through Timer 

    WARNING: Performs CUDA syncrhonization for accurate events
    '''
    sync_timers()

    print(f'Timing summary (trials[{ignore_first_n}:end]): ')
    for k, v in TRACES.items():
        v = v[ignore_first_n:]
        print('\tTotal', k, 'time:', round(sum(v), 4))
        print('\t\tAvg:', round(sum(v) / len(v), 6))

def export_timer_info(path, current_config: Dict[str, Any], ignore_first_n = 2):
    ''' Write data to CSV 
    
        WARNING: Performs CUDA syncrhonization for accurate events
    '''
    sync_timers()
    export_dict_as_pd(TRACES, path, current_config, ignore_first_n)

def export_dict_as_pd(export_dict, path, current_config: Dict[str, Any], ignore_first_n = 2):
    df = pd.DataFrame.from_dict(export_dict)

    # Drop first n: https://sparkbyexamples.com/pandas/pandas-drop-first-n-rows-from-dataframe/
    df = df.tail(-ignore_first_n)

    # Add keys as new columns, v is constant value
    for k, v in current_config.items():
        df[k] = v
    
    file_name = str([v for v in current_config.values()])
    file_name = file_name.strip()
    file_name = file_name.strip('[')
    file_name = file_name.strip(']')
    file_name = file_name.replace(', ', '-')
    file_name = file_name.replace('\'', '')
    file_name = file_name + '.csv'
    
    Path(path).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(path, file_name))