''' Experiment measure performance of multiprocessing.Queue vs Boost C++ shared memory queue'''
import torch
from torch.multiprocessing import Process, Queue, Barrier
from dataclasses import dataclass
from typing import Tuple, Dict
import time
import faster_fifo

import io
from multiprocessing.reduction import ForkingPickler
import pickle
class ConnectionWrapper:
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler to
    serialize objects"""

    def __init__(self, conn):
        self.conn = conn

    def send(self, obj):
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        if 'conn' in self.__dict__:
            return getattr(self.conn, name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, 'conn'))


class TorchQueue:

    def __init__(self):
        self.q = faster_fifo.Queue(1105 * 5)
        
    def put(self, obj):
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        v = buf.getvalue()
        self.q.put(v, block=True, timeout=10)

    def get(self):
        buf = self.q.get(block=True, timeout=10)
        return pickle.loads(buf)


@dataclass
class IPCMessage:
    id: int
    data: torch.Tensor
    mfg_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    timing_info: Dict[str, float]

N = 10000

class Producer(Process):
    def __init__(self, queue: Queue, barrier: Barrier):
        super().__init__()
        self.queue = queue
        self.barrier = barrier

    def run(self):
        t = torch.randn(1000, 1000, device=torch.device('cuda', 0))
        t1 = torch.randn(1000, 1000, device=torch.device('cuda', 0))
        t2 = torch.randn(1000, 1000, device=torch.device('cuda', 0))
        t3 = torch.randn(1000, 1000, device=torch.device('cuda', 0))
        # t = 0
        # t1 = 1
        # t2 = 2
        # t3 = 3
        for i in range(50):
            self.queue.put(IPCMessage(i, t, (t1, t2, t3), {'start': time.perf_counter()}))

        time.sleep(5)

        for i in range(N):
            self.queue.put(IPCMessage(i, t, (t1, t2, t3), {'start': time.perf_counter()}))

        self.barrier.wait()
        

class Consumer(Process):
    def __init__(self, queue: Queue, barrier: Barrier):
        super().__init__()
        self.queue = queue
        self.barrier = barrier

    def run(self):
        sum = 0
        for i in range(50):
            msg = self.queue.get()

        for i in range(N):
            msg = self.queue.get()
            sum += time.perf_counter() - msg.timing_info['start']

        self.barrier.wait()
        print(sum / N, 's')

def main():
    torch.multiprocessing.set_start_method('spawn')
    torch.cuda.init()
    # Create cuda tensor on device 0 and pass to queue
    # q = TorchQueue()
    # q = faster_fifo.Queue()
    q = torch.multiprocessing.Queue(1000)
    b = Barrier(2)
    p = Producer(q, b)
    c = Consumer(q, b)

    p.start()
    c.start()
    p.join()
    c.join()

if __name__ == '__main__':
    main()