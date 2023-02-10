
class InferenceDataloader:
    def __init__(self, g, trace_nids, trace_edges, num_requests, batch_size, sampler):
        self._g = g
        self._trace_nids = trace_nids
        self._trace_edges = trace_edges
        self._num_requests = num_requests
        self._batch_size = batch_size
        self._sampler = sampler
        self.requests_sent = 0


    def __iter__(self):
        
        return self
 
    def __next__(self):
        if self.num > self.end:
            raise StopIteration
        else:
            self.num += 1
            return self.num - 1