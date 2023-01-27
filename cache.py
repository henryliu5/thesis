import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache
import pickle
from tqdm import tqdm
import sys
import queue

# ten_gb_mem = 312_500_000
# four_gb_mem = 125_000_000
two_gb_mem = 500_000_000 # how many 32-bit values can be held

features = 100
CACHE_SIZE = two_gb_mem // features

CACHE_SIZE = int(CACHE_SIZE * float(sys.argv[1]))

total_evictions = 0

class LRUPolicy:
    def __init__(self, n, preload):
        ...
        # for layer_0 in tqdm(preload):
        #     for x in layer_0:
        #         self._try_get(x.item())

    def try_get(self, x):
        info = self._try_get.cache_info()
        self._try_get(x)
        post_info = self._try_get.cache_info()
        # Return true if cache hit
        return info.hits != post_info.hits

    # Dummy function to do inspection on later
    @lru_cache(maxsize=CACHE_SIZE)
    def _try_get(self, x):
        return x

class FIFOPolicy:
    def __init__(self, n, preload):
        self.n = n
        self.s = set()
        self.q = queue.SimpleQueue()

    def try_get(self, x):
        global total_evictions
        if x in self.s:
            return True

        if self.q.qsize() == self.n:
            evicted = self.q.get()  
            self.s.remove(evicted)
            total_evictions += 1

        self.q.put(x)
        self.s.add(x)

        return False

class StaticPolicy:
    def __init__(self, n, graph_name):
        self.n = n
        # TODO fix
        sort_nid = np.load(f'{graph_name}_ordered.npz')['arr_0']
        cache_nid = sort_nid[:n]
        self.s = set(cache_nid)

    def try_get(self, x):
        global total_evictions
        if x in self.s:
            return True

        return False

class MyPolicy:
    def __init__(self, n, preload):
        self.n = n
        self.s = set()
        self.list = []
        self.count = {}

        self.accesses_since_update = 0

    def update_with_lookahead(self, new_nodes):
        for layer_0 in new_nodes:
            for x in layer_0:
                x = x.item()
                if x not in self.count:
                    self.count[x] = 0
                self.count[x] += 1


    def try_get(self, x):
        # sort by own count
        if self.accesses_since_update > 100000:
            self.list.sort(key=lambda x: -self.count[x])
            self.accesses_since_update = 0
        self.accesses_since_update += 1

        global total_evictions

        if x in self.s:
            return True

        if len(self.list) == self.n:
            evicted = self.list.pop()
            self.s.remove(evicted)
            total_evictions += 1

        self.list.append(x)
        self.s.add(x)

        return False


def simulate(graph_name, batch_size, policy):
    print('starting simulation with policy', policy)
    global total_evictions
    total_evictions = 0
    infile = open(f'{graph_name}_{batch_size}', 'rb')
    nodes = pickle.load(infile)
    infile.close()

    # each element is "time" cache miss occurs, # of misses is length
    cache_misses = []
    num_cache_misses = 0
    cache_total = 0

    if graph_name == 'reddit':
        preload = nodes[:100]
    else:    
        preload = nodes[:500]
    
    if policy == 'mine':
        cache = MyPolicy(CACHE_SIZE, preload=[])
        cache.update_with_lookahead(preload)
    elif policy == 'fifo':
        cache = FIFOPolicy(CACHE_SIZE, preload=[])
    elif policy == 'lru':
        cache = LRUPolicy(CACHE_SIZE, preload=[])
    elif policy == 'static':
        cache = StaticPolicy(CACHE_SIZE, graph_name)

    # Warm caches with some queries
    for layer_0 in tqdm(preload):
        for x in layer_0:
            x = x.item()
            hit = cache.try_get(x)

    print('preload done')
    time = 0
    if graph_name == 'reddit':
        nodes = nodes[100:250]
    else:    
        nodes = nodes[500:1500]

    # naive
    for i, layer_0 in enumerate(tqdm(nodes)):
        if policy == '***disabled':
            lookahead = 5
            if i % lookahead == 0:
                cache.update_with_lookahead(nodes[i:i+lookahead])

                scores = {}
                # Compute the 'score' of each i..i+lookahead query
                for offset in range(lookahead):
                    key = i + offset
                    if key not in scores:
                        scores[key] = 0

                    for x in nodes[key]:
                        x = x.item()
                        scores[key] += cache.count[x]

                execution_order = [i + x for x in range(lookahead)]
                execution_order.sort(key=lambda x: -scores[x])

                for index in execution_order:
                    for x in nodes[index]:
                        x = x.item()
                        hit = cache.try_get(x)
                        if not hit:
                            # cache_misses.append(time)
                            num_cache_misses += 1
                        time += 1
                        cache_total += 1
        else:
            if policy == 'mine':
                lookahead = 100
                if i % lookahead == 0:
                    cache.update_with_lookahead(nodes[i:i+lookahead])

            for x in layer_0:
                x = x.item()
                hit = cache.try_get(x)
                if not hit:
                    # cache_misses.append(time)
                    num_cache_misses += 1
                time += 1
                cache_total += 1

    print('cache misses:', num_cache_misses, 'total accesses:', cache_total)
    hit_ratio = 1 - (num_cache_misses / cache_total)
    print('hit ratio:', hit_ratio)
    print('total evictions', total_evictions)

    f = open(f"results/{policy}_hit_ratio_cache_{sys.argv[1]}_{sys.argv[2]}.txt", "w")
    f.write(str(hit_ratio))
    f.close()

    # df = pd.DataFrame(cache_misses, columns=['cache misses over time'])
    # df = df.sample(n=5000, replace=False, random_state=1)

    # my_plot = sns.displot(df)
    # my_plot.savefig(f'{graph_name}_{batch_size}_{policy}.png')
    # plt.show()

if __name__ == '__main__':
    simulate(str(sys.argv[2]), 1, 'fifo')
    simulate(str(sys.argv[2]), 1, 'mine')
    simulate(str(sys.argv[2]), 1, 'lru')
    simulate(str(sys.argv[2]), 1, 'static')

