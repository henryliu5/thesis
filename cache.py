import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache
import pickle
from tqdm import tqdm

# ten_gb_cache = 312_500_000
# four_gb_cache = 125_000_000
two_gb_cache = 500_000_000 # how many 32-bit integers can be held

features = 100
CACHE_SIZE = two_gb_cache // features

CACHE_SIZE = CACHE_SIZE // 8

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
        # TODO fix
        self.s = set()
        self.list = []
        # for layer_0 in tqdm(preload):
        #     done = False
        #     for x in layer_0:
        #         self.s.add(x.item())
        #         if len(self.s) == n:
        #             done = True
        #             break
        #     if done:
        #         break

        # self.list = list(self.s) #[x.item() for layer_0 in tqdm(preload) for x in layer_0]

        # if len(self.list) > n:
        #     print('warning: list is length', len(self.list), 'but n is:', n)
        self.test_count = {}

    def try_get(self, x):
        # if x not in self.test_count:
        #     self.test_count[x] = 0
        # self.test_count[x] += 1

        global total_evictions
        if x in self.s:
            return True

        if len(self.list) == self.n:
            evicted = self.list.pop()     
            self.s.remove(evicted)
            # print(self.test_count[evicted])
            total_evictions += 1

        self.list.append(x)
        self.s.add(x)

        return False


class MyPolicy:
    def __init__(self, n, preload):
        self.n = n
        self.s = set()
        self.list = []
        self.count = {}
        # for layer_0 in tqdm(preload):
        #     done = False
        #     for x in layer_0:
        #         self.s.add(x.item())
        #         if len(self.s) == n:
        #             done = True
        #             break
        #     if done:
        #         break

        # self.list = list(self.s) #[x.item() for layer_0 in tqdm(preload) for x in layer_0]

        # if len(self.list) > n:
        #     print('warning: list is length', len(self.list), 'but n is:', n)

        # self.count = {}
        # self.update_with_lookahead(preload)
        # self.list.sort(key=lambda x: -self.count[x])

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
            # print('evicting', evicted, 'has count', self.count[evicted])


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

    preload = nodes[:500]
    
    if policy == 'mine':
        cache = MyPolicy(CACHE_SIZE, preload=[])
        cache.update_with_lookahead(preload)
    elif policy == 'fifo':
        cache = FIFOPolicy(CACHE_SIZE, preload=[])
    elif policy == 'lru':
        cache = LRUPolicy(CACHE_SIZE, preload=[])

    # Warm caches with some queries
    for layer_0 in tqdm(preload):
        for x in layer_0:
            x = x.item()
            hit = cache.try_get(x)

    print('preload done')
    time = 0
    nodes = nodes[500:2500]

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

    # df = pd.DataFrame(cache_misses, columns=['cache misses over time'])
    # df = df.sample(n=5000, replace=False, random_state=1)

    # my_plot = sns.displot(df)
    # my_plot.savefig(f'{graph_name}_{batch_size}_{policy}.png')
    # plt.show()

if __name__ == '__main__':
    simulate('ogbn-products', 1, 'mine')
    simulate('ogbn-products', 1, 'lru')
    simulate('ogbn-products', 1, 'fifo')
