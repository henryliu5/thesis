python benchmark/infer_cache.py -c static -b 0.8 -p 0.1 -o fast_sampling --use_pinned_mem
python benchmark/infer_cache.py -c count -b 0.8 -p 0.1 -o fast_sampling --use_pinned_mem
python benchmark/infer_cache.py -c async -b 0.8 -p 0.1 -o fast_sampling --use_pinned_mem

python benchmark/infer_cache.py -c static -b 0.8 -p 0.2 -o fast_sampling --use_pinned_mem
python benchmark/infer_cache.py -c count -b 0.8 -p 0.2 -o fast_sampling --use_pinned_mem
python benchmark/infer_cache.py -c async -b 0.8 -p 0.2 -o fast_sampling --use_pinned_mem

python benchmark/infer_cache.py -c static -b 0.8 -p 0.3 -o fast_sampling --use_pinned_mem
python benchmark/infer_cache.py -c count -b 0.8 -p 0.3 -o fast_sampling --use_pinned_mem
python benchmark/infer_cache.py -c async -b 0.8 -p 0.3 -o fast_sampling --use_pinned_mem



# # # no pinned memory tests
# python benchmark/infer_cache.py -c static
# python benchmark/infer_cache.py -c count
# python benchmark/infer_cache.py -c lfu
# python benchmark/infer_cache.py -c async

# python benchmark/infer_cache.py -c static -b 0.8
# python benchmark/infer_cache.py -c count -b 0.8
# python benchmark/infer_cache.py -c lfu -b 0.8
# python benchmark/infer_cache.py -c async -b 0.8

# # tests with pinned memory
# python benchmark/infer_cache.py -c static --use_pinned_mem
# python benchmark/infer_cache.py -c count --use_pinned_mem
# python benchmark/infer_cache.py -c lfu --use_pinned_mem
# python benchmark/infer_cache.py -c async --use_pinned_mem

# python benchmark/infer_cache.py -c static -b 0.8 --use_pinned_mem
# python benchmark/infer_cache.py -c count -b 0.8 --use_pinned_mem
# python benchmark/infer_cache.py -c lfu -b 0.8 --use_pinned_mem
# python benchmark/infer_cache.py -c async -b 0.8 --use_pinned_mem

# # repeat all but with profiling
# # no pinned memory tests
# # python benchmark/infer_cache.py -c static --profile
# # python benchmark/infer_cache.py -c count --profile
# # python benchmark/infer_cache.py -c lfu --profile
# # python benchmark/infer_cache.py -c async --profile

# # python benchmark/infer_cache.py -c static -b 0.8 --profile
# # python benchmark/infer_cache.py -c count -b 0.8 --profile
# # python benchmark/infer_cache.py -c lfu -b 0.8 --profile
# # python benchmark/infer_cache.py -c async -b 0.8 --profile

# # # tests with pinned memory
# # python benchmark/infer_cache.py -c static --use_pinned_mem --profile
# # python benchmark/infer_cache.py -c count --use_pinned_mem --profile
# # python benchmark/infer_cache.py -c lfu --use_pinned_mem --profile
# # python benchmark/infer_cache.py -c async --use_pinned_mem --profile

# # python benchmark/infer_cache.py -c static -b 0.8 --use_pinned_mem --profile
# # python benchmark/infer_cache.py -c count -b 0.8 --use_pinned_mem --profile
# # python benchmark/infer_cache.py -c lfu -b 0.8 --use_pinned_mem --profile
# # python benchmark/infer_cache.py -c async -b 0.8 --use_pinned_mem --profile