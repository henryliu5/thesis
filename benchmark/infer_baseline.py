from fast_inference.dataset import InferenceDataset

if __name__ == '__main__':
    graph1 = InferenceDataset('reddit', 0.1, force_reload=True, verbose=True)
    print(graph1)
    graph2 = InferenceDataset('reddit', 0.1, force_reload=False, verbose=True)
    print(graph2)