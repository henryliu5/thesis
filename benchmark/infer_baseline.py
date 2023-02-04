from fast_inference.dataset import InferenceDataset

if __name__ == '__main__':
    infer_data = InferenceDataset('reddit', 0.1, force_reload=True, verbose=True)
    g = infer_data[0]
    print(g)