


            if(use_gpu_transfer){
                torch::Tensor new_feats, new_nids;
                int cache_slots = sliding_replace_nids.sizes()[0];
                if(gpu_feat_holder.tryGet(new_feats, new_nids) && cache_slots > 0){
                    // If we get a new node id, only add if in Top K and not in cache yet
                    auto new_candidate_mask = topk_mask.index({new_nids}) & ~cache_mask.index({new_nids});

                    new_nids = new_nids.masked_select(new_candidate_mask);
                    new_feats = new_feats.index({new_candidate_mask.to(new_feats.device())});

                    int new_options = new_nids.sizes()[0];
                    
                    if(new_options > 0){
                        
                        int num_to_add = std::min(new_options, cache_slots);
                        new_nids = new_nids.slice(0, 0, num_to_add);
                        new_feats = new_feats.slice(0, 0, num_to_add);

                        torch::Tensor replace_nids = sliding_replace_nids.slice(0, 0, num_to_add);
                        sliding_replace_nids = sliding_replace_nids.slice(0, num_to_add, cache_slots);

                        cacheUpdateFromHolder(new_nids, new_feats, replace_nids);
                    }

                }
            }



                    auto start2 = high_resolution_clock::now();
                    torch::Tensor replace_cache_idxs = getLeastUsedCacheIndices(cache_size);
                    sliding_replace_nids = reverse_mapping.index({replace_cache_idxs});
                    auto stop2 = high_resolution_clock::now();
                    auto duration2 = duration_cast<microseconds>(stop2 - start2);
                    cout << "compute replace time: " << duration2.count() << endl;