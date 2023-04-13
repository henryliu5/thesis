import torch
from fast_inference.models.factory import load_model
from fast_inference.pipeline.pipeline_worker import WorkerType, ControlMessage
from fast_inference.pipeline.pipeline import create_pipeline
from torch.multiprocessing import Barrier, Queue, set_start_method
import dgl


def test_pipeline_creation_teardown():
    # if torch.multiprocessing.get_start_method() != 'spawn':
    set_start_method('spawn', force=True)

    g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
    g.ndata['feat'] = torch.ones(6, 1)
    model = load_model('gcn', 4, 4)

    samplers = 2
    data_loaders = 2
    model_executors = 2

    num_workers = samplers + data_loaders + model_executors

    request_queue = Queue()
    response_queue = Queue()
    barriers = {'start': Barrier(1 + num_workers)}

    manager, workers = create_pipeline(num_devices=1, samplers=samplers, data_loaders=data_loaders, model_executors=model_executors, cache_type='static', cache_percent=0.1, 
                              g=g, model=model, request_queue=request_queue, response_queue=response_queue, barriers=barriers)

    barriers['start'].wait()

    assert len(workers) == num_workers, f'Expected {num_workers} workers, got {len(workers)}'
    assert workers[0].worker_type == WorkerType.SAMPLER
    assert workers[1].worker_type == WorkerType.SAMPLER
    assert workers[2].worker_type == WorkerType.DATA_LOADER
    assert workers[3].worker_type == WorkerType.DATA_LOADER
    assert workers[4].worker_type == WorkerType.MODEL_EXECUTOR
    assert workers[5].worker_type == WorkerType.MODEL_EXECUTOR

    request_queue.put(ControlMessage(0))
    request_queue.put(ControlMessage(0))

    # # manager.shutdown()
    # barriers['finish'].wait()
    [worker.join() for worker in workers]
    print('joined')
    manager.worker_types_buf.close()
    print('closing shm')
    manager.worker_types_buf.unlink()
    print('unlink shm')
    print('main exiting')
    

if __name__ == '__main__':
    test_pipeline_creation_teardown()