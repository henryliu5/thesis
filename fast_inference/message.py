import torch
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from fast_inference.dataset import FastEdgeRepr

class MessageType(Enum):
    INFERENCE = 1
    RESET = 2
    SHUTDOWN = 3
    RESPONSE = 4
    WARMUP = 5

@dataclass(frozen=True)
class RequestPayload:
    nids: torch.Tensor
    features: torch.Tensor
    edges: FastEdgeRepr

@dataclass(frozen=True)
class SamplerQueuePayload:
    required_feats: torch.Tensor
    mfg_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

@dataclass(frozen=True)
class FeatureQueuePayload:
    inputs: torch.Tensor
    mfg_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

@dataclass(frozen=True)
class ResponsePayload:
    pass

@dataclass(frozen=True)
class Message:
    id: int
    trial: int
    timing_info: Dict[str, float]
    msg_type: MessageType
    payload: None | RequestPayload | SamplerQueuePayload | FeatureQueuePayload | ResponsePayload = None
