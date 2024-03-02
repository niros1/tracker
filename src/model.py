import typing

import numpy
from torch import tensor
import torch
from pydantic import BaseModel

class VideoProperties(BaseModel):
    fps: float
    width: int
    height: int
    fourcc: str

class TrackingFrameData(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    index: int
    source_index: int
    boxes: torch.Tensor # Bounding boxes returned from gdino
    logits: torch.Tensor
    phrases: typing.List[str]
    cordinates: typing.Optional[numpy.ndarray] = None # Real frame cordinates, respected to the frame size. based on boxes


class TrackinfVideoData(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    all: list[TrackingFrameData] = []
    X: numpy.ndarray = None # X smoothed cordinates
    Y: numpy.ndarray = None # X smoothed cordinates


class Stack:
    def __init__(self, max_size):
        self.stack = []
        self.max_size = max_size

    def push(self, item):
        if len(self.stack) == self.max_size:
            self.stack.pop(0)  # Remove the oldest item
        self.stack.append(item)

    def pop(self):
        if len(self.stack) < 1:
            return None
        return self.stack.pop()

    def size(self):
        return len(self.stack)
    def __str__(self):
        return str(self.stack)
    def __iter__(self):
        return iter(self.stack)