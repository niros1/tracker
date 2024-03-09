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
    boxes: torch.Tensor  # Bounding boxes returned from gdino
    logits: torch.Tensor
    phrases: typing.List[str]
    cordinates: typing.Optional[numpy.ndarray] = (
        None  # Real frame cordinates, respected to the frame size. based on boxes
    )

    @property
    def logit_trashold(self) -> float:
        return 0.5

    @property
    def top_score_index(self) -> int | None:
        """
        Returns the index of the top score founded by the vision model
        """
        if len(self.logits) > 0:
            max_index = torch.argmax(self.logits)
            return int(max_index)
        else:
            return 0

    @property
    def box(self):
        return self.boxes[self.top_score_index].unsqueeze(0)

    @property
    def logit(self):
        return self.logits[self.top_score_index].unsqueeze(0)

    @property
    def phras(self):
        return [self.phrases[self.top_score_index]]

    @property
    def cordinate(self):
        assert self.cordinates is not None
        if bool(self.logit[0] < self.logit_trashold):
            return tensor([1000, 1200, 0, 0])
        # if(self.logit)
        return self.cordinates[self.top_score_index]


class TrackinfVideoData(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    all: list[TrackingFrameData] = []
    X: numpy.ndarray = None  # X smoothed cordinates
    Y: numpy.ndarray = None  # X smoothed cordinates


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
