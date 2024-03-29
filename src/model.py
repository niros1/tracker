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


def is_point_in_boxes(point):
    # Should be uniqie for each video

    for box in blind_spots:
        if (box[0][0] <= point[0] and box[1][0] >= point[0]) and (
            box[0][1] <= point[1] and box[1][1] >= point[1]
        ):
            return True

        # if box[0][0] <= point[0] <= box[1][0] and box[0][1] <= point[1] <= box[1][1]:
        #     return True
    return False


# Example
# TrackingFrameData(index=0, source_index=0,
#                   boxes=tensor([[0.8665, 0.5211, 0.0061, 0.0093]]),
#                   logits=tensor([0.4713]), phrases=['basketball'],
#                   cordinates=array([[2334.7947,  785.0676, 2351.2825,  799.1918]],
#                                    dtype=float32))
last_coor: numpy.ndarray = tensor([3000, 1200, 0, 0])


class TrackingFrameData(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return f"{self.index} {self.source_index} {self.logits} {self.cordinates} {self.phrases}"

    index: int
    source_index: int
    boxes: torch.Tensor  # Bounding boxes returned from gdino
    logits: torch.Tensor
    phrases: typing.List[str]
    cordinates: typing.Optional[numpy.ndarray] = (
        None  # Real frame cordinates, respected to the frame size. based on boxes
    )

    top_score_index_mem: int = -1

    # def __setstate__(self, state):
    #     if not hasattr(self, "__pydantic_model__"):
    #         self.__pydantic_model__ = set()
    #     if not hasattr(self, "last_coor"):
    #         self.last_coor = None
    #     self.__dict__.update(state)

    @property
    def logit_trashold(self) -> float:
        # Inferrance trashold
        return 0.45

    @property
    def top_score_index(self) -> int | None:
        """
        Returns the index of the top score founded by the vision model
        """
        # if self.top_score_index_mem > -1:
        #     return self.top_score_index_mem
        # assert self.cordinates
        # Get the indices of the sorted elements
        numpy_array = self.logits.detach().cpu().numpy()
        indices = numpy.argsort(numpy_array)[::-1]
        for i in indices:
            if not is_point_in_boxes(self.cordinates[i]):
                self.top_score_index_mem = i
                return i
        return None

        # if len(self.logits) > 0:
        #     max_index = torch.argmax(self.logits)
        #     return int(max_index)
        # else:
        #     return 0

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
    def raw_cordinate(self):
        return self.cordinates[0]

    @property
    def cordinate(self):
        global last_coor
        if self.cordinates is None:
            return last_coor

        # if bool(self.logit[0] < self.logit_trashold):
        #     return tensor([1000, 1200, 0, 0])
        # if(self.logit)
        i = self.top_score_index

        if i is None:
            # return tensor([1000, 1200, 0, 0])
            return last_coor

        coor = self.cordinates[i]
        s1 = coor[0]
        if coor[0] > 1500 and coor[0] < 2300:
            coor[0] = min(int(coor[0]) * 1.5, 2500)
        if coor[0] < 1200:
            coor[0] = int(coor[0]) * 0.6
        # if coor[0] < 300:
        #     print(f"coor[0] < 300 {coor[0]} {s1}")
        # No vertical movment
        coor[1] = 1200
        last_coor = coor
        return coor


class TrackinfVideoData(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    all: list[TrackingFrameData] = []
    X: numpy.ndarray = None  # X smoothed cordinates
    Y: numpy.ndarray = None  # X smoothed cordinates


x_factor = -20
y_factor = 100
blind_spots: typing.List = [
    [(489+x_factor, 707+y_factor), (630+x_factor, 820+y_factor)],
    [(706+x_factor, 744+y_factor), (815+x_factor, 838+y_factor)],
    [(1359+x_factor, 749+y_factor), (1491+x_factor, 836+y_factor)],
    [(2006+x_factor, 770+y_factor), (2100+x_factor, 825+y_factor)],
    [(2279+x_factor, 730+y_factor), (2415+x_factor, 806+y_factor)],
]


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
