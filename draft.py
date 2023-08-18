from typing import Union, NamedTuple
import tensorflow as tf
from dataclasses import dataclass


class KeyPoint(NamedTuple):
    pt: list = []
    size: list = []
    angle: list = []
    octave: list = []
    octave_id: list = []
    response: list = []


