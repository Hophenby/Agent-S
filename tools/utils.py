
from typing import Generic, TypeVar

T = TypeVar("T")

class AtomicObject(Generic[T]):
    def __init__(self, value: T):
        self.value: T = value