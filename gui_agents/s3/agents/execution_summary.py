


from dataclasses import dataclass
from typing import Callable


@dataclass
class ExecutionSummary:
    plan: str = None
    plan_code: str = None
    executable: str | Callable = None
    reflection: str = None
    reflection_thoughts: str = None


    @property
    def exec_str(self) -> str:
        if isinstance(self.executable, str):
            return self.executable
        elif callable(self.executable):
            return self.executable.__name__
        else:
            return None
        
    def call_executable(self, *args, **kwargs):
        if callable(self.executable):
            return self.executable(*args, **kwargs)
        if isinstance(self.executable, str):
            return exec(self.executable)
        raise ValueError(f"{self.executable} is neither a string nor a callable function.")