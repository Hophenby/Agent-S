


from dataclasses import dataclass
from typing import Callable


@dataclass
class ExecutionSummary:
    plan: str = None
    plan_action: str = None
    executable: str | Callable = None
    additionaal_info: str = None
    reflection_thoughts: str = None


    @property
    def exec_str(self) -> str:
        if isinstance(self.executable, str):
            return self.executable
        elif callable(self.executable):
            return self.executable.__name__
        else:
            return None
        
    @property
    def can_execute(self) -> bool:
        return self.executable is not None and (isinstance(self.executable, str) or callable(self.executable))
        
    def call_executable(self, *args, **kwargs):
        if callable(self.executable):
            return self.executable(*args, **kwargs)
        if isinstance(self.executable, str):
            return exec(self.executable)
        raise ValueError(f"{self.executable} is neither a string nor a callable function.")
    

    def format_summary(self) -> str:
        reflection_str = f"Plan:\n{self.plan}\n\n"
        if self.plan_action:
            reflection_str += f"Plan Action:\n{self.plan_action}\n\n"
        if self.additionaal_info:
            reflection_str += f"Additional Info:\n{self.additionaal_info}\n\n"
        return reflection_str