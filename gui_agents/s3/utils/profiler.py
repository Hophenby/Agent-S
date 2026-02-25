


import time


class Profiler:
    def __init__(self):
        self.timings = {}
        self.current_key = None

    def start_step(self, key: str):
        self.start(key)
        self.current_key = key

    def next_step(self, key: str):
        self.end_step(self.current_key)
        self.start_step(key)

    def end_step(self):
        self.end(self.current_key)
        self.current_key = None

    def start(self, key: str):
        self.timings[key] = {"start": time.time(), "end": None}

    def end(self, key: str):
        if key in self.timings and self.timings[key]["end"] is None:
            self.timings[key]["end"] = time.time()
        

    def get_duration(self, key: str) -> float:
        if key in self.timings and self.timings[key]["end"] is not None:
            return self.timings[key]["end"] - self.timings[key]["start"]
        return 0.0
    
    def report(self):
        report_lines = [
            "Profiler Report:", 
            "-----------------", 
            f"Total Steps: {len(self.timings)}; Total Time: {sum(self.get_duration(key) for key in self.timings):.2f} seconds", 
            "-----------------"]
        for key in self.timings.keys():
            duration = self.get_duration(key)
            report_lines.append(f"{key}: {duration:.2f} seconds")
        return "\n".join(report_lines)