import time
from collections import defaultdict
import threading

class SimpleProfiler:
    def __init__(self):
        self.timers = defaultdict(float)
        self.counts = defaultdict(int)
        self.starts = {}
        self.lock = threading.Lock()

    def start(self, name):
        """Start a timer for a named block."""
        with self.lock:
            with open("profiler_debug.log", "a") as f: f.write(f"START {name}\n")
            self.starts[name] = time.perf_counter()

    def stop(self, name):
        """Stop the timer for a named block and accumulate time."""
        end_time = time.perf_counter()
        with self.lock:
            if name in self.starts:
                start_time = self.starts.pop(name)
                elapsed = end_time - start_time
                self.timers[name] += elapsed
                self.counts[name] += 1
                with open("profiler_debug.log", "a") as f: f.write(f"STOP {name} {elapsed}\n")
            else:
                pass
                with open("profiler_debug.log", "a") as f: f.write(f"STOP FAIL {name}\n")

    def report(self):
        """Return a formatted string report of timings."""
        with self.lock:
            with open("profiler_debug.log", "a") as f: f.write(f"REPORT called. Keys: {list(self.timers.keys())}\n")
            report_lines = ["\n=== Profiler Report ==="]
            # specific order if possible, or alphabetical
            sorted_keys = sorted(self.timers.keys())
            for key in sorted_keys:
                total_time = self.timers[key]
                count = self.counts[key]
                avg_time = total_time / count if count > 0 else 0
                report_lines.append(f"{key:<20} | Total: {total_time:.4f}s | Avg: {avg_time:.4f}s | Count: {count}")
            report_lines.append("=======================\n")
            return "\n".join(report_lines)

    def reset(self):
        """Reset all timers."""
        with self.lock:
            self.timers.clear()
            self.counts.clear()
            self.starts.clear()

class NoOpProfiler:
    """A dummy profiler that does nothing."""
    def start(self, name):
        pass

    def stop(self, name):
        pass

    def report(self):
        return ""

    def reset(self):
        pass
