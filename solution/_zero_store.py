"""One metered, exactly representable float32 zero, independent of inputs."""
import flopscope.numpy as f

_zero = None

def zero32():
    global _zero
    if _zero is None:
        _zero = f.zeros((), dtype=f.float32)
    return _zero
