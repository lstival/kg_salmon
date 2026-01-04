import time
import torch
from typing import Any, Callable, Tuple

class KGProfiler:
    """
    Utilities for profiling Knowledge Graph models.
    """
    @staticmethod
    def measure_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """
        Measure the execution time of a function.
        
        Returns:
            (result, elapsed_time_seconds)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    @staticmethod
    def estimate_gflops(model: Any, batch_size: int = 128) -> float:
        """
        Estimate GFLOPS for a forward pass of a KG model.
        """
        if hasattr(model, 'embedding_dim'):
            dim = model.embedding_dim
        else:
            dim = 50
            
        total_ops = 5 * dim * batch_size
        gflops = total_ops / 1e9
        return gflops
