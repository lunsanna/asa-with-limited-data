#!/usr/bin/env python3
import logging
import sys
import time
import psutil
import os 
import sys
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from torch.cuda import device_count

def configure_logger(verbose:bool) -> None:
    """
    Args:
        verbose (bool): determine the logging level
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
def print_time(start:int) -> str:
    """Produce a readble time period string based on the input s

    Args:
        start (int): start time

    Returns:
        str: readable time period string
    """
    s:float = time.time()-start
    hours:int = int(s // 3600)
    minutes:int = int((s // 60) % 60)
    seconds:int = int(s % 60)
    return f"Duration: {hours}:{minutes:02d}:{seconds:02d}"


def print_memory_usage():
    """Produce a readable string that reports mem usage.
    
    Returns:
        str: print out for the mem usage of the current process.
    """
    gb_factor = 1024 ** 3

    pid = os.getpid()
    current_process = psutil.Process(pid)
    cpu_memory = current_process.memory_info()

    n_device = device_count()
    gpu_memory = ""
    if n_device > 0:
        nvmlInit()
        
        for i in range(n_device):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            gpu_memory += f" Cuda {i}: memory acclocated: {info.used//gb_factor} G."
        gpu_memory.strip("\n")

    return f"CPU mem usage: {cpu_memory.rss/gb_factor:.2f}G. {gpu_memory}"
