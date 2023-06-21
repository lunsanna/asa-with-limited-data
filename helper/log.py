import logging
import sys
import time

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
    return f"{hours}:{minutes:02d}:{seconds:02d}"