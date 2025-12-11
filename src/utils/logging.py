def print_rank0(msg: str, is_main_process: bool = True):
    """
    Later, when we use DDP, only rank 0 will print.
    For now, it's always main process.
    """
    if is_main_process:
        print(msg)
