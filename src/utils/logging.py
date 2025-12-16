def print_rank0(msg: str, is_main_process: bool = True):
    """
    Print only from rank 0 / main process (DDP safe).
    flush=True ensures logs appear immediately in Slurm output files.
    """
    if is_main_process:
        print(msg, flush=True)
