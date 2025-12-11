import os
import socket

import torch
import torch.distributed as dist


class DistState:
    """Holds information about the current (possibly distributed) setup."""
    def __init__(self, use_ddp: bool, world_size: int, rank: int, local_rank: int, device):
        self.use_ddp = use_ddp
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.device = device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _infer_slurm_local_rank(rank: int) -> int:
    """Derive local_rank from global rank under Slurm if LOCAL_RANK is not set."""
    if torch.cuda.is_available():
        return rank % torch.cuda.device_count()
    return 0


def setup_distributed(dist_cfg: dict) -> DistState:
    """
    Initialize (or skip) DDP depending on dist_cfg["use_ddp"].

    Supports three launch styles:
    - torchrun ... (sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT)
    - srun torchrun ... (same as above, but under Slurm)
    - srun python ... (uses SLURM_PROCID / SLURM_NTASKS, MASTER_* from script)
    """
    use_ddp = dist_cfg.get("use_ddp", False)

    # ---------- Single-process path ----------
    if not use_ddp:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            local_rank = 0
            print("🛠 Using single GPU: cuda:0")
        else:
            device = torch.device("cpu")
            local_rank = 0
            print("⚠️ CUDA not available, using CPU.")
        return DistState(
            use_ddp=False,
            world_size=1,
            rank=0,
            local_rank=local_rank,
            device=device,
        )

    # ---------- DDP path ----------
    # 1) torchrun style: RANK/WORLD_SIZE/LOCAL_RANK are set
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    # 2) pure Slurm: srun python ...
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("LOCAL_RANK", _infer_slurm_local_rank(rank)))
    else:
        raise RuntimeError(
            "DDP is enabled but RANK/WORLD_SIZE or SLURM_PROCID/SLURM_NTASKS were not found.\n"
            "Make sure you launch with torchrun or srun correctly."
        )

    backend = dist_cfg.get("backend", "nccl")

    # Select device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # MASTER_* should be set by torchrun or the Slurm script; fall back for single-node
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    if rank == 0:
        host = socket.gethostname()
        print(f"🔗 Initializing DDP: rank {rank}/{world_size} on host {host}, backend={backend}")

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

    return DistState(
        use_ddp=True,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
    )


def cleanup_distributed(state: DistState):
    """Cleanup for DDP."""
    if state.use_ddp and dist.is_initialized():
        dist.destroy_process_group()
