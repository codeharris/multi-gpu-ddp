import os
from dataclasses import dataclass
import torch
import torch.distributed as dist


@dataclass
class DistState:
    use_ddp: bool
    rank: int
    local_rank: int
    world_size: int
    is_main_process: bool
    device: torch.device


def setup_distributed(cfg):
    use_ddp = cfg.get("use_ddp", False)

    if not use_ddp:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return DistState(
            use_ddp=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main_process=True,
            device=device,
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    is_main = (rank == 0)

    if is_main:
        host = os.uname().nodename
        print(
            f"🔗 Initializing DDP: rank {rank}/{world_size} on host {host}, backend=nccl",
            flush=True,
        )

    return DistState(
        use_ddp=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main_process=is_main,
        device=device,
    )


def cleanup_distributed(state: DistState):
    if state.use_ddp and dist.is_initialized():
        dist.destroy_process_group()
