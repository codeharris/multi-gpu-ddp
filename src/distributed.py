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

    # Validate CUDA availability and visible device count before selecting device
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. If you intend to run DDP on CPU, set distributed.use_ddp=false or use backend 'gloo' and remove device_ids in DDP."
        )

    num_visible = torch.cuda.device_count()
    if local_rank >= num_visible:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        raise RuntimeError(
            (
                f"CUDA invalid device ordinal: LOCAL_RANK={local_rank} but only {num_visible} device(s) are visible. "
                f"CUDA_VISIBLE_DEVICES={vis}. "
                "Fix by matching --nproc_per_node to the number of visible GPUs on this node, "
                "or adjust CUDA_VISIBLE_DEVICES accordingly."
            )
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        backend = cfg.get("backend", "nccl")
        dist.init_process_group(backend=backend, init_method="env://")

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
