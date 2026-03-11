import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    x = torch.ones(1, device="cuda") * (rank + 1)
    dist.all_reduce(x)
    print(f"rank={rank}/{world}, all_reduce={x.item()}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
