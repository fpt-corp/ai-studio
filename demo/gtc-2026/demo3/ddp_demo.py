import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def setup() -> tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def main() -> None:
    rank, world_size, local_rank = setup()
    device = torch.device("cuda", local_rank)

    # Tutorial-style minimal DDP training step.
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    ).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Same random seed so each rank starts from comparable data for demo.
    torch.manual_seed(42 + rank)
    x = torch.randn(64, 16, device=device)
    y = torch.randn(64, 8, device=device)

    optimizer.zero_grad(set_to_none=True)
    pred = ddp_model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    # Aggregate loss across ranks for a single demo-friendly metric.
    loss_tensor = loss.detach()
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = (loss_tensor / world_size).item()

    print(
        f"rank={rank}/{world_size} host={os.uname().nodename} "
        f"local_rank={local_rank} loss={loss.item():.6f} avg_loss={avg_loss:.6f}",
        flush=True,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
