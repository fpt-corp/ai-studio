import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP pretrained ResNet50 training demo")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def build_datasets(rank: int, args: argparse.Namespace) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    normalize = transforms.Normalize(
        mean=weights.meta["mean"] if "mean" in weights.meta else [0.485, 0.456, 0.406],
        std=weights.meta["std"] if "std" in weights.meta else [0.229, 0.224, 0.225],
    )
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ]
    )
    # Download once on rank 0, then all ranks continue.
    if rank == 0:
        datasets.MNIST(root=args.data_dir, train=True, transform=transform, download=True)
        datasets.MNIST(root=args.data_dir, train=False, transform=transform, download=True)
    dist.barrier()

    train_ds = datasets.MNIST(root=args.data_dir, train=True, transform=transform, download=False)
    test_ds = datasets.MNIST(root=args.data_dir, train=False, transform=transform, download=False)

    if args.max_train_samples > 0:
        train_ds = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    if args.max_test_samples > 0:
        test_ds = Subset(test_ds, range(min(args.max_test_samples, len(test_ds))))
    return train_ds, test_ds


def reduce_metric(total: float, count: int, device: torch.device) -> tuple[float, int]:
    metric = torch.tensor([total, count], dtype=torch.float64, device=device)
    dist.all_reduce(metric, op=dist.ReduceOp.SUM)
    return metric[0].item(), int(metric[1].item())


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)

    train_ds, test_ds = build_datasets(rank, args)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Rank 0 downloads pretrained weights first to avoid concurrent cache writes.
    if rank == 0:
        _ = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    dist.barrier()

    # Fine-tune only the classification head of pretrained ResNet50 for fast demo runtime.
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)

    if rank == 0:
        print(
            f"Start DDP pretrained ResNet50 training on MNIST: world_size={world_size}, epochs={args.epochs}, "
            f"train_samples={len(train_ds)}, test_samples={len(test_ds)}",
            flush=True,
        )

    for epoch in range(args.epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch = labels.size(0)
            total_loss += loss.item() * batch
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch

        sum_loss, sum_samples = reduce_metric(total_loss, total_samples, device)
        sum_correct, _ = reduce_metric(float(total_correct), total_samples, device)
        train_loss = sum_loss / max(sum_samples, 1)
        train_acc = sum_correct / max(sum_samples, 1)

        ddp_model.eval()
        eval_correct = 0
        eval_samples = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = ddp_model(images)
                eval_correct += (logits.argmax(dim=1) == labels).sum().item()
                eval_samples += labels.size(0)

        sum_eval_correct, sum_eval_samples = reduce_metric(float(eval_correct), eval_samples, device)
        eval_acc = sum_eval_correct / max(sum_eval_samples, 1)

        if rank == 0:
            print(
                f"[Epoch {epoch + 1}/{args.epochs}] train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} eval_acc={eval_acc:.4f}",
                flush=True,
            )

    if rank == 0:
        print("Training done.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
