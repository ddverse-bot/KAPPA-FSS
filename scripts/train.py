import argparse
import os
import torch

from kappa_fss.config import Config, DATASET_EPISODES
from kappa_fss.data import discover_pairs, make_loaders
from kappa_fss.engine import seed_everything, train_one_epoch, evaluate, save_checkpoint
from kappa_fss.model import KAPPAFSS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("kvasir", "isic2016"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--val-episodes", type=int, default=None)
    parser.add_argument("--test-episodes", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config(image_size=args.image_size, seed=args.seed)
    counts = DATASET_EPISODES[args.dataset].copy()
    config.train_episodes = args.train_episodes or counts["train"]
    config.val_episodes = args.val_episodes or counts["val"]
    config.test_episodes = args.test_episodes or counts["test"]
    if args.epochs is not None:
        config.epochs = args.epochs
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = discover_pairs(args.dataset, args.data_root)
    datasets, loaders = make_loaders(pairs, config)
    model = KAPPAFSS(config).to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    os.makedirs(args.output, exist_ok=True)
    for epoch in range(1, config.epochs + 1):
        training = train_one_epoch(model, loaders["train"], optimizer, config, device)
        validation = evaluate(model, loaders["val"], 5, config, device)
        print({"epoch": epoch, "train": training, "validation_5shot": validation})
        save_checkpoint(
            os.path.join(args.output, f"kappa_fss_{args.dataset}_epoch_{epoch:03d}.pt"),
            model, optimizer, config, epoch, args.dataset,
        )
    for shots in config.shots:
        print({"test_shots": shots, "metrics": evaluate(model, loaders["test"], shots, config, device)})

if __name__ == "__main__":
    main()
