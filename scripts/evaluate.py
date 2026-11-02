import argparse
import torch

from kappa_fss.config import Config
from kappa_fss.data import discover_pairs, make_loaders
from kappa_fss.model import KAPPAFSS
from kappa_fss.engine import evaluate
from kappa_fss.visualize import show_episode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("kvasir", "isic2016"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = Config(**checkpoint["config"])
    config.image_size = args.image_size
    config.test_episodes = args.episodes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = discover_pairs(args.dataset, args.data_root)
    datasets, loaders = make_loaders(pairs, config, {
        "train": 1, "val": 1, "test": args.episodes
    })
    model = KAPPAFSS(config).to(device)
    model.load_state_dict(checkpoint["model"])
    for shots in config.shots:
        print({"shots": shots, "metrics": evaluate(model, loaders["test"], shots, config, device)})
    if args.visualize:
        figure = show_episode(model, datasets["test"][0], 5, device, config.threshold)
        figure.savefig(f"{args.dataset}_qualitative.png", dpi=180, bbox_inches="tight")

if __name__ == "__main__":
    main()
