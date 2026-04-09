import argparse
from torch_geometric.datasets import TUDataset

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--dataset", type=str, default="PROTEINS")
args = parser.parse_args()

print(f"Downloading {args.dataset} to {args.data_dir}...")
dataset = TUDataset(root=args.data_dir, name=args.dataset)
print(f"Done. {len(dataset)} graphs.")