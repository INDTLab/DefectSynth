import argparse
import json
import os

import cv2
import numpy as np

from torch.utils.data import Dataset


class ControlNetPairDataset(Dataset):

    def __init__(
        self,
        prompt_json_path="./training/prompt.json",
        data_root="./training/MVTec_AD",
    ):
        self.data = []
        self.prompt_json_path = prompt_json_path
        self.data_root = os.path.abspath(data_root)

        with open(self.prompt_json_path, "rt") as f:
            for line in f:
                self.data.append(json.loads(line))

        for i, item in enumerate(self.data):
            try:
                source_filename = item["source"]
                target_filename = item["target"]
                _ = item["prompt"]
            except KeyError as e:
                raise KeyError(
                    f"Dataset entry {i} in {self.prompt_json_path!r} missing required key: {e.args[0]!r}"
                ) from e

            for key, rel in (("source", source_filename), ("target", target_filename)):
                path = os.path.join(self.data_root, rel)
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"Dataset entry {i}: {key} file does not exist: {path}"
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        target_filename = item["target"]
        prompt = item["prompt"]

        source_path = os.path.join(self.data_root, source_filename)
        target_path = os.path.join(self.data_root, target_filename)

        source = cv2.imread(source_path)
        target = cv2.imread(target_path)

        if source is None:
            raise RuntimeError(
                f"cv2.imread failed for source: {source_path}"
            )

        if target is None:
            raise RuntimeError(
                f"cv2.imread failed for target: {target_path}"
            )

        source = cv2.resize(source, (512, 512))
        target = cv2.resize(target, (512, 512))

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = source.astype(np.float32) / 255.0

        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset sanity check."
    )
    parser.add_argument("--prompt_json_path", type=str, default="./training/prompt.json")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./training/MVTec_AD",
        help="Dataset root.",
    )
    parser.add_argument(
        "--check_count",
        type=int,
        default=0,
        help="Num samples to check.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = ControlNetPairDataset(
        prompt_json_path=args.prompt_json_path,
        data_root=args.data_root,
    )
    print(f"Loaded dataset from: {args.prompt_json_path}")
    print(f"Data root (absolute): {dataset.data_root}")
    print(f"Total samples: {len(dataset)}")

    if args.check_count > 0:
        n = min(args.check_count, len(dataset))
        for i in range(n):
            sample = dataset[i]
            if i == 0:
                print(f"Sample keys: {list(sample.keys())}")
                print(f"hint shape: {sample['hint'].shape}, jpg shape: {sample['jpg'].shape}")
        print(f"Checked {n} samples successfully.")


if __name__ == "__main__":
    main()
