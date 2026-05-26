import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MVTecDRAEMTestDataset_partial(Dataset):
    def __init__(self, root_dir, resize_shape=None, seed=123):
        set_seed(seed)
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.images = []
        self.anomaly_names = os.listdir(self.root_dir)
        for _, anomaly_name in enumerate(self.anomaly_names):
            img_path = os.path.join(root_dir, anomaly_name)
            img_files = os.listdir(img_path)
            img_files.sort(key=lambda x: int(x[:3]))
            self.images += [
                os.path.join(img_path, file_name) for file_name in img_files
            ]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            print("mask_path: ", mask_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape is not None:
            image = cv2.resize(
                image, dsize=(self.resize_shape[1], self.resize_shape[0])
            )
            mask = cv2.resize(
                mask, dsize=(self.resize_shape[1], self.resize_shape[0])
            )

        image = image / 255.0
        mask = mask / 255.0

        image = (
            np.array(image)
            .reshape((image.shape[0], image.shape[1], 3))
            .astype(np.float32)
        )
        mask = (
            np.array(mask)
            .reshape((mask.shape[0], mask.shape[1], 1))
            .astype(np.float32)
        )

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == "good":
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            dir_path = os.path.dirname(os.path.dirname(dir_path))
            mask_path = os.path.join(dir_path, "ground_truth")
            mask_path = os.path.join(mask_path, base_dir)
            print("img_path: ", img_path)
            mask_file_name = file_name.split(".")[0] + "_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {
            "image": image,
            "has_anomaly": has_anomaly,
            "mask": mask,
            "idx": idx,
        }
        return sample


class MVTec_Anomaly_Detection(Dataset):
    def __init__(
        self,
        args,
        sample_name,
        length=5000,
        anomaly_id=None,
        recon=False,
    ):
        self.recon = recon
        self.good_path = os.path.join(args.mvtec_path, sample_name, "train", "good")
        self.good_files = [
            os.path.join(self.good_path, i) for i in os.listdir(self.good_path)
        ]
        self.root_dir = os.path.join(args.generated_data_path, sample_name)
        self.mask_dir = os.path.join(args.generated_mask_path, sample_name)
        self.anomaly_names = os.listdir(self.root_dir)
        if anomaly_id is not None:
            self.anomaly_names = self.anomaly_names[anomaly_id : anomaly_id + 1]
            print("training subsets", self.anomaly_names)
        l = len(self.anomaly_names)
        self.anomaly_num = l
        self.img_paths = []
        self.mask_paths = []
        for _, anomaly in enumerate(self.anomaly_names):
            img_path = []
            mask_path = []
            anomaly_img_dir = os.path.join(self.root_dir, anomaly)
            anomaly_mask_dir = os.path.join(self.mask_dir, anomaly)

            img_files = [
                f
                for f in os.listdir(anomaly_img_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]
            img_files.sort()

            for img_file in img_files:
                stem, _ = os.path.splitext(img_file)
                img_full = os.path.join(anomaly_img_dir, img_file)
                mask_full = os.path.join(anomaly_mask_dir, f"{stem}_mask.png")
                if os.path.exists(mask_full):
                    img_path.append(img_full)
                    mask_path.append(mask_full)

            img_path = img_path[:500]
            mask_path = mask_path[:500]
            print(f"Loaded {len(img_path)} image-mask pairs from {anomaly}")
            self.img_paths.append(img_path.copy())
            self.mask_paths.append(mask_path.copy())
        for i in range(l):
            print(len(self.img_paths[i]), len(self.mask_paths[i]))
        self.loader = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize([256, 256])]
        )
        self.length = length
        if self.length is None:
            self.length = len(self.good_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if random.random() > 0.5:
            image = self.loader(
                Image.open(
                    self.good_files[idx % len(self.good_files)]
                ).convert("RGB")
            )
            mask = torch.zeros((1, image.size(-2), image.size(-1)))
            has_anomaly = np.array([0], dtype=np.float32)
            sample = {
                "image": image,
                "has_anomaly": has_anomaly,
                "mask": mask,
                "anomaly_id": -1,
            }
            if self.recon:
                sample["source"] = image
        else:
            anomaly_id = random.randint(0, self.anomaly_num - 1)
            img_path = self.img_paths[anomaly_id][
                idx % len(self.mask_paths[anomaly_id])
            ]
            image = self.loader(Image.open(img_path).convert("RGB"))
            mask_path = self.mask_paths[anomaly_id][
                idx % len(self.mask_paths[anomaly_id])
            ]
            mask = self.loader(Image.open(mask_path).convert("L"))
            mask = (mask > 0.5).float()
            if mask.sum() == 0:
                has_anomaly = np.array([0], dtype=np.float32)
                anomaly_id = -1
            else:
                has_anomaly = np.array([1], dtype=np.float32)
            sample = {
                "image": image,
                "has_anomaly": has_anomaly,
                "mask": mask,
                "anomaly_id": anomaly_id,
            }
            if self.recon:
                img_path = self.img_paths[anomaly_id][
                    idx % len(self.mask_paths[anomaly_id])
                ]
                img_path = img_path.replace("image", "recon")
                ori_image = self.loader(Image.open(img_path).convert("RGB"))
                sample["source"] = ori_image
        return sample


class MVTec_classification_train(Dataset):
    def __init__(self, args, sample_name):
        self.root_dir = os.path.join(args.generated_data_path, sample_name)
        self.anomaly_names = os.listdir(self.root_dir)
        self.img_paths = []
        self.labels = []
        for idx, anomaly in enumerate(self.anomaly_names):
            anomaly_img_path = os.path.join(self.root_dir, anomaly)
            if not os.path.exists(anomaly_img_path):
                print(f"Warning: Image path {anomaly_img_path} does not exist")
                continue

            img_files = [
                f
                for f in os.listdir(anomaly_img_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]

            if not img_files:
                print(f"Warning: No images found for {anomaly}")
                continue

            for img_file in img_files:
                self.img_paths.append(os.path.join(anomaly_img_path, img_file))
                self.labels.append(idx)

            print(f"Loaded {len(img_files)} images from {anomaly}")
        self.loader = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize([256, 256])]
        )
        self.length = len(self.img_paths)

    def __len__(self):
        return self.length * 5

    def class_num(self):
        return len(self.anomaly_names)

    def return_anomaly_names(self):
        return self.anomaly_names

    def __getitem__(self, idx):
        image = self.loader(
            Image.open(self.img_paths[idx % len(self.img_paths)]).convert("RGB")
        )
        label = self.labels[idx % len(self.img_paths)]
        return image, label


class MVTec_classification_test(Dataset):
    def __init__(self, args, sample_name, anomaly_names, seed=42):
        set_seed(seed)
        root_dir = os.path.join(args.mvtec_path, sample_name, "test")
        self.anomaly_names = anomaly_names
        self.img_paths = []
        self.labels = []
        for idx, anomaly_name in enumerate(self.anomaly_names):
            img_path = os.path.join(root_dir, anomaly_name)
            img_files = os.listdir(img_path)
            img_files.sort(key=lambda x: int(x[:3]))
            self.img_paths += [
                os.path.join(img_path, file_name) for file_name in img_files
            ]
            self.labels += [idx for _ in img_files]
        self.loader = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize([256, 256])]
        )
        self.length = len(self.img_paths)

    def __len__(self):
        return self.length

    def class_num(self):
        return len(self.anomaly_names)

    def __getitem__(self, idx):
        image = self.loader(
            Image.open(self.img_paths[idx % len(self.img_paths)]).convert("RGB")
        )
        label = self.labels[idx % len(self.img_paths)]
        return image, label
