import cldm.bootstrap

import argparse
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from validate_dataset import ControlNetPairDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ""
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ""
    return True, name[len(parent_name):]


def build_control_ckpt_if_needed(base_ckpt, control_ckpt, config_path):
    if os.path.exists(control_ckpt):
        print(f"Control ckpt already exists, skip building: {control_ckpt}")
        return

    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"Base ckpt does not exist: {base_ckpt}")

    out_dir = os.path.dirname(control_ckpt) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"Building control ckpt from {base_ckpt} -> {control_ckpt}")
    model = create_model(config_path=config_path)

    pretrained_weights = torch.load(base_ckpt, map_location="cpu")
    if "state_dict" in pretrained_weights:
        pretrained_weights = pretrained_weights["state_dict"]

    scratch_dict = model.state_dict()
    target_dict = {}
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, "control_")
        copy_k = "model.diffusion_" + name if is_control else k
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f"These weights are newly added: {k}")

    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), control_ckpt)
    print(f"Control ckpt saved: {control_ckpt}")


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train ControlNet on defect image pairs.")
    parser.add_argument("--copy_from_ckpt", type=str, default="./models/v1-5-pruned.ckpt")
    parser.add_argument("--copy_to_ckpt", type=str, default="./models/control_sd15_MVTec_AD.ckpt")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--save_every_n_epochs", type=int, default=200)
    parser.add_argument("--logger_freq", type=int, default=None)
    parser.add_argument("--dirpath", type=str, default="./lightning_logs/MVTec_AD")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=2000)

    parser.add_argument("--config_path", type=str, default="./models/cldm_v15.yaml")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--sd_locked", type=int, choices=[0, 1], default=1)
    parser.add_argument("--only_mid_control", type=int, choices=[0, 1], default=0)

    parser.add_argument(
        "--prompt_json_path",
        type=str,
        default="./training/prompt.json",
        help="Jsonl path.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./training/MVTec_AD",
        help="Dataset root.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.logger_freq is None:
        args.logger_freq = args.save_every_n_epochs

    build_control_ckpt_if_needed(
        base_ckpt=args.copy_from_ckpt,
        control_ckpt=args.copy_to_ckpt,
        config_path=args.config_path,
    )

    config = OmegaConf.load(args.config_path)
    model = load_model_from_config(config, args.copy_from_ckpt)
    model.load_state_dict(load_state_dict(args.copy_to_ckpt, location="cpu"), strict=False)
    model.learning_rate = args.learning_rate
    model.sd_locked = bool(args.sd_locked)
    model.only_mid_control = bool(args.only_mid_control)

    dataset = ControlNetPairDataset(
        prompt_json_path=args.prompt_json_path,
        data_root=args.data_root,
    )
    print(
        f"Dataset: prompt={os.path.abspath(args.prompt_json_path)} | "
        f"data_root={dataset.data_root} | samples={len(dataset)}"
    )
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=args.logger_freq)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.dirpath,
        filename="model-{epoch:04d}",
        every_n_epochs=args.save_every_n_epochs,
        save_top_k=-1,
        save_last=True,
    )

    trainer = pl.Trainer(
        gpus=[args.gpu],
        precision=32,
        callbacks=[logger, checkpoint_callback],
        max_epochs=args.max_epochs,
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
