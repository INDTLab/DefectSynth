import argparse
import os
import random

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision import transforms

import cldm.bootstrap
from annotator.util import HWC3, resize_image
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim_wm2 import DDIMSampler


def batched_index_select(input_tensor, dim, index):
    views = [input_tensor.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input_tensor.shape))]
    expanse = list(input_tensor.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    return torch.gather(input_tensor, dim, index)


def batched_scatter(input_tensor, dim, index, src):
    views = [input_tensor.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input_tensor.shape))]
    expanse = list(input_tensor.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    return torch.scatter(input_tensor, dim, index, src)


class LocalFusionModule(nn.Module):
    def __init__(self, inplanes, rate):
        super(LocalFusionModule, self).__init__()
        self.W = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inplanes),
        )
        self.rate = rate

    def forward(self, feat, refs, index, similarity, base_similarity, ref_similarities):
        num_refs = refs.size(1)
        device = feat.device
        if num_refs == 1:
            refs = feat.unsqueeze(1)
        b = feat.size(0)
        base_similarity = torch.full((b, 1, 1), float(base_similarity), device=device)
        ref_similarities = torch.full((b, 1, num_refs), float(ref_similarities), device=device)

        b, n, c, h, w = refs.size()
        refs = refs.view(b * n, c, h, w)

        w_feat = feat.view(b, c, -1).permute(0, 2, 1).contiguous()
        w_feat = F.normalize(w_feat, dim=2)

        w_refs = refs.view(b, n, c, -1).permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        w_refs = F.normalize(w_refs, dim=1)

        rate = self.rate
        num_points = int(rate * h * w)
        num_points = max(num_points, 1)
        feat_indices = torch.cat(
            [torch.LongTensor(random.sample(range(h * w), num_points)).unsqueeze(0) for _ in range(b)],
            dim=0,
        ).to(device)

        feat = feat.view(b, c, -1)
        feat_select = batched_index_select(feat, dim=2, index=feat_indices)
        w_feat_select = batched_index_select(w_feat, dim=1, index=feat_indices)
        w_feat_select = F.normalize(w_feat_select, dim=2)

        refs = refs.view(b, n, c, h * w)
        ref_indices = []
        ref_selects = []
        for j in range(n):
            ref = refs[:, j, :, :]
            w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]
            fx = torch.matmul(w_feat_select, w_ref)
            _, indice = torch.topk(fx, dim=2, k=1)
            indice = indice.squeeze(0).squeeze(-1)
            select = batched_index_select(ref, dim=2, index=indice)
            ref_indices.append(indice)
            ref_selects.append(select)

        ref_indices = torch.cat([item.unsqueeze(1) for item in ref_indices], dim=1)
        ref_selects = torch.cat([item.unsqueeze(1) for item in ref_selects], dim=1)

        base_similarity = base_similarity.view(b, 1, 1)
        ref_similarities = ref_similarities.view(b, 1, n)
        feat_select = feat_select.view(b, 1, -1)
        ref_selects = ref_selects.view(b, n, -1)

        feat_fused = torch.matmul(base_similarity, feat_select) + torch.matmul(ref_similarities, ref_selects)
        feat_fused = feat_fused.view(b, c, num_points)
        feat = batched_scatter(feat, dim=2, index=feat_indices, src=feat_fused)
        feat = feat.view(b, c, h, w)
        return feat, feat_indices, ref_indices


def hist_matching(img2, img1, alpha):
    alpha = float(alpha)
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1.")

    hist_img2, bins_img2 = np.histogram(img2.flatten(), 256, range=(0, 255))
    hist_img1, bins_img1 = np.histogram(img1.flatten(), 256, range=(0, 255))
    cdf_img2 = hist_img2.cumsum()
    cdf_img2 = cdf_img2 / cdf_img2[-1]
    cdf_img1 = hist_img1.cumsum()
    cdf_img1 = cdf_img1 / cdf_img1[-1]
    img2_cdf = np.interp(img2.flatten(), bins_img2[:-1], cdf_img2)
    img2_cdf = np.interp(img2_cdf, cdf_img1, bins_img1[:-1])
    img2_cdf = alpha * img2_cdf + (1 - alpha) * img2.flatten()
    return img2_cdf.reshape(img2.shape).astype(np.uint8)


def mask_filename_to_output_png(mask_filename: str) -> str:
    base = os.path.splitext(os.path.basename(mask_filename))[0]
    if base.endswith("_mask"):
        base = base[: -len("_mask")]
    return base + ".png"


def get_prefix_from_mask_filename(mask_filename: str) -> str:
    base = os.path.splitext(os.path.basename(mask_filename))[0]
    if base.endswith("_mask"):
        return base[: -len("_mask")]
    return base


def parse_jobs(txt_path):
    jobs = []
    with open(txt_path, "rt", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("+", 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid line in txt: {line}")
            sample_name, defect_name, prompt = [x.strip() for x in parts]
            if not sample_name or not defect_name or not prompt:
                raise ValueError(f"Invalid empty field in line: {line}")
            jobs.append((sample_name, defect_name, prompt))
    return jobs


def build_paths(sample_name, defect_name):
    target_dir = os.path.join("./outputs/defects_gen/MVTec_AD", sample_name, defect_name)
    mask_dir = os.path.join("./outputs/masks_gen/MVTec_AD", sample_name, defect_name)
    good_img_dir = os.path.join("./training/MVTec_AD", sample_name, "train", "good")
    return target_dir, mask_dir, good_img_dir


def prepare_real_mask_and_image(
    mask_file,
    real_mask_dir,
    real_image_dir,
    real_mask_files,
    real_image_files,
    device,
    num_samples,
    model,
):
    real_mask_tensor = None
    real_image_tensor = None
    mask_prefix_to_file = {}
    image_prefix_to_file = {}
    for f in real_mask_files:
        if not f.lower().endswith(".png"):
            continue
        prefix = get_prefix_from_mask_filename(f)
        mask_prefix_to_file[prefix] = f
    for f in real_image_files:
        if not f.lower().endswith(".png"):
            continue
        prefix = os.path.splitext(f)[0]
        image_prefix_to_file[prefix] = f

    common_prefixes = sorted(set(mask_prefix_to_file.keys()) & set(image_prefix_to_file.keys()))
    print(
        f"[wm2] prefix candidates: mask={len(mask_prefix_to_file)}, "
        f"image={len(image_prefix_to_file)}, common={len(common_prefixes)}"
    )
    if not common_prefixes:
        print("[wm2] no common prefixes, skip wm2 references for this sample.")
        return real_mask_tensor, real_image_tensor

    current_prefix = get_prefix_from_mask_filename(mask_file)
    if current_prefix in common_prefixes:
        chosen_prefix = current_prefix
    else:
        chosen_prefix = random.choice(common_prefixes)
    print(f"[wm2] chosen_prefix={chosen_prefix} (current_mask_prefix={current_prefix})")

    real_mask_path = os.path.join(real_mask_dir, mask_prefix_to_file[chosen_prefix])
    real_image_path = os.path.join(real_image_dir, image_prefix_to_file[chosen_prefix])

    if os.path.exists(real_mask_path):
        real_mask = cv2.imread(real_mask_path)
        real_mask = HWC3(real_mask)
        real_mask = resize_image(real_mask, 512)
        real_mask_t = torch.from_numpy(real_mask).permute(2, 0, 1)
        weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)
        real_mask_t = torch.sum(real_mask_t * weights, dim=0, keepdim=True)
        real_mask_t[real_mask_t < 127.5] = 0
        real_mask_t[real_mask_t >= 127.5] = 1
        n = real_mask.shape[0] // 8
        resize_n = transforms.Resize((n, n), interpolation=transforms.InterpolationMode.NEAREST)
        real_mask_t = resize_n(real_mask_t)
        real_mask_tensor = torch.tile(real_mask_t, (num_samples, 1, 1, 1)).to(device)

    if os.path.exists(real_image_path):
        real_img = cv2.imread(real_image_path)
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        real_img = HWC3(real_img)
        real_img = resize_image(real_img, 512)
        to_tensor = transforms.ToTensor()
        real_img_t = to_tensor(real_img).unsqueeze(0).to(device)
        encoder_posterior = model.encode_first_stage(real_img_t)
        real_image_tensor = model.get_first_stage_encoding(encoder_posterior).detach()

    return real_mask_tensor, real_image_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_path",
        type=str,
        default="./lightning_logs/MVTec_AD/model-epoch=1999-val_loss=0.00.ckpt",
    )
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--txt_path", type=str, required=True, help="Path to txt: sample+defect+prompt per line")
    parser.add_argument("--use_wm2", action="store_true", help="Enable weight_map2 using real mask/image references", default=True)
    parser.add_argument("--num_samples", type=int, default=2, help="LFM batch size for fusion")
    parser.add_argument("--lfm_rate", type=float, default=0.2, help="LFM local mixing rate")
    parser.add_argument("--base_similarities", type=float, default=0.7, help="Base feature weight in LFM")
    parser.add_argument("--ref_similarities", type=float, default=0.3, help="Reference feature weight in LFM")
    args = parser.parse_args()

    jobs = parse_jobs(args.txt_path)
    if not jobs:
        raise ValueError("No valid jobs found in txt_path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = args.num_samples
    a_prompt = "best quality"
    eta = 1.0
    scale = 9.0
    lfm_rate = args.lfm_rate

    model = create_model("./models/cldm_v15.yaml").cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location=str(device)), strict=False)
    model = model.to(device)
    ddim_sampler = DDIMSampler(model)
    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for sample_name, defect_name, prompt in jobs:
            target_dir, mask_dir, good_img_dir = build_paths(sample_name, defect_name)
            real_mask_dir = os.path.join("./training/MVTec_AD", sample_name, "ground_truth", defect_name)
            real_image_dir = os.path.join("./training/MVTec_AD", sample_name, "test", defect_name)
            os.makedirs(target_dir, exist_ok=True)

            if not os.path.isdir(mask_dir):
                print(f"Skip {sample_name}/{defect_name}: mask_dir not found -> {mask_dir}")
                continue
            if not os.path.isdir(good_img_dir):
                print(f"Skip {sample_name}/{defect_name}: good_img_dir not found -> {good_img_dir}")
                continue

            mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(".png")])
            good_img_files = [f for f in os.listdir(good_img_dir) if f.lower().endswith(".png")]
            if not mask_files or not good_img_files:
                print(f"Skip {sample_name}/{defect_name}: no masks or good images.")
                continue
            real_mask_files = os.listdir(real_mask_dir) if os.path.exists(real_mask_dir) else []
            real_image_files = os.listdir(real_image_dir) if os.path.exists(real_image_dir) else []

            print(f"Generating {sample_name}/{defect_name}, masks={len(mask_files)}")
            for mask_file in mask_files:
                source_path = os.path.join(mask_dir, mask_file)
                source = cv2.imread(source_path)
                if source is None:
                    print(f"Skip unreadable mask: {source_path}")
                    continue

                source = HWC3(source)
                detected_map = source.copy()
                source = resize_image(source, 512)
                mask_result = source.copy()
                h, w, _ = source.shape
                detected_map = cv2.resize(detected_map, (w, h), interpolation=cv2.INTER_LINEAR)

                control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
                control = torch.stack([control for _ in range(num_samples)], dim=0)
                control = einops.rearrange(control, "b h w c -> b c h w").clone()

                seed_everything(random.randint(0, 65535))
                cond = {
                    "c_concat": [control],
                    "c_crossattn": [model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)],
                }
                shape = (4, h // 8, w // 8)
                model.control_scales = [1.0] * 13

                mask = torch.from_numpy(source).permute(2, 0, 1)
                weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)
                mask = torch.sum(mask * weights, dim=0, keepdim=True)
                n = source.shape[0] // 8
                resize_n = transforms.Resize((n, n))
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1
                mask = resize_n(mask)
                mask = torch.tile(mask, (num_samples, 1, 1, 1)).to(device)

                good_img_file = random.choice(good_img_files)
                good_img_path = os.path.join(good_img_dir, good_img_file)
                x0 = cv2.imread(good_img_path)
                if x0 is None:
                    print(f"Skip unreadable good image: {good_img_path}")
                    continue
                x0 = x0[:, :, ::-1]
                x0 = HWC3(x0)
                x0 = resize_image(x0, 512)
                good_img = np.tile(x0, (num_samples, 1, 1, 1))
                x0 = to_tensor(x0).to(device).unsqueeze(0)

                encoder_posterior = model.encode_first_stage(x0)
                z = model.get_first_stage_encoding(encoder_posterior).detach()

                real_mask_tensor, real_image_tensor = None, None
                if args.use_wm2:
                    real_mask_tensor, real_image_tensor = prepare_real_mask_and_image(
                        mask_file=mask_file,
                        real_mask_dir=real_mask_dir,
                        real_image_dir=real_image_dir,
                        real_mask_files=real_mask_files,
                        real_image_files=real_image_files,
                        device=device,
                        num_samples=num_samples,
                        model=model,
                    )

                samples, _ = ddim_sampler.sample(
                    args.ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=eta,
                    mask=mask,
                    x0=z,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=None,
                    use_weight_map=True,
                    real_mask=real_mask_tensor,
                    real_image=real_image_tensor,
                    adaptive_real_image=args.use_wm2,
                )

                samples = samples.to(device)
                batch_size, channels, _, _ = samples.shape
                local_fusion = LocalFusionModule(channels, float(lfm_rate)).to(device)
                similarity = torch.full((1, batch_size), 0.0, device=device)
                index = torch.randint(0, batch_size, (), device=device)
                base_similarities = args.base_similarities
                ref_similarities = args.ref_similarities

                new_samples = torch.zeros(batch_size, *samples.shape[1:], dtype=samples.dtype, device=device)
                for i in range(batch_size):
                    feat = samples[i : i + 1].clone().to(device)
                    refs = torch.cat([samples[0:i], samples[i + 1 :]], dim=0).to(device) if batch_size > 1 else feat.clone()
                    refs = refs.unsqueeze(0)
                    fused_feat, _, _ = local_fusion(feat, refs, index, similarity, base_similarities, ref_similarities)
                    new_samples[i] = fused_feat.squeeze(0)

                # keep untouched regions stable
                samples = new_samples * mask + (1 - mask) * samples

                x_samples = model.decode_first_stage(samples)
                x_samples = (
                    einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5
                ).cpu().numpy().clip(0, 255).astype(np.uint8)
                x_samples = x_samples[:, :, :, ::-1]
                x_samples = hist_matching(x_samples, good_img, alpha=1.0)

                out_img = x_samples[0]
                save_path = os.path.join(target_dir, mask_filename_to_output_png(mask_file))
                cv2.imwrite(save_path, out_img)

            print(f"Done {sample_name}/{defect_name} -> {target_dir}")


if __name__ == "__main__":
    main()
