import os
import random
import tempfile
from argparse import ArgumentParser

from PIL import Image

from morph.image_morphing import (
    DiffMorpherPipeline,
    count_white_pixels,
    get_mask_area_ratio,
    morph_masks_pair,
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


def list_image_files(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    ]


def discover_mvtec_ground_truth(root_folder):
    targets = []
    for sample_name in sorted(os.listdir(root_folder)):
        sample_path = os.path.join(root_folder, sample_name)
        if not os.path.isdir(sample_path):
            continue
        gt_dir = os.path.join(sample_path, "ground_truth")
        if not os.path.isdir(gt_dir):
            continue
        for defect_name in sorted(os.listdir(gt_dir)):
            defect_folder = os.path.join(gt_dir, defect_name)
            if os.path.isdir(defect_folder):
                targets.append((sample_name, defect_name, defect_folder))
    return targets


def choose_two_different_images(image_paths):
    if len(image_paths) < 2:
        return None, None
    p0, p1 = random.sample(image_paths, 2)
    return p0, p1


def get_mean_area_ratio(image_paths, threshold):
    ratios = []
    for p in image_paths:
        mask = Image.open(p).convert("L")
        ratios.append(get_mask_area_ratio(mask, threshold=threshold))
    return sum(ratios) / len(ratios) if ratios else 0.0


def main():
    parser = ArgumentParser(
        description="Generate interpolated masks for MVTec defects."
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to MVTec dataset root.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output root",
    )
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["latent", "morph", "auto"],
        default="auto",
        help="Interpolation mode.",
    )
    parser.add_argument(
        "--auto_tau",
        type=float,
        default=0.053,
        help="Threshold used in auto mode.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="Mask threshold.",
    )
    parser.add_argument(
        "--target_total",
        type=int,
        default=1000,
        help="Max masks per defect type.",
    )
    parser.add_argument(
        "--max_area_threshold",
        type=float,
        default=2.0,
        help="Area constraint for morph mode.",
    )
    parser.add_argument("--prompt_0", type=str, default="")
    parser.add_argument("--prompt_1", type=str, default="")
    parser.add_argument("--use_adain", action="store_true")
    parser.add_argument("--use_reschedule", action="store_true")
    parser.add_argument("--save_inter", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        raise ValueError(f"Invalid folder_path: {args.folder_path}")
    os.makedirs(args.output_path, exist_ok=True)

    # lazy init to avoid loading sd unless needed
    pipeline = None

    def get_pipeline():
        nonlocal pipeline
        if pipeline is None:
            pipeline = DiffMorpherPipeline.from_pretrained("stable-diffusion-v1-5")
            pipeline.to("cuda")
        return pipeline

    targets = discover_mvtec_ground_truth(args.folder_path)
    if not targets:
        raise ValueError("No valid sample/ground_truth/defect folders found.")

    for sample_name, defect_name, defect_folder in targets:
        image_paths = list_image_files(defect_folder)
        if len(image_paths) < 2:
            print(f"Skip {sample_name}/{defect_name}: fewer than 2 images.")
            continue

        out_dir = os.path.join(args.output_path, sample_name, defect_name)
        os.makedirs(out_dir, exist_ok=True)

        if args.mode == "auto":
            mean_ratio = get_mean_area_ratio(image_paths, args.threshold)
            defect_mode = "latent" if mean_ratio >= args.auto_tau else "morph"
            print(
                f"[{sample_name}/{defect_name}] auto: mean_area_ratio={mean_ratio:.4f}, "
                f"tau={args.auto_tau} -> {defect_mode}"
            )
        else:
            defect_mode = args.mode
            print(f"[{sample_name}/{defect_name}] mode: {defect_mode}")

        saved = 0
        while saved < args.target_total:
            image_path_0, image_path_1 = choose_two_different_images(image_paths)
            if image_path_0 is None:
                break

            if defect_mode == "latent":
                with tempfile.TemporaryDirectory(prefix="morph_pipeline_") as pipeline_out:
                    images, mask_frames = get_pipeline()(
                        img_path_0=image_path_0,
                        img_path_1=image_path_1,
                        prompt_0=args.prompt_0,
                        prompt_1=args.prompt_1,
                        use_adain=args.use_adain,
                        use_reschedule=args.use_reschedule,
                        lamd=0.6,
                        output_path=pipeline_out,
                        num_frames=args.num_frames,
                        save_intermediates=args.save_inter,
                        use_lora=False,
                        potential_interpolation=True,
                        position_interpolation=False,
                        start_n=0,
                    )
                    frames = images if images is not None else (mask_frames or [])
            else:
                frames = morph_masks_pair(
                    image_path_0,
                    image_path_1,
                    args.num_frames,
                    args.max_area_threshold,
                )

            valid_frames = [m for m in frames if count_white_pixels(m) > 0]
            for mask in valid_frames:
                if saved >= args.target_total:
                    break
                out_path = os.path.join(out_dir, f"{saved:03d}_mask.png")
                mask.save(out_path)
                saved += 1

        print(f"[{sample_name}/{defect_name}] saved {saved} masks -> {out_dir}")


if __name__ == "__main__":
    main()
