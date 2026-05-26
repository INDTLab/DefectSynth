import os
import random
import numpy as np
import cv2
from PIL import Image
from argparse import ArgumentParser

from .diff_morpher_pipeline import DiffMorpherPipeline


def validate_folder_path(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError(
            f"The folder path '{folder_path}' does not exist or is not a directory."
        )
    return folder_path


def select_random_image_from_folder(folder_path):
    image_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]
    if not image_files:
        raise ValueError(f"No image files found in the folder '{folder_path}'.")
    return os.path.join(folder_path, random.choice(image_files))


def get_white_coordinates(image):
    img_array = np.array(image)
    white_coords = np.where(img_array > 50)
    return list(zip(white_coords[0], white_coords[1]))


def resample_coords_to_length(coords, target_len):
    if target_len <= 0 or len(coords) == 0:
        return []
    if len(coords) == 1:
        return [coords[0]] * target_len

    arr = np.array(coords, dtype=np.float32)
    old_idx = np.linspace(0, len(arr) - 1, len(arr), dtype=np.float32)
    new_idx = np.linspace(0, len(arr) - 1, target_len, dtype=np.float32)
    ys = np.interp(new_idx, old_idx, arr[:, 0])
    xs = np.interp(new_idx, old_idx, arr[:, 1])
    return list(zip(ys.astype(np.int32), xs.astype(np.int32)))


def interpolate_coordinates(coords1, coords2, alpha):
    if len(coords1) == 0 or len(coords2) == 0:
        return []

    n = max(len(coords1), len(coords2))
    if len(coords1) < n:
        coords1 = resample_coords_to_length(coords1, n)
    if len(coords2) < n:
        coords2 = resample_coords_to_length(coords2, n)

    interp_coords = []
    for (y1, x1), (y2, x2) in zip(coords1[:n], coords2[:n]):
        y = int((1 - alpha) * y1 + alpha * y2)
        x = int((1 - alpha) * x1 + alpha * x2)
        interp_coords.append((y, x))

    if len(interp_coords) > 0:
        additional_coords = []
        for i in range(len(interp_coords) - 1):
            y1, x1 = interp_coords[i]
            y2, x2 = interp_coords[i + 1]
            dist = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
            if dist > 2:
                steps = int(dist)
                for step in range(1, steps):
                    t = step / steps
                    y = int(y1 + t * (y2 - y1))
                    x = int(x1 + t * (x2 - x1))
                    additional_coords.append((y, x))
        interp_coords.extend(additional_coords)

    return interp_coords


def create_mask_from_coords(coords, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for y, x in coords:
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            mask[y, x] = 255

    kernel = np.ones((5, 5), np.uint8)
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_smooth, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_smooth, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return Image.fromarray(mask)


def count_white_pixels(mask):
    return np.sum(np.array(mask) > 127)


def get_mask_area_ratio(mask: Image.Image, threshold: int = 50) -> float:
    arr = np.array(mask.convert("L"), dtype=np.uint8)
    fg = (arr > threshold).sum()
    total = arr.size
    return float(fg) / float(total) if total > 0 else 0.0


def get_largest_connected_component_area(mask):
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(binary)
    if num_labels <= 1:
        return 0
    largest_area = 0
    for label in range(1, num_labels):
        area = np.sum(labels == label)
        if area > largest_area:
            largest_area = area
    return largest_area


def morph_masks_pair(mask1_path, mask2_path, num_frames, max_area_threshold):
    if not os.path.exists(mask1_path):
        raise FileNotFoundError(f"Mask1 path does not exist: {mask1_path}")
    if not os.path.exists(mask2_path):
        raise FileNotFoundError(f"Mask2 path does not exist: {mask2_path}")

    mask1 = Image.open(mask1_path).convert("L")
    mask2 = Image.open(mask2_path).convert("L")

    if mask1.size != mask2.size:
        mask2 = mask2.resize(mask1.size, Image.NEAREST)

    coords1 = get_white_coordinates(mask1)
    coords2 = get_white_coordinates(mask2)

    frames = []
    shape = np.array(mask1).shape
    first_mask = create_mask_from_coords(coords1, shape)
    last_mask = create_mask_from_coords(coords2, shape)

    if num_frames <= 2:
        return [first_mask, last_mask]

    for i in range(1, num_frames - 1):
        alpha = i / (num_frames - 1)
        interp_coords = interpolate_coordinates(coords1, coords2, alpha)
        mask = create_mask_from_coords(interp_coords, shape)
        largest_area = get_largest_connected_component_area(mask)
        max_area_1 = get_largest_connected_component_area(first_mask)
        max_area_2 = get_largest_connected_component_area(last_mask)
        max_allowed_area = max(max_area_1, max_area_2) * max_area_threshold
        if largest_area <= max_allowed_area:
            frames.append(mask)

    return frames


def infer_category_defect_from_path(path):
    parts = os.path.normpath(path).replace("\\", "/").split("/")
    if "ground_truth" in parts:
        i = parts.index("ground_truth")
        if i - 1 >= 0 and i + 1 < len(parts):
            return parts[i - 1], parts[i + 1]
    if "test" in parts:
        i = parts.index("test")
        if i - 1 >= 0 and i + 1 < len(parts):
            return parts[i - 1], parts[i + 1]
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "unknown", "unknown"


def build_group_index(folder_path_0, folder_path_1):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    files0 = sorted(
        os.path.join(folder_path_0, f)
        for f in os.listdir(folder_path_0)
        if f.lower().endswith(exts)
    )
    files1 = sorted(
        os.path.join(folder_path_1, f)
        for f in os.listdir(folder_path_1)
        if f.lower().endswith(exts)
    )
    pairs = list(zip(files0, files1))
    groups = {}
    for p0, p1 in pairs:
        c0, d0 = infer_category_defect_from_path(p0)
        c1, d1 = infer_category_defect_from_path(p1)
        category = c0 if c0 != "unknown" else c1
        defect = d0 if d0 != "unknown" else d1
        key = (category, defect)
        groups.setdefault(key, []).append((p0, p1))
    return groups


def compute_group_mean_area_ratio(pairs, threshold):
    ratios = []
    for p0, p1 in pairs:
        m0 = Image.open(p0).convert("L")
        m1 = Image.open(p1).convert("L")
        ratios.append(get_mask_area_ratio(m0, threshold=threshold))
        ratios.append(get_mask_area_ratio(m1, threshold=threshold))
    return float(np.mean(ratios)) if ratios else 0.0


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="./stabilityai/stable-diffusion-2-1-base",
    )
    parser.add_argument(
        "--folder_path_0",
        type=str,
        default=None,
        help="Folder containing images for source",
    )
    parser.add_argument(
        "--folder_path_1",
        type=str,
        default=None,
        help="Folder containing images for target",
    )
    parser.add_argument(
        "--image_path_0",
        type=str,
        default=None,
        help="Path to a single source image",
    )
    parser.add_argument(
        "--image_path_1",
        type=str,
        default=None,
        help="Path to a single target image",
    )
    parser.add_argument("--prompt_0", type=str, default="")
    parser.add_argument("--prompt_1", type=str, default="")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--duration", type=int, default=100)
    parser.add_argument("--use_adain", action="store_true")
    parser.add_argument("--use_reschedule", action="store_true")
    parser.add_argument("--save_inter", action="store_true")
    parser.add_argument("--max_area_threshold", type=float, default=2.0)
    parser.add_argument(
        "--target_total",
        type=int,
        default=1000,
        help="Total number of valid masks to save",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="Threshold for area ratio in auto mode",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["latent", "morph", "auto"],
        default="auto",
        help="latent, morph, or auto",
    )
    parser.add_argument(
        "--auto_tau",
        type=float,
        default=0.053,
        help="If group mean area ratio >= tau, use latent; else morph",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    pipeline = None

    def get_pipeline():
        nonlocal pipeline
        if pipeline is None:
            print("Loading model...", args.model_path)
            pipeline = DiffMorpherPipeline.from_pretrained(args.model_path)
            pipeline.to("cuda")
        return pipeline

    group_mode = None
    if args.mode == "auto" and args.folder_path_0 and args.folder_path_1:
        groups = build_group_index(args.folder_path_0, args.folder_path_1)
        group_mode = {}
        for key, pairs in groups.items():
            mean_ratio = compute_group_mean_area_ratio(pairs, threshold=args.threshold)
            chosen = "latent" if mean_ratio >= args.auto_tau else "morph"
            group_mode[key] = chosen
            print(
                f"Auto group decision: category={key[0]}, defect={key[1]}, "
                f"Abar={mean_ratio:.4f}, Tau={args.auto_tau} -> selected {chosen}"
            )

    def decide_mode(p0, p1):
        if args.mode != "auto":
            return args.mode
        c0, d0 = infer_category_defect_from_path(p0)
        c1, d1 = infer_category_defect_from_path(p1)
        category = c0 if c0 != "unknown" else c1
        defect = d0 if d0 != "unknown" else d1
        key = (category, defect)
        if group_mode is not None and key in group_mode:
            chosen = group_mode[key]
            print(
                f"Auto mode (group): category={category}, defect={defect}, "
                f"Tau={args.auto_tau} -> selected {chosen}"
            )
            return chosen
        print(
            f"Auto mode fallback: no group stats for category={category}, "
            f"defect={defect}. Using morph."
        )
        return "morph"

    if args.image_path_0 is not None and args.image_path_1 is not None:
        if args.mode == "auto" and group_mode is None:
            print(
                "Auto mode: group stats unavailable in single-pair mode without "
                "--folder_path_0/--folder_path_1. Using morph."
            )
        if not os.path.exists(args.image_path_0):
            raise FileNotFoundError(f"image_path_0 does not exist: {args.image_path_0}")
        if not os.path.exists(args.image_path_1):
            raise FileNotFoundError(f"image_path_1 does not exist: {args.image_path_1}")

        run_mode = decide_mode(args.image_path_0, args.image_path_1)

        if run_mode == "latent":
            images, mask_frames = get_pipeline()(
                img_path_0=args.image_path_0,
                img_path_1=args.image_path_1,
                prompt_0=args.prompt_0,
                prompt_1=args.prompt_1,
                use_adain=args.use_adain,
                use_reschedule=args.use_reschedule,
                lamd=0.6,
                output_path=args.output_path,
                num_frames=args.num_frames,
                save_intermediates=args.save_inter,
                use_lora=False,
                potential_interpolation=True,
                position_interpolation=False,
                start_n=0,
            )
            frames = images if images is not None else mask_frames or []
        else:
            frames = morph_masks_pair(
                args.image_path_0,
                args.image_path_1,
                args.num_frames,
                args.max_area_threshold,
            )

        valid_frames = [m for m in frames if count_white_pixels(m) > 0]
        for i, mask in enumerate(valid_frames):
            mask.save(os.path.join(args.output_path, f"{i:03d}_mask.png"))
        print(f"Done. Saved {len(valid_frames)} frames to {args.output_path}")
        return

    if not args.folder_path_0 or not args.folder_path_1:
        raise ValueError(
            "Either specify --image_path_0/--image_path_1 for single pair, "
            "or --folder_path_0/--folder_path_1 for batch mode"
        )

    folder_path_0 = validate_folder_path(args.folder_path_0)
    folder_path_1 = validate_folder_path(args.folder_path_1)

    total_saved_images = 0
    while total_saved_images < args.target_total:
        image_path_0 = select_random_image_from_folder(folder_path_0)
        image_path_1 = select_random_image_from_folder(folder_path_1)
        run_mode = decide_mode(image_path_0, image_path_1)
        print(f"Selected image from folder 0: {image_path_0}")
        print(f"Selected image from folder 1: {image_path_1}")

        if run_mode == "latent":
            images, mask_frames = get_pipeline()(
                img_path_0=image_path_0,
                img_path_1=image_path_1,
                prompt_0=args.prompt_0,
                prompt_1=args.prompt_1,
                use_adain=args.use_adain,
                use_reschedule=args.use_reschedule,
                lamd=0.6,
                output_path=args.output_path,
                num_frames=args.num_frames,
                save_intermediates=args.save_inter,
                use_lora=False,
                potential_interpolation=True,
                position_interpolation=False,
                start_n=0,
            )
            frames = images if images is not None else mask_frames or []
        else:
            frames = morph_masks_pair(
                image_path_0,
                image_path_1,
                args.num_frames,
                args.max_area_threshold,
            )

        valid_frames = [m for m in frames if count_white_pixels(m) > 0]
        for i, mask in enumerate(valid_frames, start=total_saved_images):
            mask.save(os.path.join(args.output_path, f"{i:03d}_mask.png"))
        total_saved_images += len(valid_frames)


if __name__ == "__main__":
    main()
