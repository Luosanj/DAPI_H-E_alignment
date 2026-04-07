import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract only image-cropping functionality from alignment workflow."
    )
    parser.add_argument(
        "--image_path",
        type=Path,
        required=True,
        help="Path to the source image (e.g., H&E or DAPI image).",
    )
    parser.add_argument(
        "--centers_csv",
        type=Path,
        required=True,
        help="CSV containing crop centers with columns: x,y",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write cropped patches.",
    )
    parser.add_argument(
        "--crop_radius_pixel",
        type=int,
        default=400,
        help="Half-width/half-height of crop patch in pixels.",
    )
    parser.add_argument(
        "--center_move_pixel",
        type=int,
        default=300,
        help="Pixel offset used to create additional shifted crops.",
    )
    parser.add_argument(
        "--crop_image_resize",
        type=int,
        default=224,
        help="Resize each crop patch to this square size.",
    )
    parser.add_argument(
        "--keep_original_size",
        action="store_true",
        help="Do not resize patches.",
    )
    return parser.parse_args()


def load_centers(csv_path: Path) -> np.ndarray:
    centers = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    centers = np.atleast_2d(centers)
    if centers.shape[1] < 2:
        raise ValueError("centers_csv must include at least two columns: x,y")
    return centers[:, :2].astype(int)


def crop_with_padding(
    image_np: np.ndarray,
    center_xy: Tuple[int, int],
    crop_radius: int,
) -> np.ndarray:
    h, w = image_np.shape[:2]
    cx, cy = center_xy

    x0, x1 = cx - crop_radius, cx + crop_radius
    y0, y1 = cy - crop_radius, cy + crop_radius

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image_np = np.pad(
            image_np,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

    return image_np[y0:y1, x0:x1]


def center_variants(center_xy: Tuple[int, int], delta: int) -> List[Tuple[int, int]]:
    cx, cy = center_xy
    return [
        (cx, cy),
        (cx + delta, cy),
        (cx - delta, cy),
        (cx, cy + delta),
        (cx, cy - delta),
    ]


def save_crops(
    image_np: np.ndarray,
    centers: Sequence[Tuple[int, int]],
    output_dir: Path,
    crop_radius_pixel: int,
    center_move_pixel: int,
    crop_image_resize: int,
    keep_original_size: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, center in enumerate(centers):
        for j, shifted_center in enumerate(center_variants(center, center_move_pixel)):
            crop = crop_with_padding(image_np, shifted_center, crop_radius_pixel)
            crop_pil = Image.fromarray(crop)
            if not keep_original_size:
                crop_pil = crop_pil.resize((crop_image_resize, crop_image_resize))
            crop_pil.save(output_dir / f"cell_{i:05d}_variant_{j}.png")


def main() -> None:
    args = parse_args()

    image = Image.open(args.image_path).convert("RGB")
    image_np = np.array(image)

    centers_xy = [tuple(v) for v in load_centers(args.centers_csv)]
    save_crops(
        image_np=image_np,
        centers=centers_xy,
        output_dir=args.output_dir,
        crop_radius_pixel=args.crop_radius_pixel,
        center_move_pixel=args.center_move_pixel,
        crop_image_resize=args.crop_image_resize,
        keep_original_size=args.keep_original_size,
    )


if __name__ == "__main__":
    main()
