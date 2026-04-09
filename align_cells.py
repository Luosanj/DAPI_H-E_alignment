#!/usr/bin/env python3
"""Cell-to-cell alignment between H&E and DAPI images.

Key design points:
- H&E and DAPI are assumed to share the same pixel coordinate system.
- For each H&E cell, DAPI candidates are searched inside a coordinate radius.
- Patch similarity is measured on 5 directional crops (center/up/down/left/right).
- Supports multiple 2D patch metrics (NCC/cosine/SSIM) instead of only flattened cosine.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("OpenCV (cv2) is required. Install with: pip install opencv-python") from exc

Direction = Tuple[int, int]


@dataclass(frozen=True)
class Cell:
    cell_id: str
    x: float
    y: float
    radius: float | None = None


@dataclass
class MatchResult:
    he_cell_id: str
    dapi_cell_id: str
    aggregated_score: float
    direction_scores: Dict[str, float]
    metric: str
    patch_radius_used: int
    he_xy: Tuple[float, float]
    dapi_xy: Tuple[float, float]


DEFAULT_DIRECTIONS: Dict[str, Direction] = {
    "center": (0, 0),
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def read_cells_csv(
    path: Path,
    id_col: str,
    x_col: str,
    y_col: str,
    radius_col: str | None,
) -> List[Cell]:
    """Read cell centroids from CSV.

    Handles CSVs with an unnamed first index column (e.g., header starts with ',x_center_pixel,...').
    """
    cells: List[Cell] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if id_col in row and row[id_col] not in (None, ""):
                cell_id = str(row[id_col])
            elif "" in row and row[""] not in (None, ""):
                cell_id = str(row[""])
            else:
                cell_id = str(row_idx)

            radius: float | None = None
            if radius_col and radius_col in row and row[radius_col] not in (None, ""):
                radius = float(row[radius_col])

            cells.append(Cell(cell_id=cell_id, x=float(row[x_col]), y=float(row[y_col]), radius=radius))
    return cells


def to_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image is None (load failed).")
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def extract_patch(image: np.ndarray, center_xy: Tuple[float, float], radius: int) -> np.ndarray | None:
    x, y = center_xy
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = cx - radius, cx + radius + 1
    y0, y1 = cy - radius, cy + radius + 1
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        return None
    return image[y0:y1, x0:x1]


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    arr = patch.astype(np.float32)
    return (arr - arr.mean()) / (arr.std() + 1e-6)


def cosine_similarity_2d(a: np.ndarray, b: np.ndarray) -> float:
    va = a.reshape(-1)
    vb = b.reshape(-1)
    denom = np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8
    return float(np.dot(va, vb) / denom)


def ncc_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation in 2D (equivalent to cosine on z-scored vectors)."""
    an = normalize_patch(a)
    bn = normalize_patch(b)
    return float(np.mean(an * bn))


def ssim_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Single-window SSIM over the patch.

    Returns value approximately in [-1, 1], where larger is more similar.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    mu_a, mu_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var()), float(b.var())
    cov_ab = float(((a - mu_a) * (b - mu_b)).mean())

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return float(num / (den + 1e-8))


def patch_similarity(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        return cosine_similarity_2d(normalize_patch(a), normalize_patch(b))
    if metric == "ncc":
        return ncc_similarity(a, b)
    if metric == "ssim":
        return ssim_similarity(a, b)
    raise ValueError(f"Unsupported metric: {metric}")


def per_pair_patch_radius(
    he_cell: Cell,
    fixed_patch_radius: int,
    use_cell_radius: bool,
    radius_scale: float,
    min_patch_radius: int,
    max_patch_radius: int,
) -> int:
    if not use_cell_radius or he_cell.radius is None:
        return fixed_patch_radius
    dynamic_radius = int(round(he_cell.radius * radius_scale))
    return max(min_patch_radius, min(max_patch_radius, dynamic_radius))


def multi_direction_similarity(
    he_img: np.ndarray,
    dapi_img: np.ndarray,
    he_xy: Tuple[float, float],
    dapi_xy: Tuple[float, float],
    patch_radius: int,
    move_radius: int,
    metric: str,
    directions: Dict[str, Direction] = DEFAULT_DIRECTIONS,
) -> Dict[str, float] | None:
    scores: Dict[str, float] = {}
    for name, (dx, dy) in directions.items():
        he_center = (he_xy[0] + dx * move_radius, he_xy[1] + dy * move_radius)
        dapi_center = (dapi_xy[0] + dx * move_radius, dapi_xy[1] + dy * move_radius)

        he_patch = extract_patch(he_img, he_center, patch_radius)
        dapi_patch = extract_patch(dapi_img, dapi_center, patch_radius)
        if he_patch is None or dapi_patch is None:
            return None

        scores[name] = patch_similarity(he_patch, dapi_patch, metric=metric)
    return scores


def aggregate_consistency(direction_scores: Dict[str, float], consistency_penalty: float = 0.5) -> float:
    vals = np.array(list(direction_scores.values()), dtype=np.float32)
    return float(vals.mean() - consistency_penalty * vals.std())


def build_neighbor_index(cells: Sequence[Cell]):
    points = np.array([(c.x, c.y) for c in cells], dtype=np.float32)
    if cKDTree is not None:
        return cKDTree(points), points
    return None, points


def radius_query(index, points: np.ndarray, query_point: Tuple[float, float], radius: float) -> List[int]:
    if index is not None:
        return list(index.query_ball_point(np.array(query_point, dtype=np.float32), radius))
    delta = points - np.array(query_point, dtype=np.float32)
    dist2 = np.sum(delta * delta, axis=1)
    return list(np.where(dist2 <= radius * radius)[0])


def align_cells(
    he_img: np.ndarray,
    dapi_img: np.ndarray,
    he_cells: Sequence[Cell],
    dapi_cells: Sequence[Cell],
    search_radius: float,
    patch_radius: int,
    move_radius: int,
    top_k: int,
    consistency_penalty: float,
    metric: str,
    use_cell_radius: bool,
    radius_scale: float,
    min_patch_radius: int,
    max_patch_radius: int,
) -> List[MatchResult]:
    dapi_index, dapi_points = build_neighbor_index(dapi_cells)
    all_results: List[MatchResult] = []

    for he in he_cells:
        candidate_ids = radius_query(dapi_index, dapi_points, (he.x, he.y), search_radius)
        if not candidate_ids:
            continue

        patch_r = per_pair_patch_radius(
            he_cell=he,
            fixed_patch_radius=patch_radius,
            use_cell_radius=use_cell_radius,
            radius_scale=radius_scale,
            min_patch_radius=min_patch_radius,
            max_patch_radius=max_patch_radius,
        )

        scored: List[MatchResult] = []
        for cid in candidate_ids:
            dapi = dapi_cells[cid]
            direction_scores = multi_direction_similarity(
                he_img=he_img,
                dapi_img=dapi_img,
                he_xy=(he.x, he.y),
                dapi_xy=(dapi.x, dapi.y),
                patch_radius=patch_r,
                move_radius=move_radius,
                metric=metric,
            )
            if direction_scores is None:
                continue

            agg = aggregate_consistency(direction_scores, consistency_penalty=consistency_penalty)
            scored.append(
                MatchResult(
                    he_cell_id=he.cell_id,
                    dapi_cell_id=dapi.cell_id,
                    aggregated_score=agg,
                    direction_scores=direction_scores,
                    metric=metric,
                    patch_radius_used=patch_r,
                    he_xy=(he.x, he.y),
                    dapi_xy=(dapi.x, dapi.y),
                )
            )

        scored.sort(key=lambda x: x.aggregated_score, reverse=True)
        all_results.extend(scored[:top_k])

    return all_results


def save_results_json(results: Iterable[MatchResult], output_path: Path) -> None:
    payload = [
        {
            "he_cell_id": r.he_cell_id,
            "dapi_cell_id": r.dapi_cell_id,
            "aggregated_score": r.aggregated_score,
            "metric": r.metric,
            "patch_radius_used": r.patch_radius_used,
            "direction_scores": r.direction_scores,
            "he_xy": {"x": r.he_xy[0], "y": r.he_xy[1]},
            "dapi_xy": {"x": r.dapi_xy[0], "y": r.dapi_xy[1]},
        }
        for r in results
    ]
    output_path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Align H&E cells to DAPI cells via directional patch similarity.")
    p.add_argument("--he-image", type=Path, required=True)
    p.add_argument("--dapi-image", type=Path, required=True)
    p.add_argument("--he-cells", type=Path, required=True)
    p.add_argument("--dapi-cells", type=Path, required=True)

    p.add_argument("--id-col", default="cell_id", help="Cell ID column (if missing, falls back to unnamed first column or row index)")
    p.add_argument("--x-col", default="x_center_pixel", help="X coordinate column (e.g., x_center_pixel)")
    p.add_argument("--y-col", default="y_center_pixel", help="Y coordinate column (e.g., y_center_pixel)")
    p.add_argument("--radius-col", default="center_radius_pixel", help="Optional per-cell radius column")

    p.add_argument("--search-radius", type=float, default=30.0)
    p.add_argument("--patch-radius", type=int, default=12)
    p.add_argument("--move-radius", type=int, default=4)
    p.add_argument("--top-k", type=int, default=3)

    p.add_argument("--metric", choices=["ncc", "ssim", "cosine"], default="ncc")
    p.add_argument("--consistency-penalty", type=float, default=0.5)

    p.add_argument("--use-cell-radius", action="store_true", help="Use HE center_radius_pixel as patch radius seed")
    p.add_argument("--radius-scale", type=float, default=2.0, help="patch_radius = round(center_radius_pixel * radius_scale)")
    p.add_argument("--min-patch-radius", type=int, default=6)
    p.add_argument("--max-patch-radius", type=int, default=24)

    p.add_argument("--output", type=Path, default=Path("alignment_results.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    he_img = to_gray(cv2.imread(str(args.he_image), cv2.IMREAD_UNCHANGED))
    dapi_img = to_gray(cv2.imread(str(args.dapi_image), cv2.IMREAD_UNCHANGED))

    he_cells = read_cells_csv(args.he_cells, args.id_col, args.x_col, args.y_col, args.radius_col)
    dapi_cells = read_cells_csv(args.dapi_cells, args.id_col, args.x_col, args.y_col, args.radius_col)

    results = align_cells(
        he_img=he_img,
        dapi_img=dapi_img,
        he_cells=he_cells,
        dapi_cells=dapi_cells,
        search_radius=args.search_radius,
        patch_radius=args.patch_radius,
        move_radius=args.move_radius,
        top_k=args.top_k,
        consistency_penalty=args.consistency_penalty,
        metric=args.metric,
        use_cell_radius=args.use_cell_radius,
        radius_scale=args.radius_scale,
        min_patch_radius=args.min_patch_radius,
        max_patch_radius=args.max_patch_radius,
    )
    save_results_json(results, args.output)
    print(f"Saved {len(results)} alignments to {args.output}")


if __name__ == "__main__":
    main()
