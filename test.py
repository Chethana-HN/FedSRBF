import argparse
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset
import torch

def _resize_image(arr, size, is_mask=False):
    pil = Image.fromarray(arr)
    if is_mask:
        pil = pil.resize((size, size), resample=Image.NEAREST)
    else:
        pil = pil.resize((size, size), resample=Image.BILINEAR)
    return np.array(pil)

def _load_rgb(path, image_size):
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    arr = _resize_image(arr, image_size, is_mask=False)
    arr = arr.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return arr

def _load_mask(path, image_size):
    m = Image.open(path).convert('L')
    arr = np.array(m)
    arr = _resize_image(arr, image_size, is_mask=True)
    arr = (arr > 127).astype(np.float32)
    return arr[None, ...]

def _normalize(v, eps=1e-08):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n

def _active_coords(mask):
    coords = np.argwhere(mask > 0)
    return coords.astype(np.int32)

def _nearest_active(mask, point, coords=None):
    h, w = mask.shape
    py = int(round(float(point[0])))
    px = int(round(float(point[1])))
    if 0 <= py < h and 0 <= px < w and mask[py, px]:
        return (py, px)
    for rad in [1, 2, 3, 4, 6]:
        y0 = max(0, py - rad)
        y1 = min(h, py + rad + 1)
        x0 = max(0, px - rad)
        x1 = min(w, px + rad + 1)
        sub = mask[y0:y1, x0:x1]
        ys, xs = np.where(sub)
        if len(ys) > 0:
            ys = ys + y0
            xs = xs + x0
            d2 = (ys - point[0]) ** 2 + (xs - point[1]) ** 2
            k = int(np.argmin(d2))
            return (int(ys[k]), int(xs[k]))
    if coords is None:
        coords = _active_coords(mask)
    if len(coords) == 0:
        return (py, px)
    d2 = (coords[:, 0] - point[0]) ** 2 + (coords[:, 1] - point[1]) ** 2
    k = int(np.argmin(d2))
    return (int(coords[k, 0]), int(coords[k, 1]))

def _draw_cubic_bezier_on_mask(mask, start, end, ctrl1, ctrl2):
    coords = _active_coords(mask)
    if len(coords) == 0:
        return np.zeros_like(mask, dtype=np.float32)
    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(ctrl1, dtype=np.float64)
    p2 = np.asarray(ctrl2, dtype=np.float64)
    p3 = np.asarray(end, dtype=np.float64)
    span = np.linalg.norm(p3 - p0)
    steps = max(16, int(np.ceil(span * 2.5)))
    out = np.zeros_like(mask, dtype=bool)
    for i in range(steps + 1):
        t = i / float(steps)
        omt = 1.0 - t
        p = omt ** 3 * p0 + 3.0 * omt ** 2 * t * p1 + 3.0 * omt * t ** 2 * p2 + t ** 3 * p3
        y, x = _nearest_active(mask, p, coords=coords)
        out[y, x] = True
    return out.astype(np.float32)

def _cubic_point(t, p0, p1, p2, p3):
    omt = 1.0 - t
    return omt ** 3 * p0 + 3.0 * omt ** 2 * t * p1 + 3.0 * omt * t ** 2 * p2 + t ** 3 * p3

def _principal_axes(fg):
    coords = _active_coords(fg)
    if len(coords) < 2:
        return (coords.mean(axis=0), np.array([1.0, 0.0]), np.array([0.0, 1.0]), coords)
    center = coords.mean(axis=0).astype(np.float64)
    centered = coords.astype(np.float64) - center[None, :]
    cov = centered.T @ centered / max(len(coords) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    major = _normalize(vecs[:, order[0]])
    minor = _normalize(np.array([-major[1], major[0]], dtype=np.float64))
    return (center, major, minor, coords)

def _connected_foreground_scribble(fg, rng, width=0):
    fg = fg.astype(bool)
    if not fg.any():
        return np.zeros_like(fg, dtype=np.float32)
    center, major, minor, coords = _principal_axes(fg)
    centered = coords.astype(np.float64) - center[None, :]
    proj_major = centered @ major
    q_lo = float(np.quantile(proj_major, 0.08))
    q_hi = float(np.quantile(proj_major, 0.92))
    if q_hi <= q_lo:
        q_lo = float(proj_major.min())
        q_hi = float(proj_major.max())
    p0 = center + q_lo * major
    p3 = center + q_hi * major
    span = float(np.linalg.norm(p3 - p0))
    bend1 = float(rng.uniform(-0.16, 0.16) * span)
    bend2 = float(rng.uniform(-0.18, 0.18) * span)
    if abs(bend1 - bend2) < 0.04 * span:
        bend2 *= -1.0
    p1 = p0 + 0.32 * (p3 - p0) + bend1 * minor
    p2 = p0 + 0.68 * (p3 - p0) + bend2 * minor
    scribble = _draw_cubic_bezier_on_mask(fg, p0, p3, p1, p2).astype(bool)
    branch_specs = [(0.34, 1.0), (0.68, -1.0)]
    branch_len = float(np.clip(0.24 * span, 14.0, 27.0))
    for t0, side in branch_specs:
        b0 = _cubic_point(t0, p0, p1, p2, p3)
        by, bx = _nearest_active(fg, b0, coords=coords)
        b0 = np.array([float(by), float(bx)], dtype=np.float64)
        forward = float(rng.uniform(-0.45, 0.45))
        bdir = _normalize(side * minor + forward * major)
        b3 = b0 + branch_len * bdir
        c1 = b0 + 0.35 * branch_len * bdir + float(rng.uniform(-0.12, 0.12) * branch_len) * major
        c2 = b0 + 0.72 * branch_len * bdir + float(rng.uniform(-0.18, 0.18) * branch_len) * major
        branch = _draw_cubic_bezier_on_mask(fg, b0, b3, c1, c2).astype(bool)
        scribble |= branch
    if int(width) > 0:
        scribble = binary_dilation(scribble, iterations=int(width))
    scribble &= fg
    return scribble.astype(np.float32)

def _single_background_branch(fg, rng, width=0, inner_margin=3, outer_margin=18, side=1):
    fg = fg.astype(bool)
    if not fg.any():
        return np.zeros_like(fg, dtype=np.float32)
    center, major, minor, _ = _principal_axes(fg)
    inner = binary_dilation(fg, iterations=max(1, int(inner_margin)))
    outer = binary_dilation(fg, iterations=max(int(outer_margin), int(inner_margin) + 2))
    band = outer & ~inner & ~fg
    if band.sum() < 20:
        band = outer & ~binary_dilation(fg, iterations=1) & ~fg
    if not band.any():
        band = ~fg
    coords_band = _active_coords(band)
    if len(coords_band) == 0:
        return np.zeros_like(fg, dtype=np.float32)
    radial = 0.55 * (float(inner_margin) + float(outer_margin))
    start_guess = center + float(side) * radial * minor
    sy, sx = _nearest_active(band, start_guess, coords=coords_band)
    p0 = np.array([float(sy), float(sx)], dtype=np.float64)
    branch_len = float(np.clip(np.sqrt(max(int(fg.sum()), 1)) * 0.52, 22.0, 36.0))
    tangent_sign = 1.0 if rng.random() < 0.5 else -1.0
    direction = _normalize(tangent_sign * major + float(rng.uniform(-0.3, 0.3)) * minor)
    p3 = p0 + branch_len * direction
    c1 = p0 + 0.33 * branch_len * direction + float(side) * float(rng.uniform(0.08, 0.18)) * branch_len * minor
    c2 = p0 + 0.7 * branch_len * direction - float(side) * float(rng.uniform(0.05, 0.14)) * branch_len * minor
    branch = _draw_cubic_bezier_on_mask(band, p0, p3, c1, c2).astype(bool)
    if int(width) > 0:
        branch = binary_dilation(branch, iterations=int(width))
    branch &= ~fg
    branch &= outer
    return branch.astype(np.float32)

def _connected_background_scribble(fg, rng, width=0, inner_margin=3, outer_margin=18, num_branches=2):
    fg = fg.astype(bool)
    out = np.zeros_like(fg, dtype=np.float32)
    if not fg.any():
        return out
    sides = [1, -1]
    for i in range(min(int(num_branches), 2)):
        out = np.maximum(out, _single_background_branch(fg, rng, width=width, inner_margin=inner_margin, outer_margin=outer_margin, side=sides[i]))
    for i in range(max(0, int(num_branches) - 2)):
        side = 1 if i % 2 == 0 else -1
        out = np.maximum(out, _single_background_branch(fg, rng, width=width, inner_margin=inner_margin, outer_margin=outer_margin, side=side))
    out *= (~fg).astype(np.float32)
    return out.astype(np.float32)

def make_sparse(mask, mode='scribble', points_per_class=8, scribble_width=0, rng=None, bg_inner_margin=3, bg_outer_margin=18, bg_num_branches=2):
    if rng is None:
        rng = np.random.default_rng(42)
    c, _, _ = mask.shape
    pos = np.zeros_like(mask, dtype=np.float32)
    neg = np.zeros_like(mask, dtype=np.float32)
    for ch in range(c):
        fg = mask[ch] > 0.5
        if mode == 'point':
            fidx = np.argwhere(fg)
            if len(fidx):
                n = min(int(points_per_class), len(fidx))
                chosen = rng.choice(len(fidx), size=n, replace=False)
                take = fidx[chosen]
                pos[ch, take[:, 0], take[:, 1]] = 1.0
            bg = ~fg
            bidx = np.argwhere(bg)
            if len(bidx):
                n = min(int(points_per_class), len(bidx))
                chosen = rng.choice(len(bidx), size=n, replace=False)
                take = bidx[chosen]
                neg[ch, take[:, 0], take[:, 1]] = 1.0
        elif mode == 'scribble':
            pos[ch] = _connected_foreground_scribble(fg, rng=rng, width=scribble_width)
            neg[ch] = _connected_background_scribble(fg, rng=rng, width=scribble_width, inner_margin=bg_inner_margin, outer_margin=bg_outer_margin, num_branches=bg_num_branches)
        else:
            raise ValueError(f'Unknown sparse mode: {mode}')
        pos[ch] *= fg.astype(np.float32)
        neg[ch] *= (~fg).astype(np.float32)
    return np.concatenate([pos, neg], axis=0).astype(np.float32)

def _augment(image, mask, rng):
    if rng.random() < 0.5:
        image = image[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()
    if rng.random() < 0.5:
        image = image[:, ::-1, :].copy()
        mask = mask[:, ::-1, :].copy()
    k = int(rng.integers(4))
    if k:
        image = np.rot90(image, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(1, 2)).copy()
    return (image, mask)

class MedicalSegDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_size=256, num_classes=1, mask_values=None, sparse_mode='scribble', points_per_class=8, mask_suffix='', augment=False, sparse_seed=42, scribble_width=0, bg_inner_margin=3, bg_outer_margin=18, bg_num_branches=2):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.mask_values = mask_values
        self.sparse_mode = sparse_mode
        self.points_per_class = int(points_per_class)
        self.mask_suffix = str(mask_suffix)
        self.augment = bool(augment)
        self.sparse_seed = int(sparse_seed)
        self.scribble_width = int(scribble_width)
        self.bg_inner_margin = int(bg_inner_margin)
        self.bg_outer_margin = int(bg_outer_margin)
        self.bg_num_branches = int(bg_num_branches)
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        self.images = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in exts])
        if len(self.images) == 0:
            raise RuntimeError(f'No images found in {self.image_dir}')

    def __len__(self):
        return len(self.images)

    def _mask_path(self, image_path):
        return self.mask_dir / f'{image_path.stem}{self.mask_suffix}{image_path.suffix}'

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self._mask_path(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(f'Mask not found for {image_path.name}: {mask_path}')
        image = _load_rgb(image_path, self.image_size)
        mask = _load_mask(mask_path, self.image_size)
        rng = np.random.default_rng(self.sparse_seed + index)
        if self.augment:
            image, mask = _augment(image, mask, rng)
        sparse = make_sparse(mask, mode=self.sparse_mode, points_per_class=self.points_per_class, scribble_width=self.scribble_width, rng=rng, bg_inner_margin=self.bg_inner_margin, bg_outer_margin=self.bg_outer_margin, bg_num_branches=self.bg_num_branches)
        return {'image': torch.from_numpy(image).float(), 'mask': torch.from_numpy(mask).float(), 'sparse': torch.from_numpy(sparse).float(), 'name': image_path.name}

def hd95(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    if not a.any() and not b.any():
        return 0.0
    if not a.any() or not b.any():
        return float(max(a.shape))
    sa = a ^ binary_erosion(a)
    sb = b ^ binary_erosion(b)
    dta = distance_transform_edt(~a)
    dtb = distance_transform_edt(~b)
    distances = np.concatenate([dtb[sa], dta[sb]])
    return float(np.percentile(distances, 95)) if len(distances) else 0.0


def image_metrics(prob, target):
    pred = prob.detach().cpu().numpy() >= 0.5
    true = target.detach().cpu().numpy() >= 0.5
    scores = []
    for n in range(pred.shape[0]):
        for c in range(pred.shape[1]):
            a = pred[n, c]
            b = true[n, c]
            intersection = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            total = a.sum() + b.sum()
            dice = (2 * intersection + 1e-6) / (total + 1e-6)
            scores.append((dice, hd95(a, b)))
    return np.asarray(scores, dtype=np.float64).mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--model', default='model.pth')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    device_name = 'cuda:0' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    device = torch.device(device_name)
    model = torch.jit.load(args.model, map_location=device)
    model.eval()
    root = Path(args.data_root)
    names = ('B1', 'B2', 'B3', 'B4')
    means = []
    print(f'{"Client":<24}{"Dice (%)":>21}{"HD95 (px)":>21}')
    print('-' * 66)
    for index, name in enumerate(names, 1):
        directory = root / name / 'test'
        dataset = MedicalSegDataset(
            image_dir=directory / 'image',
            mask_dir=directory / 'mask',
            mask_suffix='_mask' if index in (1, 2) else '',
            image_size=256,
            sparse_mode='scribble',
            sparse_seed=42,
            scribble_width=0,
            bg_inner_margin=3,
            bg_outer_margin=18,
            bg_num_branches=2,
            augment=False,
        )
        scores = []
        with torch.inference_mode():
            for item in dataset:
                image = item['image'].unsqueeze(0).to(device)
                mask = item['mask'].unsqueeze(0).to(device)
                scribble = item['sparse'].unsqueeze(0).to(device)
                generator = torch.Generator(device=device)
                generator.manual_seed(1729)
                noise = torch.randn((1, 32, 128, 128), generator=generator, device=device, dtype=image.dtype)
                prediction = model(image, scribble, noise)
                scores.append(image_metrics(prediction, mask))
        values = np.asarray(scores, dtype=np.float64)
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=0)
        means.append(mean)
        print(f'{name:<24}{f"{mean[0] * 100:.2f} ± {sd[0] * 100:.2f}":>21}{f"{mean[1]:.2f} ± {sd[1]:.2f}":>21}')
    values = np.asarray(means, dtype=np.float64)
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=0)
    print('-' * 66)
    print(f'{"Average":<24}{f"{mean[0] * 100:.2f} ± {sd[0] * 100:.2f}":>21}{f"{mean[1]:.2f} ± {sd[1]:.2f}":>21}')


if __name__ == '__main__':
    main()
