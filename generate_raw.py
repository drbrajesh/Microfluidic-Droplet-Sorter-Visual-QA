from pathlib import Path
import hashlib
import math
import struct
import zlib

import numpy as np
import pandas as pd


QUESTION_TYPES = [
    "target_outlet",
    "clog_location",
    "fluorescent_count",
    "gate_state",
    "bubble_position",
    "satellite_side",
]

CHOICES = {
    "target_outlet": ["upper outlet", "center outlet", "lower outlet", "inlet channel"],
    "clog_location": ["no clog", "inlet channel", "sorting junction", "lower outlet"],
    "fluorescent_count": ["0", "1", "2", "3 or more"],
    "gate_state": ["inlet electrodes", "junction electrodes", "upper branch electrodes", "lower branch electrodes"],
    "bubble_position": ["no visible bubble", "sorting junction", "upper outlet", "lower outlet"],
    "satellite_side": ["upstream of target", "downstream of target", "both sides", "not visible"],
}


def _write_png(path: Path, arr: np.ndarray) -> None:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _disk(arr, cx, cy, r, color, alpha=1.0):
    yy, xx = np.ogrid[: arr.shape[0], : arr.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    arr[mask] = (1 - alpha) * arr[mask] + alpha * np.array(color)


def _line(arr, p0, p1, width, color, alpha=1.0):
    x0, y0 = p0
    x1, y1 = p1
    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1)) + 1
    for t in np.linspace(0, 1, steps):
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        _disk(arr, x, y, max(1, width // 2), color, alpha)


def _rect(arr, x0, y0, x1, y1, color, alpha=1.0):
    x0, x1 = sorted((max(0, x0), min(arr.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(arr.shape[0], y1)))
    arr[y0:y1, x0:x1] = (1 - alpha) * arr[y0:y1, x0:x1] + alpha * np.array(color)


def _route_point(zone, rng):
    if zone == "inlet":
        return int(rng.integers(26, 58)), int(rng.integers(72, 88))
    if zone == "upper":
        t = rng.uniform(0.35, 0.86)
        return int(70 + 76 * t), int(80 - 42 * t + rng.normal(0, 2))
    if zone == "center":
        return int(rng.integers(92, 145)), int(rng.integers(72, 88))
    if zone == "lower":
        t = rng.uniform(0.35, 0.86)
        return int(70 + 76 * t), int(80 + 42 * t + rng.normal(0, 2))
    return 72, 80


def _region_point(region, rng):
    if region == "none":
        return None
    if region == "inlet":
        return int(rng.integers(36, 62)), int(rng.integers(73, 87))
    if region == "junction":
        return int(rng.integers(66, 82)), int(rng.integers(70, 90))
    if region == "upper":
        return int(rng.integers(110, 144)), int(rng.integers(34, 50))
    if region == "lower":
        return int(rng.integers(110, 144)), int(rng.integers(110, 126))
    return None


def _draw_channels(arr, layout):
    border = [41, 63, 72]
    channel = [154, 198, 207]
    if layout == "wide":
        upper_y, lower_y = 32, 128
    elif layout == "compact":
        upper_y, lower_y = 48, 112
    else:
        upper_y, lower_y = 40, 120

    segments = [
        ((10, 80), (72, 80)),
        ((72, 80), (150, upper_y)),
        ((72, 80), (150, 80)),
        ((72, 80), (150, lower_y)),
    ]
    for p0, p1 in segments:
        _line(arr, p0, p1, 19, border, 1.0)
    for p0, p1 in segments:
        _line(arr, p0, p1, 13, channel, 1.0)
    _disk(arr, 72, 80, 13, channel)


def _add_noise(arr, rng, visibility):
    sigma = {"clear": 3.0, "noisy": 9.0, "low_contrast": 6.0, "crowded": 7.0}[visibility]
    arr[:] = arr + rng.normal(0, sigma, arr.shape)
    if visibility == "low_contrast":
        arr[:] = 128 + (arr - 128) * 0.68
    if visibility in {"noisy", "crowded"}:
        for _ in range(22 if visibility == "crowded" else 10):
            x0 = int(rng.integers(4, 150))
            y0 = int(rng.integers(4, 150))
            color = rng.choice([[210, 215, 205], [70, 84, 88], [170, 194, 196]])
            _disk(arr, x0, y0, int(rng.choice([1, 1, 2, 3])), color, rng.uniform(0.35, 0.75))


def _draw_scene(rng, scene):
    arr = np.zeros((160, 160, 3), dtype=float)
    base = np.array([22, 30, 34], dtype=float)
    arr[:] = base + rng.normal(0, 2.0, arr.shape)
    _draw_channels(arr, scene["layout_family"])

    # Analysis chamber used for counting fluorescent droplets.
    _rect(arr, 18, 112, 68, 148, [37, 54, 61], 1.0)
    _rect(arr, 21, 115, 65, 145, [92, 118, 123], 0.75)

    droplet_colors = [[236, 177, 73], [210, 94, 150], [92, 187, 236], [215, 229, 130]]
    for zone in ["inlet", "upper", "center", "lower"]:
        for _ in range(scene[f"{zone}_droplets"]):
            x, y = _route_point(zone, rng)
            _disk(arr, x, y, int(rng.integers(5, 8)), droplet_colors[int(rng.integers(0, len(droplet_colors)))], 0.88)

    tx, ty = _route_point(scene["target_zone"], rng)
    scene["target_xy"] = f"{tx},{ty}"
    _disk(arr, tx, ty, 9, [250, 218, 70], 0.95)
    _disk(arr, tx, ty, 5, [165, 55, 180], 0.95)
    _disk(arr, tx, ty, 11, [255, 250, 190], 0.35)

    if scene["satellite_side"] != "not_visible":
        offsets = []
        if scene["satellite_side"] in {"upstream", "both"}:
            offsets += [(-13, -4), (-16, 4)]
        if scene["satellite_side"] in {"downstream", "both"}:
            offsets += [(13, -4), (16, 4)]
        for dx, dy in offsets:
            _disk(arr, tx + dx, ty + dy, 2, [242, 211, 92], 0.9)

    for _ in range(scene["fluorescent_count"]):
        _disk(arr, int(rng.integers(28, 60)), int(rng.integers(121, 140)), int(rng.integers(4, 6)), [76, 255, 116], 0.95)

    clog_pt = _region_point(scene["clog_location"], rng)
    if clog_pt is not None:
        x, y = clog_pt
        _rect(arr, x - 7, y - 5, x + 8, y + 6, [33, 26, 21], 0.92)
        _disk(arr, x + 4, y, 5, [78, 55, 37], 0.65)

    bubble_pt = _region_point(scene["bubble_position"], rng)
    if bubble_pt is not None:
        x, y = bubble_pt
        _disk(arr, x, y, 7, [231, 253, 255], 0.82)
        _disk(arr, x - 2, y - 2, 3, [255, 255, 255], 0.95)

    gate_centers = {
        "inlet": (48, 64),
        "junction": (78, 62),
        "upper": (116, 31),
        "lower": (116, 129),
    }
    for gate, (x, y) in gate_centers.items():
        color = [241, 82, 60] if gate == scene["gate_state"] else [86, 101, 108]
        alpha = 0.95 if gate == scene["gate_state"] else 0.55
        _rect(arr, x - 9, y - 3, x + 9, y + 1, color, alpha)
        _rect(arr, x - 9, y + 6, x + 9, y + 10, color, alpha)

    if scene["ood_axis"] == "partial_occlusion":
        for _ in range(3):
            y = int(rng.integers(20, 145))
            _line(arr, (0, y), (160, y + int(rng.integers(-12, 12))), 2, [10, 16, 18], 0.45)
    elif scene["ood_axis"] == "dim_fluorescence":
        arr[:, :, 1] *= 0.86
    elif scene["ood_axis"] == "dense_debris":
        for _ in range(35):
            _disk(arr, int(rng.integers(0, 160)), int(rng.integers(0, 160)), 1, [190, 200, 190], 0.55)

    _add_noise(arr, rng, scene["visibility"])
    return np.clip(arr, 0, 255).astype(np.uint8)


def _answer_for(scene, question_type):
    if question_type == "target_outlet":
        return {"upper": "A", "center": "B", "lower": "C", "inlet": "D"}[scene["target_zone"]]
    if question_type == "clog_location":
        return {"none": "A", "inlet": "B", "junction": "C", "lower": "D"}[scene["clog_location"]]
    if question_type == "fluorescent_count":
        return ["A", "B", "C", "D"][min(scene["fluorescent_count"], 3)]
    if question_type == "gate_state":
        return {"inlet": "A", "junction": "B", "upper": "C", "lower": "D"}[scene["gate_state"]]
    if question_type == "bubble_position":
        return {"none": "A", "junction": "B", "upper": "C", "lower": "D"}[scene["bubble_position"]]
    if question_type == "satellite_side":
        return {"upstream": "A", "downstream": "B", "both": "C", "not_visible": "D"}[scene["satellite_side"]]
    raise ValueError(question_type)


def _question_text(question_type):
    return {
        "target_outlet": "Where is the ringed target droplet located?",
        "clog_location": "Which chip region contains the dark obstruction?",
        "fluorescent_count": "How many bright green droplets are visible in the counting chamber?",
        "gate_state": "Which electrode pair is active in red?",
        "bubble_position": "Where is the bright air bubble located?",
        "satellite_side": "Where are the small satellite droplets relative to the target droplet?",
    }[question_type]


def _stable_id(prefix, value):
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:14]}"


def main():
    rng = np.random.default_rng(88421)
    root = Path(__file__).resolve().parent
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for old in image_dir.glob("*.png"):
        old.unlink()

    rows = []
    scene_count = 900
    for i in range(scene_count):
        difficulty = rng.choice(["easy", "medium", "hard"], p=[0.28, 0.46, 0.26])
        visibility = (
            rng.choice(["clear", "noisy"], p=[0.82, 0.18])
            if difficulty == "easy"
            else rng.choice(["clear", "noisy", "low_contrast", "crowded"], p=[0.32, 0.30, 0.20, 0.18])
            if difficulty == "medium"
            else rng.choice(["noisy", "low_contrast", "crowded"], p=[0.30, 0.35, 0.35])
        )
        ood_axis = rng.choice(
            ["standard", "partial_occlusion", "dim_fluorescence", "dense_debris"],
            p=[0.62, 0.12, 0.13, 0.13] if difficulty != "hard" else [0.26, 0.24, 0.23, 0.27],
        )
        scene = {
            "scene_id": f"raw_mf_{i:05d}",
            "layout_family": rng.choice(["standard", "wide", "compact"], p=[0.55, 0.22, 0.23]),
            "difficulty": difficulty,
            "visibility": visibility,
            "ood_axis": ood_axis,
            "target_zone": rng.choice(["upper", "center", "lower", "inlet"], p=[0.26, 0.26, 0.28, 0.20]),
            "clog_location": rng.choice(["none", "inlet", "junction", "lower"], p=[0.38, 0.18, 0.24, 0.20]),
            "fluorescent_count": int(rng.choice([0, 1, 2, 3, 4], p=[0.16, 0.26, 0.25, 0.21, 0.12])),
            "gate_state": rng.choice(["inlet", "junction", "upper", "lower"]),
            "bubble_position": rng.choice(["none", "junction", "upper", "lower"], p=[0.42, 0.22, 0.18, 0.18]),
            "satellite_side": rng.choice(["upstream", "downstream", "both", "not_visible"], p=[0.24, 0.24, 0.22, 0.30]),
            "inlet_droplets": int(rng.integers(1, 4)),
            "upper_droplets": int(rng.integers(0, 3)),
            "center_droplets": int(rng.integers(0, 3)),
            "lower_droplets": int(rng.integers(0, 3)),
        }
        arr = _draw_scene(rng, scene)
        image_path = f"images/{scene['scene_id']}.png"
        _write_png(root / image_path, arr)

        qtypes = list(rng.choice(QUESTION_TYPES, size=3, replace=False))
        if difficulty == "hard" and "target_outlet" not in qtypes:
            qtypes[0] = "target_outlet"
        for qtype in qtypes:
            question_id = _stable_id("raw_q", f"{scene['scene_id']}::{qtype}")
            choices = CHOICES[qtype]
            answer = _answer_for(scene, qtype)
            rows.append(
                {
                    "question_id": question_id,
                    "scene_id": scene["scene_id"],
                    "image_path": image_path,
                    "question_type": qtype,
                    "question": _question_text(qtype),
                    "choice_a": choices[0],
                    "choice_b": choices[1],
                    "choice_c": choices[2],
                    "choice_d": choices[3],
                    "answer_label": answer,
                    "difficulty": difficulty,
                    "visibility": visibility,
                    "layout_family": scene["layout_family"],
                    "ood_axis": ood_axis,
                    "trace_target_zone": scene["target_zone"],
                    "trace_clog_location": scene["clog_location"],
                    "trace_gate_state": scene["gate_state"],
                    "trace_bubble_position": scene["bubble_position"],
                    "trace_satellite_side": scene["satellite_side"],
                    "trace_fluorescent_count": scene["fluorescent_count"],
                }
            )

    pd.DataFrame(rows).to_csv(root / "data.csv", index=False)
    print(f"wrote {scene_count} images and {len(rows)} question rows")


if __name__ == "__main__":
    main()
