#!/usr/bin/env python3
"""
mosaic_censor.py - MP4動画の性器にモザイクをかけるツール

使い方:
  python mosaic_censor.py input.mp4
  python mosaic_censor.py input.mp4 --block-size 20
  python mosaic_censor.py input.mp4 --confidence 0.25
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# COCO 17 keypoints
# 0:鼻  1:左目  2:右目  3:左耳  4:右耳
# 5:左肩  6:右肩  7:左肘  8:右肘  9:左手首  10:右手首
# 11:左腰骨  12:右腰骨  13:左膝  14:右膝  15:左足首  16:右足首
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # 顔
    (5, 6),                                    # 両肩
    (5, 7), (7, 9), (6, 8), (8, 10),          # 腕
    (5, 11), (6, 12), (11, 12),               # 体幹
    (11, 13), (13, 15), (12, 14), (14, 16),   # 脚
]

CENSOR_LABELS = {
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
}

YOLO_CENSOR_LABELS = {
    "penis",
    "vagina",
    "vaginal",
    "vulva",
}

LABEL_SHORT: dict[str, str] = {
    "MALE_GENITALIA_EXPOSED":  "penis",
    "FEMALE_GENITALIA_EXPOSED": "vagina",
    "ANUS_EXPOSED":            "anus",
}

YOLO_LABEL_SHORT: dict[str, str] = {
    "penis": "penis",
    "vagina": "vagina",
    "vaginal": "vagina",
    "vulva": "vagina",
}

POSE_BACKENDS = ("yolo11", "yolo8", "vitpose-h", "rtmpose", "rtmpose-wholebody")
MAX_CSV_MOSAICS = 255


def load_pose_model(backend: str):
    """
    ポーズ検出モデルをロードして返す。
    戻り値は (backend_name, *models) のタプル。
    """
    if backend in ("yolo11", "yolo8"):
        from ultralytics import YOLO
        model_name = "yolo11l-pose.pt" if backend == "yolo11" else "yolov8n-pose.pt"
        return ("yolo", YOLO(model_name, task="pose"))
    elif backend == "vitpose-h":
        try:
            from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
            import torch
        except ImportError:
            print(
                "エラー: ViTPose-H には transformers が必要です。\n"
                "  pip install transformers",
                file=sys.stderr,
            )
            sys.exit(1)
        from ultralytics import YOLO
        person_detector = YOLO("yolo11n.pt", task="detect")
        model_id = "nielsr/vitpose-base-simple"
        processor = VitPoseImageProcessor.from_pretrained(model_id)
        vit_model = VitPoseForPoseEstimation.from_pretrained(model_id)
        vit_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vit_model = vit_model.to(device)
        return ("vitpose", person_detector, processor, vit_model)
    elif backend in ("rtmpose", "rtmpose-wholebody"):
        try:
            from rtmlib import Body, Wholebody
        except ImportError:
            print(
                "エラー: RTMPose には rtmlib が必要です。\n"
                "  pip install rtmlib",
                file=sys.stderr,
            )
            sys.exit(1)
        if backend == "rtmpose":
            model = Body(to_openpose=False, backend="onnxruntime", device="cpu")
        else:
            model = Wholebody(to_openpose=False, backend="onnxruntime", device="cpu")
        return ("rtmpose", model)
    else:
        raise ValueError(f"未知のポーズモデル: {backend}（選択肢: {POSE_BACKENDS}）")


def apply_mosaic(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, block_size: int) -> np.ndarray:
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    result = frame.copy()
    result[y1:y2, x1:x2] = mosaic
    return result


def _crotch_box_from_hips(
    lhip: np.ndarray, rhip: np.ndarray, W: int, H: int
) -> tuple[int, int, int, int] | None:
    """腰骨キーポイント2点から股間検索ボックスを計算する。信頼度不足は None を返す。"""
    if lhip[2] < 0.3 or rhip[2] < 0.3:
        return None
    dx = lhip[0] - rhip[0]
    dy = lhip[1] - rhip[1]
    hip_w = max(int((dx * dx + dy * dy) ** 0.5), 40)
    cx = int((lhip[0] + rhip[0]) / 2)
    cy = int((lhip[1] + rhip[1]) / 2)
    return (
        max(0, cx - hip_w * 3 // 2),
        max(0, cy - hip_w * 3 // 2),
        min(W, cx + hip_w * 3 // 2),
        min(H, cy + hip_w * 3 // 2),
    )


def _get_crotch_boxes_yolo(
    frame: np.ndarray, pose_model
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    H, W = frame.shape[:2]
    results = pose_model(frame, verbose=False, device="cpu")
    search_boxes: list[tuple[int, int, int, int]] = []
    all_kps: list[np.ndarray] = []
    for r in results:
        if r.keypoints is None or r.keypoints.data.shape[0] == 0:
            continue
        for kps_tensor in r.keypoints.data:
            kps = kps_tensor.cpu().numpy()  # [17, 3]
            all_kps.append(kps)
            box = _crotch_box_from_hips(kps[11], kps[12], W, H)
            if box is not None:
                search_boxes.append(box)
    return search_boxes, all_kps


def _get_crotch_boxes_vitpose(
    frame: np.ndarray, person_detector, processor, vit_model
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    import torch
    from PIL import Image

    H, W = frame.shape[:2]

    # Step 1: 人物検出
    det_results = person_detector(frame, verbose=False, device="cpu", classes=[0])
    person_boxes: list[list[float]] = []
    for r in det_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            person_boxes.append([x1, y1, x2, y2])

    if not person_boxes:
        return [], []

    # Step 2: ViTPose でキーポイント推定
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    inputs = processor(images=pil_image, boxes=[person_boxes], return_tensors="pt")
    device = next(vit_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = vit_model(**inputs)
    pose_results = processor.post_process_pose_estimation(outputs, boxes=[person_boxes])

    search_boxes: list[tuple[int, int, int, int]] = []
    all_kps: list[np.ndarray] = []
    for person in pose_results[0]:
        # keypoints: [17, 2], scores: [17]
        kps_xy = person["keypoints"].cpu().numpy()
        scores = person["scores"].cpu().numpy()
        kps = np.concatenate([kps_xy, scores[:, None]], axis=1)  # [17, 3]
        all_kps.append(kps)
        box = _crotch_box_from_hips(kps[11], kps[12], W, H)
        if box is not None:
            search_boxes.append(box)
    return search_boxes, all_kps


def _get_crotch_boxes_rtmpose(
    frame: np.ndarray, body_model
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    H, W = frame.shape[:2]
    keypoints, scores = body_model(frame)  # (N,17,2), (N,17)
    search_boxes: list[tuple[int, int, int, int]] = []
    all_kps: list[np.ndarray] = []
    for kps_xy, kps_scores in zip(keypoints, scores):
        kps = np.concatenate([kps_xy, kps_scores[:, None]], axis=1)  # [17, 3]
        all_kps.append(kps)
        box = _crotch_box_from_hips(kps[11], kps[12], W, H)
        if box is not None:
            search_boxes.append(box)
    return search_boxes, all_kps


def get_crotch_boxes(
    frame: np.ndarray, pose_model_bundle
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    """
    ポーズ検出で腰骨ランドマークを検出し、2つを返す。
    - search_boxes: NudeNet 検出フィルタ用ボックス
    - all_kps: 人物ごとのキーポイント配列 [17, 3] (x, y, conf) のリスト（デバッグ描画用）
    """
    backend = pose_model_bundle[0]
    if backend == "yolo":
        return _get_crotch_boxes_yolo(frame, pose_model_bundle[1])
    elif backend == "rtmpose":
        return _get_crotch_boxes_rtmpose(frame, pose_model_bundle[1])
    else:  # vitpose
        return _get_crotch_boxes_vitpose(frame, *pose_model_bundle[1:])


def detect_genitalia_multicrop(
    detector,
    frame: np.ndarray,
    tmp_path: str,
    confidence: float,
) -> list[tuple[tuple[int, int, int, int], float, str]]:
    """
    フル画面で NudeNet 検出し、
    (フル画面座標ボックス, スコア, ラベル) のリストを返す。
    """
    fh, fw = frame.shape[:2]
    all_labels = CENSOR_LABELS
    all_boxes: list[tuple[tuple[int, int, int, int], float, str]] = []
    seen: set[tuple[int, int, int, int]] = set()

    # フル画面
    cv2.imwrite(tmp_path, frame)
    for det in detector.detect(tmp_path):
        if det["class"] not in all_labels or det["score"] < confidence:
            continue
        b = det["box"]
        box = (
            max(0, int(b[0])),
            max(0, int(b[1])),
            min(fw, int(b[0]) + int(b[2])),
            min(fh, int(b[1]) + int(b[3])),
        )
        if box[2] > box[0] and box[3] > box[1] and box not in seen:
            seen.add(box)
            all_boxes.append((box, float(det["score"]), det["class"]))

    return all_boxes


def detect_yolo_genitalia_multicrop(
    model,
    frame: np.ndarray,
    confidence: float,
    source_name: str,
) -> list[tuple[tuple[int, int, int, int], float, str]]:
    """
    YOLO系NSFW検出器でフル画面を検出する。
    (フル画面座標ボックス, スコア, ラベル) のリストを返す。
    """
    if model is None:
        return []

    fh, fw = frame.shape[:2]
    all_boxes: list[tuple[tuple[int, int, int, int], float, str]] = []
    seen: set[tuple[int, int, int, int, str]] = set()

    def _collect(img: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> None:
        results = model(img, verbose=False, device="cpu", conf=confidence)
        names = getattr(model, "names", {})
        for r in results:
            if r.boxes is None:
                continue
            for det in r.boxes:
                cls_id = int(det.cls[0])
                label = str(names.get(cls_id, cls_id)).lower()
                if label not in YOLO_CENSOR_LABELS:
                    continue
                display_label = f"{source_name} {YOLO_LABEL_SHORT.get(label, label)}"
                score = float(det.conf[0])
                x1, y1, x2, y2 = [int(v) for v in det.xyxy[0].tolist()]
                box = (
                    max(0, x1 + offset_x),
                    max(0, y1 + offset_y),
                    min(fw, x2 + offset_x),
                    min(fh, y2 + offset_y),
                )
                key = (*box, label)
                if box[2] > box[0] and box[3] > box[1] and key not in seen:
                    seen.add(key)
                    all_boxes.append((box, score, display_label))

    _collect(frame)

    return all_boxes


def boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def box_overlaps_any(
    box: tuple[int, int, int, int],
    crotch_boxes: list[tuple[int, int, int, int]],
) -> bool:
    return any(boxes_overlap(box, cb) for cb in crotch_boxes)


def clip_to_crotch_box(
    box: tuple[int, int, int, int],
    crotch_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """中心が入っている crotch 枠にクリップして返す。"""
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    for cb in crotch_boxes:
        if cb[0] <= cx <= cb[2] and cb[1] <= cy <= cb[3]:
            return (max(box[0], cb[0]), max(box[1], cb[1]),
                    min(box[2], cb[2]), min(box[3], cb[3]))
    return box


def filter_by_crotch(
    nudenet_boxes_with_score: list[tuple[tuple[int, int, int, int], float, str]],
    crotch_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[tuple[int, int, int, int], str]]:
    """
    検出矩形が crotch 矩形と重なる場合だけ採用する。
    モザイク対象は検出器が返した矩形そのもの。
    """
    result: list[tuple[tuple[int, int, int, int], str]] = []
    for box, score, label in nudenet_boxes_with_score:
        short = LABEL_SHORT.get(label, label)
        if crotch_boxes and box_overlaps_any(box, crotch_boxes):
            result.append((box, short))
    return result


def merge_adopted_boxes(
    boxes: list[tuple[tuple[int, int, int, int], str]],
    iou_threshold: float = 0.65,
) -> list[tuple[tuple[int, int, int, int], str]]:
    """複数検出器の近いbboxを軽く重複排除する。"""
    merged: list[tuple[tuple[int, int, int, int], str]] = []
    for box, label in boxes:
        duplicate = False
        for kept_box, kept_label in merged:
            if normalize_display_label(label) == normalize_display_label(kept_label) and box_iou(box, kept_box) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            merged.append((box, label))
    return merged


def normalize_display_label(label: str) -> str:
    parts = label.split(maxsplit=1)
    if len(parts) == 2 and parts[0] in {"NudeNet", "erax", "Felldude", "Throaway", "YOLO"}:
        return parts[1]
    return label


def detector_display_label(label: str) -> str:
    if label.startswith(("NudeNet ", "erax ", "Felldude ", "Throaway ", "YOLO ")):
        return label
    if label in CENSOR_LABELS:
        return f"NudeNet {LABEL_SHORT.get(label, label)}"
    if label in YOLO_CENSOR_LABELS:
        return f"YOLO {YOLO_LABEL_SHORT.get(label, label)}"
    return label


def split_model_label(label: str) -> tuple[str, str]:
    parts = label.split(maxsplit=1)
    if len(parts) == 2 and parts[0] in {"NudeNet", "erax", "Felldude", "Throaway", "YOLO"}:
        return parts[0], parts[1]
    return "YOLO", label


def yolo_source_name(model_path: str) -> str:
    stem = Path(model_path).stem.lower()
    if "erax" in stem:
        return "erax"
    if "felldude" in stem or "yolo_nsfw_n" in stem:
        return "Felldude"
    if "throaway" in stem or "penis_detection" in stem:
        return "Throaway"
    return "YOLO"


def detector_name_suffix(yolo_nsfw_model_path: str | None, pose_backend: str = "yolo11") -> str:
    base = "NudeNet" if not yolo_nsfw_model_path else f"NudeNet_{Path(yolo_nsfw_model_path).stem}"
    return f"{base}_{pose_backend}"


def effective_yolo_confidence(yolo_nsfw_model_path: str | None, yolo_confidence: float | None) -> float:
    if yolo_confidence is not None:
        return yolo_confidence
    if yolo_nsfw_model_path and yolo_source_name(yolo_nsfw_model_path) == "erax":
        return 0.3
    return 0.03


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def interpolate_box(
    previous_box: tuple[int, int, int, int],
    current_box: tuple[int, int, int, int],
    t: float,
) -> tuple[int, int, int, int]:
    return tuple(
        int(round(previous_box[i] + (current_box[i] - previous_box[i]) * t))
        for i in range(4)
    )


def interpolate_adopted_boxes(
    previous_boxes: list[tuple[tuple[int, int, int, int], str]],
    current_boxes: list[tuple[tuple[int, int, int, int], str]],
    t: float,
) -> list[tuple[tuple[int, int, int, int], str]]:
    """前後の採用boxを対応付け、座標を線形補間する。"""
    interpolated: list[tuple[tuple[int, int, int, int], str]] = []
    used_current: set[int] = set()

    for previous_box, previous_label in previous_boxes:
        previous_part = normalize_display_label(previous_label)
        previous_cx, previous_cy = box_center(previous_box)
        best_idx: int | None = None
        best_distance: float | None = None

        for idx, (current_box, current_label) in enumerate(current_boxes):
            if idx in used_current:
                continue
            if normalize_display_label(current_label) != previous_part:
                continue
            current_cx, current_cy = box_center(current_box)
            distance = (current_cx - previous_cx) ** 2 + (current_cy - previous_cy) ** 2
            if best_distance is None or distance < best_distance:
                best_idx = idx
                best_distance = distance

        if best_idx is None:
            continue

        current_box, _ = current_boxes[best_idx]
        used_current.add(best_idx)
        interpolated.append((interpolate_box(previous_box, current_box, t), previous_label))

    return interpolated


def draw_debug_frame(
    frame: np.ndarray,
    pose_keypoints: list[np.ndarray],
    search_boxes: list[tuple[int, int, int, int]],
    nudenet_boxes: list[tuple[tuple[int, int, int, int], float, str]],
    confidence: float,
    new_boxes: list[tuple[tuple[int, int, int, int], str]],
    applied_boxes: list[tuple[tuple[int, int, int, int], str]],
    frame_idx: int,
) -> np.ndarray:
    dbg = frame.copy()

    # ポーズスケルトン
    for kps in pose_keypoints:  # kps: np.ndarray [17, 3]
        for (a, b) in SKELETON_EDGES:
            if kps[a][2] > 0.3 and kps[b][2] > 0.3:
                p1 = (int(kps[a][0]), int(kps[a][1]))
                p2 = (int(kps[b][0]), int(kps[b][1]))
                cv2.line(dbg, p1, p2, (180, 180, 180), 2)
        for i, kp in enumerate(kps):
            if kp[2] > 0.3:
                x, y = int(kp[0]), int(kp[1])
                # 腰骨（11,12）は黄色で強調
                color = (0, 255, 255) if i in (11, 12) else (0, 220, 0)
                cv2.circle(dbg, (x, y), 6 if i in (11, 12) else 4, color, -1)
                cv2.putText(dbg, str(i), (x + 6, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 股間領域（NudeNet 検出フィルタ用、腰骨幅以内）（青・太線）
    for box in search_boxes:
        cv2.rectangle(dbg, (box[0], box[1]), (box[2], box[3]), (255, 80, 0), 2)
        cv2.putText(dbg, "crotch", (box[0] + 4, box[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 0), 2)

    # 検出結果（緑=OK, 赤=NG, 灰=LOW(confidence未満)）
    adopted_boxes = {b for b, _ in new_boxes}
    adopted_labels = {b: label for b, label in new_boxes}
    for box_idx, (box, score, label) in enumerate(nudenet_boxes, start=1):
        adopted = box in adopted_boxes
        above_conf = score >= confidence
        if adopted:
            color, tag = (0, 220, 0), "OK"
        elif above_conf:
            color, tag = (0, 0, 220), "NG"
        else:
            color, tag = (160, 160, 160), "LOW"
        thickness = 2 if adopted else 1
        display_label = adopted_labels.get(box, detector_display_label(label))
        model_name, part_label = split_model_label(display_label)
        cv2.rectangle(dbg, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(dbg, f"box_{box_idx} {model_name} mosaic({part_label})",
                    (box[0] + 4, box[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness)
        cv2.putText(dbg, f"score {score:.2f} [{tag}]",
                    (box[0] + 4, box[1] + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness)

    # フレーム番号
    cv2.putText(dbg, f"frame {frame_idx}", (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 0), 2)

    return dbg


def log(msg: str, log_file) -> None:
    print(msg)
    print(msg, file=log_file, flush=True)


def merge_audio_from_source(
    source_video_path: str,
    tmp_video_path: str,
    output_path: str,
    start_frame: int,
    fps: float,
) -> None:
    input_args = ["-i", source_video_path]
    if start_frame:
        input_args = ["-ss", f"{start_frame / fps:.6f}", "-i", source_video_path]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", tmp_video_path, *input_args,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0?", "-shortest",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(tmp_video_path)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg エラー:\n{r.stderr}")


def mosaic_csv_header() -> list[str]:
    header = ["frame_no", "checked"]
    for i in range(1, MAX_CSV_MOSAICS + 1):
        header.extend([
            f"mosaic{i}_on",
            f"mosaic{i}_type",
            f"mosaic{i}_x1",
            f"mosaic{i}_y1",
            f"mosaic{i}_x2",
            f"mosaic{i}_y2",
        ])
    return header


def mosaic_csv_row(
    frame_idx: int,
    boxes: list[tuple[tuple[int, int, int, int], str]],
) -> dict[str, str | int]:
    row: dict[str, str | int] = {
        "frame_no": frame_idx,
        "checked": "",
    }
    for i in range(1, MAX_CSV_MOSAICS + 1):
        row[f"mosaic{i}_on"] = "0"
        row[f"mosaic{i}_type"] = ""
        row[f"mosaic{i}_x1"] = ""
        row[f"mosaic{i}_y1"] = ""
        row[f"mosaic{i}_x2"] = ""
        row[f"mosaic{i}_y2"] = ""

    for i, (box, label) in enumerate(boxes[:MAX_CSV_MOSAICS], start=1):
        row[f"mosaic{i}_on"] = "1"
        row[f"mosaic{i}_type"] = label
        row[f"mosaic{i}_x1"] = box[0]
        row[f"mosaic{i}_y1"] = box[1]
        row[f"mosaic{i}_x2"] = box[2]
        row[f"mosaic{i}_y2"] = box[3]
    return row


def csv_on(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "on", "yes", "y"}


def boxes_from_csv_row(row: dict[str, str]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, MAX_CSV_MOSAICS + 1):
        if not csv_on(row.get(f"mosaic{i}_on")):
            continue
        try:
            box = (
                int(float(row.get(f"mosaic{i}_x1", ""))),
                int(float(row.get(f"mosaic{i}_y1", ""))),
                int(float(row.get(f"mosaic{i}_x2", ""))),
                int(float(row.get(f"mosaic{i}_y2", ""))),
            )
        except ValueError:
            continue
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return boxes


def process_video(
    input_path: str,
    output_path: str,
    block_size: int,
    confidence: float,
    detect_every: int,
    log_file=None,
    debug_path: str | None = None,
    interpolate: bool = True,
    yolo_nsfw_model_path: str | None = None,
    yolo_confidence: float = 0.03,
    max_interpolate_gap: int = 10,
    frame_range: tuple[int, int] | None = None,
    pose_backend: str = "yolo11",
    csv_path: str | None = None,
    render_debug_to_output: bool = False,
) -> None:
    def _log(msg: str) -> None:
        if log_file:
            log(msg, log_file)
        else:
            print(msg)

    try:
        from nudenet import NudeDetector
    except ImportError:
        print("エラー: nudenet がインストールされていません。", file=sys.stderr)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("エラー: ultralytics がインストールされていません。", file=sys.stderr)
        sys.exit(1)

    _log("モデル初期化中...")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "640m.onnx")
    detector = NudeDetector(model_path=model_path, inference_resolution=640,
                            providers=["CPUExecutionProvider"])
    _log(f"ポーズモデル: {pose_backend}")
    pose_model_bundle = load_pose_model(pose_backend)
    yolo_nsfw_model = None
    yolo_source = "YOLO"
    if yolo_nsfw_model_path:
        if not os.path.isfile(yolo_nsfw_model_path):
            raise RuntimeError(f"YOLO NSFWモデルが見つかりません: {yolo_nsfw_model_path}")
        yolo_source = yolo_source_name(yolo_nsfw_model_path)
        _log(f"YOLO NSFWモデル: {yolo_nsfw_model_path} ({yolo_source})")
        yolo_nsfw_model = YOLO(yolo_nsfw_model_path, task="detect")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = 0
    end_frame = max(0, total_frames - 1)
    if frame_range is not None:
        start_frame, end_frame = frame_range
        if start_frame < 0 or end_frame < start_frame:
            raise RuntimeError(f"フレーム範囲が不正です: {start_frame}-{end_frame}")
        if start_frame >= total_frames:
            raise RuntimeError(f"開始フレームが動画の総フレーム数を超えています: {start_frame} >= {total_frames}")
        end_frame = min(end_frame, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    process_frames = end_frame - start_frame + 1

    tmp_video_path = output_path + ".tmp_noaudio.mp4"
    writer = cv2.VideoWriter(
        tmp_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    tmp_debug_path = (debug_path + ".tmp_noaudio.mp4") if debug_path else None
    debug_writer = cv2.VideoWriter(
        tmp_debug_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    ) if debug_path else None

    last_boxes: list[tuple[tuple[int, int, int, int], str]] = []
    last_search_boxes: list[tuple[int, int, int, int]] = []
    last_nudenet_boxes: list = []
    last_new_boxes: list[tuple[tuple[int, int, int, int], str]] = []
    last_pose_kps: list[np.ndarray] = []

    pending_frames: list[dict] = []
    previous_positive_boxes: list[tuple[tuple[int, int, int, int], str]] = []
    previous_positive_idx: int | None = None
    csv_file = None
    csv_writer = None
    resolved_input_path = str(Path(input_path).resolve())
    if csv_path:
        csv_file = open(csv_path, "w", encoding="utf-8", newline="")
        meta = csv.writer(csv_file)
        meta.writerow(["source_video",    resolved_input_path])
        meta.writerow(["block_size",      block_size])
        meta.writerow(["confidence",      confidence])
        meta.writerow(["pose_model",      pose_backend])
        meta.writerow(["yolo_nsfw_model", yolo_nsfw_model_path or ""])
        meta.writerow(["yolo_confidence", yolo_confidence])
        meta.writerow(["detect_every",    detect_every])
        meta.writerow(["interpolate_gap", max_interpolate_gap])
        csv_writer = csv.DictWriter(csv_file, fieldnames=mosaic_csv_header())
        csv_writer.writeheader()

    def _write_out(
        frm: np.ndarray,
        apply_bxs: list[tuple[tuple[int, int, int, int], str]],
        pose_res: list,
        srch_bxs: list[tuple[int, int, int, int]],
        nnet_bxs: list,
        nw_bxs: list[tuple[tuple[int, int, int, int], str]],
        fidx: int,
    ) -> None:
        res = frm.copy()
        for box, _ in apply_bxs:
            res = apply_mosaic(res, *box, block_size)
        if render_debug_to_output or debug_writer:
            dbg = draw_debug_frame(res, pose_res, srch_bxs, nnet_bxs,
                                   confidence, nw_bxs, apply_bxs, fidx)
        if render_debug_to_output:
            writer.write(dbg)
        else:
            writer.write(res)
        if debug_writer:
            debug_writer.write(dbg)
        if csv_writer:
            csv_writer.writerow(mosaic_csv_row(fidx, apply_bxs))

    def _flush_pending(force: bool = False) -> None:
        nonlocal pending_frames, previous_positive_boxes, previous_positive_idx
        if not pending_frames:
            return

        current = pending_frames[-1]
        current_positive = bool(current["new_boxes"])
        if current_positive:
            frame_gap = current["idx"] - previous_positive_idx - 1 if previous_positive_idx is not None else 0
            can_interpolate = (
                interpolate
                and previous_positive_boxes
                and previous_positive_idx is not None
                and 0 < frame_gap <= max_interpolate_gap
            )
            for rec in pending_frames[:-1]:
                apply_boxes = rec["new_boxes"]
                if can_interpolate and not rec["new_boxes"]:
                    t = (rec["idx"] - previous_positive_idx) / (current["idx"] - previous_positive_idx)
                    apply_boxes = interpolate_adopted_boxes(
                        previous_positive_boxes, current["new_boxes"], t
                    )
                    if not apply_boxes:
                        apply_boxes = previous_positive_boxes
                    _log(
                        f"frame {rec['idx']}: interpolated linear "
                        f"(frame {previous_positive_idx} and frame {current['idx']} both detected)"
                    )
                _write_out(rec["frame"], apply_boxes, rec["pose"], rec["search"],
                           rec["nudenet"], rec["new_boxes"], rec["idx"])
            _write_out(current["frame"], current["new_boxes"], current["pose"],
                       current["search"], current["nudenet"], current["new_boxes"],
                       current["idx"])
            previous_positive_boxes = current["new_boxes"]
            previous_positive_idx = current["idx"]
            pending_frames = []
            return

        if not force and len(pending_frames) <= max_interpolate_gap:
            return

        flush_count = len(pending_frames) if force else 1
        for rec in pending_frames[:flush_count]:
            _write_out(rec["frame"], rec["new_boxes"], rec["pose"], rec["search"],
                       rec["nudenet"], rec["new_boxes"], rec["idx"])
        pending_frames = pending_frames[flush_count:]

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_frame_path = os.path.join(tmp_dir, "_crop.jpg")
            frame_idx = start_frame

            with tqdm(total=process_frames, desc="フレーム処理", unit="frame") as pbar:
                while True:
                    if frame_idx > end_frame:
                        _flush_pending(force=True)
                        break
                    ret, frame = cap.read()
                    if not ret:
                        _flush_pending(force=True)
                        break

                    if (frame_idx - start_frame) % detect_every == 0:
                        # Step 1: Pose で股間エリアを特定
                        search_boxes, pose_kps = get_crotch_boxes(frame, pose_model_bundle)
                        # Step 2: NudeNet で全体画像から全検出（0.01以上）を取得
                        all_nudenet_boxes = detect_genitalia_multicrop(
                            detector, frame, tmp_frame_path, 0.01
                        )
                        yolo_boxes = detect_yolo_genitalia_multicrop(
                            yolo_nsfw_model, frame, yolo_confidence, yolo_source
                        )
                        all_detector_boxes = all_nudenet_boxes + yolo_boxes
                        # Step 3: confidence 閾値でフィルタしてから crotch フィルタ適用
                        nudenet_boxes_conf = [
                            (box, score, label) for box, score, label in all_nudenet_boxes
                            if score >= confidence
                        ]
                        yolo_boxes_conf = [
                            (box, score, label) for box, score, label in yolo_boxes
                            if score >= yolo_confidence
                        ]
                        nudenet_adopted = [
                            (box, f"NudeNet {short_label}")
                            for box, short_label in filter_by_crotch(nudenet_boxes_conf, search_boxes)
                        ]
                        yolo_adopted = [
                            (box, short_label)
                            for box, short_label in filter_by_crotch(yolo_boxes_conf, search_boxes)
                        ]
                        new_boxes = merge_adopted_boxes(
                            nudenet_adopted + yolo_adopted
                        )

                        last_boxes = new_boxes
                        last_search_boxes = search_boxes
                        last_nudenet_boxes = all_detector_boxes
                        last_new_boxes = new_boxes
                        last_pose_kps = pose_kps

                        _log(
                            f"frame {frame_idx}: crotch_boxes={len(search_boxes)} "
                            f"nudenet_raw={len(all_nudenet_boxes)} nudenet_conf={len(nudenet_boxes_conf)} "
                            f"yolo_raw={len(yolo_boxes)} yolo_conf={len(yolo_boxes_conf)} "
                            f"adopted={len(new_boxes)}"
                        )
                        adopted_set = {b for b, _ in new_boxes}
                        for box_idx, (box, score, label) in enumerate(all_detector_boxes, start=1):
                            cx = (box[0] + box[2]) // 2
                            cy = (box[1] + box[3]) // 2
                            in_crotch = box_overlaps_any(box, search_boxes)
                            adopted = box in adopted_set
                            source = label.split(maxsplit=1)[0] if label.startswith((f"{yolo_source} ", "YOLO ", "erax ", "Felldude ", "Throaway ")) else "NudeNet"
                            above_conf = score >= (yolo_confidence if source != "NudeNet" else confidence)
                            _log(
                                f"  box_{box_idx} {source} label={label} center=({cx},{cy}) score={score:.2f} "
                                f"above_conf={above_conf} in_crotch={in_crotch} adopted={adopted}"
                            )

                        pending_frames.append({
                            'frame': frame, 'idx': frame_idx, 'new_boxes': new_boxes,
                            'search': search_boxes, 'nudenet': all_detector_boxes,
                            'pose': pose_kps,
                        })
                        _flush_pending()

                    else:
                        _flush_pending(force=True)
                        _write_out(frame, last_boxes, last_pose_kps, last_search_boxes,
                                   last_nudenet_boxes, last_new_boxes, frame_idx)

                    frame_idx += 1
                    pbar.update(1)
    finally:
        cap.release()
        writer.release()
        if debug_writer:
            debug_writer.release()
        if csv_file:
            csv_file.close()

    _log("音声をマージ中...")
    merge_audio_from_source(input_path, tmp_video_path, output_path, start_frame, fps)

    if debug_path:
        _log("デバッグ動画をマージ中...")
        merge_audio_from_source(input_path, tmp_debug_path, debug_path, start_frame, fps)


def process_single_frame(
    input_path: str,
    frame_number: int,
    block_size: int,
    confidence: float,
    log_file=None,
    debug: bool = False,
    yolo_nsfw_model_path: str | None = None,
    yolo_confidence: float = 0.03,
    pose_backend: str = "yolo11",
) -> None:
    def _log(msg: str) -> None:
        if log_file:
            log(msg, log_file)
        else:
            print(msg)

    try:
        from nudenet import NudeDetector
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"エラー: 必要なライブラリがインストールされていません: {exc}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {input_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        raise RuntimeError(f"フレーム番号が範囲外です: {frame_number} (0-{total_frames - 1})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"フレームを読み込めません: {frame_number}")

    _log("モデル初期化中...")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "640m.onnx")
    detector = NudeDetector(model_path=model_path, inference_resolution=640,
                            providers=["CPUExecutionProvider"])
    _log(f"ポーズモデル: {pose_backend}")
    pose_model_bundle = load_pose_model(pose_backend)
    yolo_nsfw_model = None
    yolo_source = "YOLO"
    if yolo_nsfw_model_path:
        from ultralytics import YOLO
        if not os.path.isfile(yolo_nsfw_model_path):
            raise RuntimeError(f"YOLO NSFWモデルが見つかりません: {yolo_nsfw_model_path}")
        yolo_source = yolo_source_name(yolo_nsfw_model_path)
        _log(f"YOLO NSFWモデル: {yolo_nsfw_model_path} ({yolo_source})")
        yolo_nsfw_model = YOLO(yolo_nsfw_model_path, task="detect")

    stem = Path(input_path).stem
    out_dir = Path(input_path).parent
    detector_suffix = detector_name_suffix(yolo_nsfw_model_path, pose_backend)
    source_path = out_dir / f"{stem}_{detector_suffix}_frame{frame_number}_source.jpg"
    censored_path = out_dir / f"{stem}_{detector_suffix}_frame{frame_number}_censored.jpg"
    debug_path = out_dir / f"{stem}_{detector_suffix}_frame{frame_number}_debug.jpg"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_frame_path = os.path.join(tmp_dir, "_crop.jpg")
        search_boxes, pose_kps = get_crotch_boxes(frame, pose_model_bundle)
        all_nudenet_boxes = detect_genitalia_multicrop(
            detector, frame, tmp_frame_path, 0.01
        )
        yolo_boxes = detect_yolo_genitalia_multicrop(
            yolo_nsfw_model, frame, yolo_confidence, yolo_source
        )
        nudenet_boxes_conf = [
            (box, score, label) for box, score, label in all_nudenet_boxes
            if score >= confidence
        ]
        yolo_boxes_conf = [
            (box, score, label) for box, score, label in yolo_boxes
            if score >= yolo_confidence
        ]
        nudenet_adopted = [
            (box, f"NudeNet {short_label}")
            for box, short_label in filter_by_crotch(nudenet_boxes_conf, search_boxes)
        ]
        yolo_adopted = [
            (box, short_label)
            for box, short_label in filter_by_crotch(yolo_boxes_conf, search_boxes)
        ]
        new_boxes = merge_adopted_boxes(nudenet_adopted + yolo_adopted)
        all_detector_boxes = all_nudenet_boxes + yolo_boxes

    result = frame.copy()
    for box, _ in new_boxes:
        result = apply_mosaic(result, *box, block_size)

    cv2.imwrite(str(source_path), frame)
    cv2.imwrite(str(censored_path), result)
    if debug:
        dbg = draw_debug_frame(result, pose_kps, search_boxes, all_detector_boxes,
                               confidence, new_boxes, new_boxes, frame_number)
        cv2.imwrite(str(debug_path), dbg)

    _log(
        f"frame {frame_number}: crotch_boxes={len(search_boxes)} "
        f"nudenet_raw={len(all_nudenet_boxes)} nudenet_conf={len(nudenet_boxes_conf)} "
        f"yolo_raw={len(yolo_boxes)} yolo_conf={len(yolo_boxes_conf)} "
        f"adopted={len(new_boxes)}"
    )
    adopted_set = {b for b, _ in new_boxes}
    for box_idx, (box, score, label) in enumerate(all_detector_boxes, start=1):
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        source = label.split(maxsplit=1)[0] if label.startswith(("YOLO ", "erax ", "Felldude ", "Throaway ")) else "NudeNet"
        above_conf = score >= (yolo_confidence if source != "NudeNet" else confidence)
        _log(
            f"  box_{box_idx} {source} label={label} center=({cx},{cy}) score={score:.2f} "
            f"above_conf={above_conf} in_crotch={box_overlaps_any(box, search_boxes)} "
            f"adopted={box in adopted_set}"
        )
    _log(f"source jpg  : {source_path}")
    _log(f"censored jpg: {censored_path}")
    if debug:
        _log(f"debug jpg   : {debug_path}")


def post_output_path_from_csv(csv_path: Path) -> Path:
    stem = csv_path.stem
    if stem.endswith("_pre"):
        return csv_path.with_name(f"{stem[:-4]}_post.mp4")
    return csv_path.with_name(f"{stem}_post.mp4")


def read_csv_meta_and_rows(
    csv_file_path: Path,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """
    先頭のメタデータ行（key,value 形式）を読み込み、
    frame_no から始まる列ヘッダ行以降をデータ行として返す。
    """
    meta: dict[str, str] = {}
    with open(csv_file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] == "frame_no":
                # この行が列ヘッダ → 残りをデータ行として読む
                fieldnames = row
                rows = list(csv.DictReader(f, fieldnames=fieldnames))
                return meta, rows
            if len(row) >= 2:
                meta[row[0]] = row[1]
    return meta, []


def process_post_from_csv(
    csv_path: str,
    output_path: str | None,
    log_file=None,
) -> None:
    def _log(msg: str) -> None:
        if log_file:
            log(msg, log_file)
        else:
            print(msg)

    csv_file_path = Path(csv_path)
    meta, rows = read_csv_meta_and_rows(csv_file_path)

    source_video = meta.get("source_video", "").strip()
    if not source_video:
        raise RuntimeError("CSVに source_video がありません")
    if not os.path.isfile(source_video):
        raise RuntimeError(f"CSVの source_video が見つかりません: {source_video}")

    try:
        block_size = int(meta.get("block_size", "15"))
    except ValueError:
        block_size = 15

    if not rows:
        raise RuntimeError(f"CSVにフレーム行がありません: {csv_path}")

    try:
        frame_rows = sorted(rows, key=lambda row: int(row["frame_no"]))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("CSVの frame_no が不正です") from exc

    dst = output_path or str(post_output_path_from_csv(csv_file_path))
    tmp_video_path = dst + ".tmp_noaudio.mp4"

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_frame = int(frame_rows[0]["frame_no"])
    writer = cv2.VideoWriter(
        tmp_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    try:
        current_pos: int | None = None
        with tqdm(total=len(frame_rows), desc="CSV清書", unit="frame") as pbar:
            for row in frame_rows:
                frame_idx = int(row["frame_no"])
                if current_pos != frame_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"フレームを読み込めません: {frame_idx}")
                current_pos = frame_idx + 1

                for box in boxes_from_csv_row(row):
                    frame = apply_mosaic(frame, *box, block_size)
                writer.write(frame)
                pbar.update(1)
    finally:
        cap.release()
        writer.release()

    _log("音声をマージ中...")
    merge_audio_from_source(source_video, tmp_video_path, dst, first_frame, fps)
    _log(f"完了: {dst}")


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("エラー: ffmpeg が見つかりません。", file=sys.stderr)
        sys.exit(1)


def parse_frame_range(value: str) -> tuple[int, int]:
    try:
        start_text, end_text = value.split("-", 1)
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("START-END 形式で指定してください。例: 100-130") from exc
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("フレーム範囲が不正です。例: 100-130")
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MP4動画の性器にモザイクをかけるツール（Pose + NudeNet 二段フィルタ）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
仕組み:
  1. YOLOv8 Pose で腰骨から股間エリアを特定
  2. NudeNet で3クロップ検出
  3. 股間エリア内の検出のみ採用（手・脇など誤検出を排除）

例:
  python mosaic_censor.py input.mp4
  python mosaic_censor.py input.mp4 --block-size 20
  python mosaic_censor.py input.mp4 --confidence 0.25
        """,
    )
    parser.add_argument("input", help="入力 MP4 ファイルのパス（--post の場合は CSV ファイル）")
    parser.add_argument("--pre", action="store_true",
                        help="目視確認用の _pre.mp4 と編集用CSVを作成する")
    parser.add_argument("--post", action="store_true",
                        help="CSVを読み込み、source_video から _post.mp4 を作成する")
    parser.add_argument("--block-size", type=int, default=15, metavar="N",
                        help="モザイクのブロックサイズ (デフォルト: 15)")
    parser.add_argument("--confidence", type=float, default=0.03, metavar="F",
                        help="NudeNet の信頼度閾値 (デフォルト: 0.03)")
    parser.add_argument("--detect-every", type=int, default=1, metavar="N",
                        help="N フレームごとに検出実行 (デフォルト: 1 = 全フレーム)")
    parser.add_argument("--debug", action="store_true",
                        help="ポーズスケルトン・検出ボックスを描画したデバッグ動画を出力")
    parser.add_argument("--no-interpolate", action="store_true",
                        help="前後フレーム補間を無効化 (デフォルト: 有効)")
    parser.add_argument("--yolo-nsfw-model", default=None,
                        help="追加で使うYOLO NSFWモデル(.pt)のパス (デフォルト: 無効)")
    parser.add_argument("--yolo-confidence", type=float, default=None, metavar="F",
                        help="YOLO NSFW の信頼度閾値 (デフォルト: EraX=0.3, その他=0.03)")
    parser.add_argument("--interpolate-gap", type=int, default=10, metavar="N",
                        help="前後検出で補間する最大連続未検出フレーム数 (デフォルト: 10)")
    parser.add_argument("--frames", type=parse_frame_range, metavar="START-END",
                        help="処理するフレーム範囲。例: 100-130 (デフォルト: 全フレーム)")
    parser.add_argument("--frame", type=int, metavar="N",
                        help="指定した1フレームだけをJPG画像で出力する。例: --frame 121")
    parser.add_argument(
        "--pose-model",
        choices=list(POSE_BACKENDS),
        default="yolo11",
        help="ポーズ検出バックエンド (デフォルト: yolo11)。vitpose-h は pip install transformers が必要",
    )

    args = parser.parse_args()
    check_ffmpeg()

    if not os.path.isfile(args.input):
        print(f"エラー: ファイルが見つかりません: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.pre and args.post:
        print("エラー: --pre と --post は同時に指定できません。", file=sys.stderr)
        sys.exit(1)
    if args.frame is not None and args.frames is not None:
        print("エラー: --frame と --frames は同時に指定できません。", file=sys.stderr)
        sys.exit(1)
    if args.frame is not None and (args.pre or args.post):
        print("エラー: --frame は --pre/--post と同時に指定できません。", file=sys.stderr)
        sys.exit(1)
    if args.post and args.frames is not None:
        print("エラー: --post はCSVのframe_noを使うため --frames は指定できません。", file=sys.stderr)
        sys.exit(1)

    input_path_obj = Path(args.input)
    stem = input_path_obj.stem
    detector_suffix = detector_name_suffix(args.yolo_nsfw_model, args.pose_model)
    yolo_confidence = effective_yolo_confidence(args.yolo_nsfw_model, args.yolo_confidence)
    if args.post:
        output_path = str(post_output_path_from_csv(input_path_obj))
        log_path = str(output_path[:-4] + "_log.txt")
        meta, _ = read_csv_meta_and_rows(input_path_obj)
        with open(log_path, "w", encoding="utf-8") as lf:
            log(f"CSV           : {args.input}", lf)
            log(f"出力          : {output_path}", lf)
            log(f"ログ          : {log_path}", lf)
            for key, value in meta.items():
                log(f"  {key}: {value}", lf)
            log("", lf)
            try:
                process_post_from_csv(
                    csv_path=args.input,
                    output_path=output_path,
                    log_file=lf,
                )
            except KeyboardInterrupt:
                print("\n中断されました", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"\nエラー: {e}", file=sys.stderr)
                sys.exit(1)
        return

    if args.frame is not None:
        log_path = str(Path(args.input).parent / f"{stem}_{detector_suffix}_frame{args.frame}_log.txt")
        with open(log_path, "w", encoding="utf-8") as lf:
            log(f"入力          : {args.input}", lf)
            log(f"ログ          : {log_path}", lf)
            log(f"単発フレーム  : {args.frame}", lf)
            log(f"ブロックサイズ : {args.block_size}", lf)
            log(f"信頼度閾値    : {args.confidence}", lf)
            log(f"ポーズモデル  : {args.pose_model}", lf)
            log(f"YOLO NSFW     : {args.yolo_nsfw_model or '無効'}", lf)
            if args.yolo_nsfw_model:
                log(f"YOLO信頼度閾値: {yolo_confidence}", lf)
            log("", lf)
            try:
                process_single_frame(
                    input_path=args.input,
                    frame_number=args.frame,
                    block_size=args.block_size,
                    confidence=args.confidence,
                    log_file=lf,
                    debug=args.debug,
                    yolo_nsfw_model_path=args.yolo_nsfw_model,
                    yolo_confidence=yolo_confidence,
                    pose_backend=args.pose_model,
                )
                log("\n完了", lf)
            except KeyboardInterrupt:
                print("\n中断されました", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"\nエラー: {e}", file=sys.stderr)
                sys.exit(1)
        return

    range_suffix = ""
    if args.frames:
        range_suffix = f"_frames{args.frames[0]}-{args.frames[1]}"
    if args.pre:
        output_path = str(Path(args.input).parent / f"{stem}{range_suffix}_pre.mp4")
        csv_path = str(Path(args.input).parent / f"{stem}{range_suffix}_pre.csv")
        log_path = str(Path(args.input).parent / f"{stem}{range_suffix}_pre_log.txt")
        with open(log_path, "w", encoding="utf-8") as lf:
            log(f"入力          : {args.input}", lf)
            log(f"pre動画       : {output_path}", lf)
            log(f"CSV           : {csv_path}", lf)
            log(f"ログ          : {log_path}", lf)
            log(f"ブロックサイズ : {args.block_size}", lf)
            log(f"信頼度閾値    : {args.confidence}", lf)
            log(f"ポーズモデル  : {args.pose_model}", lf)
            log(f"YOLO NSFW     : {args.yolo_nsfw_model or '無効'}", lf)
            if args.yolo_nsfw_model:
                log(f"YOLO信頼度閾値: {yolo_confidence}", lf)
            log(f"フレーム範囲  : {args.frames[0]}-{args.frames[1]}" if args.frames else "フレーム範囲  : 全フレーム", lf)
            log(f"検出間隔      : {args.detect_every} フレームごと", lf)
            log(f"フレーム補間  : {'無効' if args.no_interpolate else '有効'}", lf)
            log(f"補間最大Gap   : {args.interpolate_gap}", lf)
            log("", lf)

            try:
                process_video(
                    input_path=args.input,
                    output_path=output_path,
                    block_size=args.block_size,
                    confidence=args.confidence,
                    detect_every=args.detect_every,
                    log_file=lf,
                    debug_path=None,
                    interpolate=not args.no_interpolate,
                    yolo_nsfw_model_path=args.yolo_nsfw_model,
                    yolo_confidence=yolo_confidence,
                    max_interpolate_gap=args.interpolate_gap,
                    frame_range=args.frames,
                    pose_backend=args.pose_model,
                    csv_path=csv_path,
                    render_debug_to_output=True,
                )
                log(f"\n完了: {output_path}", lf)
            except KeyboardInterrupt:
                print("\n中断されました", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"\nエラー: {e}", file=sys.stderr)
                sys.exit(1)
        return

    output_path = str(Path(args.input).parent / f"{stem}_{detector_suffix}{range_suffix}_censored.mp4")
    log_path = str(Path(args.input).parent / f"{stem}_{detector_suffix}{range_suffix}_log.txt")
    debug_path = str(Path(args.input).parent / f"{stem}_{detector_suffix}{range_suffix}_debug.mp4") if args.debug else None

    with open(log_path, "w", encoding="utf-8") as lf:
        log(f"入力          : {args.input}", lf)
        log(f"出力          : {output_path}", lf)
        log(f"ログ          : {log_path}", lf)
        if debug_path:
            log(f"デバッグ動画  : {debug_path}", lf)
        log(f"ブロックサイズ : {args.block_size}", lf)
        log(f"信頼度閾値    : {args.confidence}", lf)
        log(f"ポーズモデル  : {args.pose_model}", lf)
        log(f"YOLO NSFW     : {args.yolo_nsfw_model or '無効'}", lf)
        if args.yolo_nsfw_model:
            log(f"YOLO信頼度閾値: {yolo_confidence}", lf)
        log(f"フレーム範囲  : {args.frames[0]}-{args.frames[1]}" if args.frames else "フレーム範囲  : 全フレーム", lf)
        log(f"検出間隔      : {args.detect_every} フレームごと", lf)
        log(f"フレーム補間  : {'無効' if args.no_interpolate else '有効'}", lf)
        log(f"補間最大Gap   : {args.interpolate_gap}", lf)
        log("", lf)

        try:
            process_video(
                input_path=args.input,
                output_path=output_path,
                block_size=args.block_size,
                confidence=args.confidence,
                detect_every=args.detect_every,
                log_file=lf,
                debug_path=debug_path,
                interpolate=not args.no_interpolate,
                yolo_nsfw_model_path=args.yolo_nsfw_model,
                yolo_confidence=yolo_confidence,
                max_interpolate_gap=args.interpolate_gap,
                frame_range=args.frames,
                pose_backend=args.pose_model,
            )
            log(f"\n完了: {output_path}", lf)
        except KeyboardInterrupt:
            print("\n中断されました", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"\nエラー: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
