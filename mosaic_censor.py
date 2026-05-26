#!/usr/bin/env python3
"""
mosaic_censor.py - MP4動画の性器にモザイクをかけるツール

使い方:
  python mosaic_censor.py input.mp4
  python mosaic_censor.py input.mp4 --block-size 20
  python mosaic_censor.py input.mp4 --confidence 0.25
"""

import argparse
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


def get_crotch_boxes(frame: np.ndarray, pose_model) -> tuple[
    list[tuple[int, int, int, int]],
    list,
]:
    """
    YOLOv8 Pose で腰骨ランドマークを検出し、2つを返す。
    - search_boxes: NudeNet 検出フィルタ用ボックス
    - pose_results: デバッグ描画用の生の Pose 推論結果
    """
    H, W = frame.shape[:2]
    results = pose_model(frame, verbose=False, device="cpu")
    search_boxes = []
    for r in results:
        if r.keypoints is None or r.keypoints.data.shape[0] == 0:
            continue
        for kps in r.keypoints.data:
            lhip = kps[11]  # LEFT_HIP  [x, y, conf]
            rhip = kps[12]  # RIGHT_HIP
            if float(lhip[2]) < 0.3 or float(rhip[2]) < 0.3:
                continue
            lhip_x = float(lhip[0])
            lhip_y = float(lhip[1])
            rhip_x = float(rhip[0])
            rhip_y = float(rhip[1])

            # 11-12 の画面上の直線距離を基準にする。
            dx = lhip_x - rhip_x
            dy = lhip_y - rhip_y
            hip_w = max(int((dx * dx + dy * dy) ** 0.5), 40)
            cx = int((lhip_x + rhip_x) / 2)
            cy = int((lhip_y + rhip_y) / 2)

            # 検出フィルタ用: 横幅・縦幅ともに腰骨幅の3倍
            search_boxes.append((
                max(0, cx - hip_w * 3 // 2),
                max(0, cy - hip_w * 3 // 2),
                min(W, cx + hip_w * 3 // 2),
                min(H, cy + hip_w * 3 // 2),
            ))
    return search_boxes, results


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


def detector_name_suffix(yolo_nsfw_model_path: str | None) -> str:
    if not yolo_nsfw_model_path:
        return "NudeNet"
    return f"NudeNet_{Path(yolo_nsfw_model_path).stem}"


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


def draw_debug_frame(
    frame: np.ndarray,
    pose_results,
    search_boxes: list[tuple[int, int, int, int]],
    nudenet_boxes: list[tuple[tuple[int, int, int, int], float, str]],
    confidence: float,
    new_boxes: list[tuple[tuple[int, int, int, int], str]],
    applied_boxes: list[tuple[tuple[int, int, int, int], str]],
    frame_idx: int,
) -> np.ndarray:
    dbg = frame.copy()

    # ポーズスケルトン
    for r in pose_results:
        if r.keypoints is None or r.keypoints.data.shape[0] == 0:
            continue
        for kps in r.keypoints.data:
            for (a, b) in SKELETON_EDGES:
                if float(kps[a][2]) > 0.3 and float(kps[b][2]) > 0.3:
                    p1 = (int(float(kps[a][0])), int(float(kps[a][1])))
                    p2 = (int(float(kps[b][0])), int(float(kps[b][1])))
                    cv2.line(dbg, p1, p2, (180, 180, 180), 2)
            for i, kp in enumerate(kps):
                if float(kp[2]) > 0.3:
                    x, y = int(float(kp[0])), int(float(kp[1]))
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
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.putText(dbg, f"frame {frame_idx}", (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 1)

    return dbg


def log(msg: str, log_file) -> None:
    print(msg)
    print(msg, file=log_file, flush=True)


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
    max_interpolate_gap: int = 1,
    frame_range: tuple[int, int] | None = None,
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
    pose_model = YOLO("yolo11l-pose.pt", task="pose")
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
    last_pose_results: list = []

    pending_frames: list[dict] = []
    previous_positive_boxes: list[tuple[tuple[int, int, int, int], str]] = []
    previous_positive_idx: int | None = None

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
        writer.write(res)
        if debug_writer:
            dbg = draw_debug_frame(res, pose_res, srch_bxs, nnet_bxs,
                                   confidence, nw_bxs, apply_bxs, fidx)
            debug_writer.write(dbg)

    def _flush_pending(force: bool = False) -> None:
        nonlocal pending_frames, previous_positive_boxes, previous_positive_idx
        if not pending_frames:
            return

        current = pending_frames[-1]
        current_positive = bool(current["new_boxes"])
        if current_positive:
            gap = len(pending_frames) - 1
            can_interpolate = (
                interpolate
                and previous_positive_boxes
                and previous_positive_idx is not None
                and 0 < gap <= max_interpolate_gap
            )
            for rec in pending_frames[:-1]:
                apply_boxes = previous_positive_boxes if can_interpolate and not rec["new_boxes"] else rec["new_boxes"]
                if apply_boxes is previous_positive_boxes:
                    _log(
                        f"frame {rec['idx']}: interpolated "
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
                    search_boxes, pose_results = get_crotch_boxes(frame, pose_model)
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
                    last_pose_results = pose_results

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
                        'pose': pose_results,
                    })
                    _flush_pending()

                else:
                    _flush_pending(force=True)
                    _write_out(frame, last_boxes, last_pose_results, last_search_boxes,
                               last_nudenet_boxes, last_new_boxes, frame_idx)

                frame_idx += 1
                pbar.update(1)

    cap.release()
    writer.release()
    if debug_writer:
        debug_writer.release()

    def ffmpeg_merge(tmp: str, dst: str) -> None:
        input_args = ["-i", input_path]
        if start_frame:
            input_args = ["-ss", f"{start_frame / fps:.6f}", "-i", input_path]
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", tmp, *input_args,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0?", "-shortest",
            dst,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        os.remove(tmp)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg エラー:\n{r.stderr}")

    _log("音声をマージ中...")
    ffmpeg_merge(tmp_video_path, output_path)

    if debug_path:
        _log("デバッグ動画をマージ中...")
        ffmpeg_merge(tmp_debug_path, debug_path)


def process_single_frame(
    input_path: str,
    frame_number: int,
    block_size: int,
    confidence: float,
    log_file=None,
    debug: bool = False,
    yolo_nsfw_model_path: str | None = None,
    yolo_confidence: float = 0.03,
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
    pose_model = YOLO("yolo11l-pose.pt", task="pose")
    yolo_nsfw_model = None
    yolo_source = "YOLO"
    if yolo_nsfw_model_path:
        if not os.path.isfile(yolo_nsfw_model_path):
            raise RuntimeError(f"YOLO NSFWモデルが見つかりません: {yolo_nsfw_model_path}")
        yolo_source = yolo_source_name(yolo_nsfw_model_path)
        _log(f"YOLO NSFWモデル: {yolo_nsfw_model_path} ({yolo_source})")
        yolo_nsfw_model = YOLO(yolo_nsfw_model_path, task="detect")

    stem = Path(input_path).stem
    out_dir = Path(input_path).parent
    detector_suffix = detector_name_suffix(yolo_nsfw_model_path)
    source_path = out_dir / f"{stem}_{detector_suffix}_frame{frame_number}_source.jpg"
    censored_path = out_dir / f"{stem}_{detector_suffix}_frame{frame_number}_censored.jpg"
    debug_path = out_dir / f"{stem}_{detector_suffix}_frame{frame_number}_debug.jpg"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_frame_path = os.path.join(tmp_dir, "_crop.jpg")
        search_boxes, pose_results = get_crotch_boxes(frame, pose_model)
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
        dbg = draw_debug_frame(result, pose_results, search_boxes, all_detector_boxes,
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
    parser.add_argument("input", help="入力 MP4 ファイルのパス")
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
    parser.add_argument("--interpolate-gap", type=int, default=1, metavar="N",
                        help="前後検出で補間する最大連続未検出フレーム数 (デフォルト: 1)")
    parser.add_argument("--frames", type=parse_frame_range, metavar="START-END",
                        help="処理するフレーム範囲。例: 100-130 (デフォルト: 全フレーム)")
    parser.add_argument("--frame", type=int, metavar="N",
                        help="指定した1フレームだけをJPG画像で出力する。例: --frame 121")

    args = parser.parse_args()
    check_ffmpeg()

    if not os.path.isfile(args.input):
        print(f"エラー: ファイルが見つかりません: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.frame is not None and args.frames is not None:
        print("エラー: --frame と --frames は同時に指定できません。", file=sys.stderr)
        sys.exit(1)

    stem = Path(args.input).stem
    detector_suffix = detector_name_suffix(args.yolo_nsfw_model)
    yolo_confidence = effective_yolo_confidence(args.yolo_nsfw_model, args.yolo_confidence)
    if args.frame is not None:
        log_path = str(Path(args.input).parent / f"{stem}_{detector_suffix}_frame{args.frame}_log.txt")
        with open(log_path, "w", encoding="utf-8") as lf:
            log(f"入力          : {args.input}", lf)
            log(f"ログ          : {log_path}", lf)
            log(f"単発フレーム  : {args.frame}", lf)
            log(f"ブロックサイズ : {args.block_size}", lf)
            log(f"信頼度閾値    : {args.confidence}", lf)
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
