#!/usr/bin/env python3
"""并排同步动作对比（A 阶核心功能）

把两段挥拍放进同一画布：各自指定击球时刻对齐 t=0，人物按体型（小腿长）
归一化缩放、以髋部为中心裁剪——不同拍摄距离、不同身高的两段视频可以直接对比。

用法：
  compare.py A.MOV --ta 39.47 B.MOV --tb 50.89 [--labels 我的动作,参考动作]
输出：
  out/compare_*.mp4          并排同步播放（含骨骼与实时角度）
  out/compare_*_contact.png  击球帧静态对比图
  out/compare_*_metrics.json 关键时刻双侧指标
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions, vision
from analyze import (CONNECTIONS, LEFT_IDS, VIS_TH, angle_deg,
                     L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)

CANVAS_W, CANVAS_H = 500, 720
BOX_W_SHIN, BOX_H_SHIN = 5.0, 7.0   # 裁剪框尺寸（小腿长倍数）
HIP_AT = (0.5, 0.42)                # 髋中点在框内的位置
BAR_H, LABEL_H = 72, 46

CJK_FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"


def make_landmarker():
    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(Path(__file__).parent / "models/pose_landmarker_full.task"),
            delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5))


def load_clip(video, t_contact, before, after):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0 = max(0, int((t_contact - before) * fps))
    f1 = int((t_contact + after) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    lm = make_landmarker()
    frames, pts, rel = [], [], []
    for fi in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = lm.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb), int(fi * 1000 / fps))
        p = np.full((33, 3), np.nan, np.float32)
        if res.pose_landmarks:
            for j, q in enumerate(res.pose_landmarks[0]):
                p[j] = (q.x * w, q.y * h, q.visibility)
        frames.append(frame)
        pts.append(p)
        rel.append(fi / fps - t_contact)
    cap.release()
    lm.close()
    return frames, np.array(pts), np.array(rel), fps


def clip_geometry(pts, fps):
    """裁剪几何：逐帧平滑髋中心 + 全段中位小腿长（缩放基准）"""
    hips = np.where((pts[:, L_HIP, 2:3] > VIS_TH) & (pts[:, R_HIP, 2:3] > VIS_TH),
                    (pts[:, L_HIP, :2] + pts[:, R_HIP, :2]) / 2, np.nan)
    n = len(hips)
    idx = np.where(~np.isnan(hips[:, 0]))[0]
    if len(idx) == 0:
        raise SystemExit("片段内没有任何有效姿态，无法对比")
    cx = np.interp(np.arange(n), idx, hips[idx, 0])
    cy = np.interp(np.arange(n), idx, hips[idx, 1])
    k = max(3, int(fps * 0.3)) | 1
    kern = np.ones(k) / k
    cx = np.convolve(np.pad(cx, k // 2, mode="edge"), kern, mode="valid")[:n]
    cy = np.convolve(np.pad(cy, k // 2, mode="edge"), kern, mode="valid")[:n]
    shin = np.nanmedian(
        (np.linalg.norm(pts[:, L_KNEE, :2] - pts[:, L_ANKLE, :2], axis=1)
         + np.linalg.norm(pts[:, R_KNEE, :2] - pts[:, R_ANKLE, :2], axis=1)) / 2)
    return cx, cy, float(shin)


def render_side(frame, p, cx, cy, shin):
    """以髋为中心、按小腿长归一化，渲染单侧画布并画骨骼"""
    bw, bh = BOX_W_SHIN * shin, BOX_H_SHIN * shin
    x0, y0 = cx - bw * HIP_AT[0], cy - bh * HIP_AT[1]
    H, W = frame.shape[:2]
    # 越界时补边
    pad_l, pad_t = max(0, -int(x0)), max(0, -int(y0))
    pad_r, pad_b = max(0, int(x0 + bw) - W), max(0, int(y0 + bh) - H)
    img = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    crop = img[int(y0) + pad_t:int(y0 + bh) + pad_t, int(x0) + pad_l:int(x0 + bw) + pad_l]
    canvas = cv2.resize(crop, (CANVAS_W, CANVAS_H))
    sx, sy = CANVAS_W / bw, CANVAS_H / bh

    draw = {}
    for j in range(33):
        if p[j, 2] > VIS_TH:
            px, py = (p[j, 0] - x0) * sx, (p[j, 1] - y0) * sy
            if -20 <= px <= CANVAS_W + 20 and -20 <= py <= CANVAS_H + 20:
                draw[j] = (int(px), int(py))
    for a, b in CONNECTIONS:
        if a in draw and b in draw:
            cv2.line(canvas, draw[a], draw[b], (80, 220, 80), 2, cv2.LINE_AA)
    for j, q in draw.items():
        cv2.circle(canvas, q, 4, (60, 120, 255) if j in LEFT_IDS else (255, 160, 40), -1, cv2.LINE_AA)
    return canvas


def side_metrics(p):
    def a(h, k, an):
        if p[h, 2] > VIS_TH and p[k, 2] > VIS_TH and p[an, 2] > VIS_TH:
            return angle_deg(p[h, :2], p[k, :2], p[an, :2])
        return np.nan
    kl, kr = a(L_HIP, L_KNEE, L_ANKLE), a(R_HIP, R_KNEE, R_ANKLE)
    st = np.nan
    if all(p[j, 2] > VIS_TH for j in (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)):
        shin = (np.linalg.norm(p[L_KNEE, :2] - p[L_ANKLE, :2])
                + np.linalg.norm(p[R_KNEE, :2] - p[R_ANKLE, :2])) / 2
        if shin > 1:
            st = abs(p[L_ANKLE, 0] - p[R_ANKLE, 0]) / shin
    return kl, kr, st


def label_bar(labels, width):
    bar = PILImage.new("RGB", (width, LABEL_H), (24, 24, 24))
    d = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype(CJK_FONT, 24)
    except OSError:
        font = ImageFont.load_default()
    for i, text in enumerate(labels):
        cx = width // 4 if i == 0 else width * 3 // 4
        tw = d.textlength(text, font=font)
        d.text((cx - tw / 2, (LABEL_H - 28) / 2), text, fill=(235, 235, 235), font=font)
    return cv2.cvtColor(np.array(bar), cv2.COLOR_RGB2BGR)


def fmt(v, suffix=""):
    return "--" if np.isnan(v) else f"{v:.0f}{suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_a")
    ap.add_argument("video_b")
    ap.add_argument("--ta", type=float, required=True, help="A 侧击球时刻（秒）")
    ap.add_argument("--tb", type=float, required=True, help="B 侧击球时刻（秒）")
    ap.add_argument("--before", type=float, default=1.2)
    ap.add_argument("--after", type=float, default=1.0)
    ap.add_argument("--labels", default="A,B")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    labels = args.labels.split(",")
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    stem = (f"compare_{Path(args.video_a).stem}_{args.ta:.1f}s"
            f"_vs_{Path(args.video_b).stem}_{args.tb:.1f}s")

    sides = []
    for video, t in ((args.video_a, args.ta), (args.video_b, args.tb)):
        print(f"提取姿态: {Path(video).name} @ {t:.2f}s ...", flush=True)
        frames, pts, rel, fps = load_clip(Path(video).expanduser(), t, args.before, args.after)
        cx, cy, shin = clip_geometry(pts, fps)
        sides.append(dict(frames=frames, pts=pts, rel=rel, fps=fps, cx=cx, cy=cy, shin=shin))

    out_fps = 60.0
    total_w = CANVAS_W * 2 + 4
    total_h = LABEL_H + CANVAS_H + BAR_H
    writer = cv2.VideoWriter(str(out_dir / f"{stem}.mp4"),
                             cv2.VideoWriter_fourcc(*"avc1"), out_fps, (total_w, total_h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_dir / f"{stem}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (total_w, total_h))
    top = label_bar(labels, total_w)

    contact_png, metrics = None, {"labels": labels, "at": {}}
    steps = np.arange(-args.before, args.after, 1 / out_fps)
    for rel_t in steps:
        cols, mrow = [], []
        for s in sides:
            i = int(np.clip(np.searchsorted(s["rel"], rel_t), 0, len(s["frames"]) - 1))
            canvas = render_side(s["frames"][i], s["pts"][i], s["cx"][i], s["cy"][i], s["shin"])
            mrow.append(side_metrics(s["pts"][i]))
            cols.append(canvas)
        row = np.hstack([cols[0], np.full((CANVAS_H, 4, 3), 24, np.uint8), cols[1]])
        bar = np.full((BAR_H, total_w, 3), 24, np.uint8)
        for k, (kl, kr, st) in enumerate(mrow):
            x0 = 16 if k == 0 else total_w // 2 + 18
            cv2.putText(bar, f"knee L{fmt(kl)} R{fmt(kr)}", (x0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(bar, f"stance {fmt(st)}" if np.isnan(st) else f"stance {st:.2f}",
                        (x0, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(bar, f"t={rel_t:+.2f}s", (total_w // 2 - 52, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, (90, 200, 255), 2, cv2.LINE_AA)
        frame_out = np.vstack([top, row, bar])
        writer.write(frame_out)

        key = None
        if abs(rel_t) < 0.5 / out_fps:
            key = "contact"
            contact_png = frame_out.copy()
        elif abs(rel_t + 0.5) < 0.5 / out_fps:
            key = "t-0.5s"
        if key:
            metrics["at"][key] = [
                {"knee_left": None if np.isnan(kl) else round(float(kl), 1),
                 "knee_right": None if np.isnan(kr) else round(float(kr), 1),
                 "stance_width": None if np.isnan(st) else round(float(st), 2)}
                for kl, kr, st in mrow]
    writer.release()

    if contact_png is not None:
        cv2.imwrite(str(out_dir / f"{stem}_contact.png"), contact_png)
    (out_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n输出: {out_dir}/{stem}.mp4 / _contact.png / _metrics.json")


if __name__ == "__main__":
    main()
