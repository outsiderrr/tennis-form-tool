#!/usr/bin/env python3
"""网球脚步分析流水线 v0（A 阶试点）

输入一段训练视频，输出：
  out/<name>_overlay.mp4   骨骼叠加视频
  out/<name>_landmarks.npz 每帧关键点原始数据（复用）
  out/<name>_metrics.json  脚步指标摘要
  out/<name>_footwork.png  脚步时间线图（屈膝角 / 站位宽度 / 重心高度）

约定：机位在球员身后（背面视角），下半身指标为主。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions, vision

# BlazePose 33 点索引
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

# 叠加视频里要画的连接（下半身为主 + 能看到的上肢）
CONNECTIONS = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (L_ANKLE, L_HEEL), (L_HEEL, L_FOOT), (L_ANKLE, L_FOOT),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (R_ANKLE, R_HEEL), (R_HEEL, R_FOOT), (R_ANKLE, R_FOOT),
]
LEFT_IDS = {L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_FOOT}

VIS_TH = 0.5  # 低于此可见度的点不用于指标
NUM_POSES = 3  # 多人场景检测上限，之后按「最近的人」选目标


def pick_target(pose_list, w, h, prev=None, canonical=None):
    """多人场景锁定目标。规则：
    1) 与上一帧目标位置连续的候选优先；
    2) 否则若某候选明显更大（更近）则取它；
    3) 目标短暂丢失（贴近镜头被裁）时不切到远处小人物——宁可留空，也不锁错。
    prev = (cx, cy, height_px, missing_frames)；canonical = 目标人物的长期中位身高（像素）
    任何时候都不接受身高 < 0.65×canonical 的候选（手持跟拍假设：目标在画面里大小基本稳定）
    """
    if not pose_list:
        return None
    cands = []
    for lm in pose_list:
        vis_pts = [p for p in lm if p.visibility > 0.3]
        if len(vis_pts) < 6:
            continue
        ys = np.array([p.y for p in vis_pts]) * h
        xs = np.array([p.x for p in vis_pts]) * w
        height = ys.max() - ys.min()
        if height > 1.6 * h or height < 0.05 * h:  # 垃圾检测（点散布到画面外）或噪点
            continue
        cands.append((height, xs.mean(), ys.mean(), lm))
    if canonical:
        cands = [c for c in cands if c[0] >= 0.65 * canonical]
    if not cands:
        return None
    if prev is None or prev[3] > 180:  # 从未锁定，或目标丢失超过 3s → 重新获取
        return max(cands, key=lambda c: c[0])[3]
    pcx, pcy, ph, missing = prev
    cands.sort(key=lambda c: np.hypot(c[1] - pcx, c[2] - pcy))
    nearest = cands[0]
    if np.hypot(nearest[1] - pcx, nearest[2] - pcy) < 0.6 * ph and nearest[0] > 0.5 * ph:
        return nearest[3]
    biggest = max(cands, key=lambda c: c[0])
    # 候选比目标小得多（<0.5 倍身高）→ 是远处别人，永远不切换（宁可留空）
    if biggest[0] < 0.5 * ph:
        return None
    # 目标刚丢（<2s）且候选明显偏小 → 也先等一等
    if missing < 120 and biggest[0] < 0.7 * ph:
        return None
    return biggest[3]


def angle_deg(a, b, c):
    """b 处的夹角（度），a/b/c 为 (x, y)"""
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return np.nan
    cosv = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def run_pose(video, start, end, out_dir, name, overlay_width, det_conf=0.5):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"无法打开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = overlay_width / w
    ow, oh = overlay_width, int(round(h * scale / 2) * 2)

    start_f = int(start * fps)
    end_f = int(end * fps) if end else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(Path(__file__).parent / "models/pose_landmarker_full.task"),
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=det_conf,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    prev_target = None
    accepted_heights = []  # 目标长期身高（像素），用于身份约束

    writer = cv2.VideoWriter(str(out_dir / f"{name}_overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"avc1"), fps, (ow, oh))
    if not writer.isOpened():  # avc1 不可用时退回 mp4v
        writer = cv2.VideoWriter(str(out_dir / f"{name}_overlay.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

    n_frames = end_f - start_f
    pts = np.full((n_frames, 33, 3), np.nan, dtype=np.float32)  # x_px, y_px, visibility
    times = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            pts, times = pts[:i], times[:i]
            break
        ts_ms = int((start_f + i) * 1000 / fps)
        times[i] = (start_f + i) / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(Image(image_format=ImageFormat.SRGB, data=rgb), ts_ms)

        small = cv2.resize(frame, (ow, oh))
        canonical = float(np.median(accepted_heights)) if len(accepted_heights) >= 30 else None
        lm = pick_target(result.pose_landmarks, w, h, prev_target, canonical)
        if lm is not None:
            for j, p in enumerate(lm):
                pts[i, j] = (p.x * w, p.y * h, p.visibility)
            ys = pts[i, :, 1]
            hgt = float(ys.max() - ys.min())
            prev_target = (float(pts[i, :, 0].mean()), float(ys.mean()), hgt, 0)
            accepted_heights.append(hgt)
            if len(accepted_heights) > 900:
                accepted_heights.pop(0)
            draw = {j: (int(pts[i, j, 0] * scale), int(pts[i, j, 1] * scale))
                    for j in range(33) if pts[i, j, 2] > VIS_TH and 0 <= pts[i, j, 1] < h}
            for a, b in CONNECTIONS:
                if a in draw and b in draw:
                    cv2.line(small, draw[a], draw[b], (80, 220, 80), 2, cv2.LINE_AA)
            for j, p in draw.items():
                cv2.circle(small, p, 4, (60, 120, 255) if j in LEFT_IDS else (255, 160, 40), -1, cv2.LINE_AA)
        elif prev_target is not None:
            prev_target = (*prev_target[:3], prev_target[3] + 1)
        cv2.putText(small, f"t={times[i]:.2f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(small)
        if i % 600 == 0:
            print(f"  {i}/{n_frames} 帧", flush=True)

    cap.release()
    writer.release()
    landmarker.close()
    return pts, times, fps


def compute_metrics(pts, times, fps):
    def vis(j):
        return pts[:, j, 2] > VIS_TH

    def xy(j):
        return pts[:, j, :2]

    m = {}
    # 屈膝角：髋-膝-踝
    knee_l = np.array([angle_deg(xy(L_HIP)[i], xy(L_KNEE)[i], xy(L_ANKLE)[i])
                       if (vis(L_HIP)[i] and vis(L_KNEE)[i] and vis(L_ANKLE)[i]) else np.nan
                       for i in range(len(times))])
    knee_r = np.array([angle_deg(xy(R_HIP)[i], xy(R_KNEE)[i], xy(R_ANKLE)[i])
                       if (vis(R_HIP)[i] and vis(R_KNEE)[i] and vis(R_ANKLE)[i]) else np.nan
                       for i in range(len(times))])

    # 站位宽度：踝间水平距离 / 平均小腿长（尺度归一，背面视角稳健）
    shin = (np.linalg.norm(xy(L_KNEE) - xy(L_ANKLE), axis=1)
            + np.linalg.norm(xy(R_KNEE) - xy(R_ANKLE), axis=1)) / 2
    ankle_dx = np.abs(xy(L_ANKLE)[:, 0] - xy(R_ANKLE)[:, 0])
    stance = np.where((vis(L_ANKLE) & vis(R_ANKLE) & vis(L_KNEE) & vis(R_KNEE) & (shin > 1)),
                      ankle_dx / shin, np.nan)

    # 重心高度代理：髋中点 y（像素，向下为正 → 取负让「高」朝上）
    hip_ok = vis(L_HIP) & vis(R_HIP)
    hip_y = np.where(hip_ok, (xy(L_HIP)[:, 1] + xy(R_HIP)[:, 1]) / 2, np.nan)

    # 分腿垫步候选 v1：双踝相对「地面基线」的短促抬升。
    # 基线取踝部 y 的 1.5s 滚动中位数——人走近走远时基线跟着走，透视稳健；
    # 只有双脚同时短促离地（0.05~0.45s、抬升 >0.12 小腿长）才算垫步/起跳。
    hops = []
    med_shin = np.nanmedian(shin)
    ank_ok = vis(L_ANKLE) & vis(R_ANKLE)
    ank_y = np.where(ank_ok, (xy(L_ANKLE)[:, 1] + xy(R_ANKLE)[:, 1]) / 2, np.nan)
    idx = np.where(~np.isnan(ank_y))[0]
    if len(idx) > fps and med_shin > 1:
        ay = np.interp(np.arange(len(ank_y)), idx, ank_y[idx])
        win = int(fps * 1.5) | 1
        pad = win // 2
        padded = np.pad(ay, pad, mode="edge")
        baseline = np.array([np.median(padded[i:i + win]) for i in range(len(ay))])
        lift = (baseline - ay) / med_shin  # 抬升量（小腿长倍数），向上为正
        i = 0
        while i < len(lift):
            if lift[i] > 0.12 and ank_ok[i]:
                j = i
                while j < len(lift) and lift[j] > 0.05:
                    j += 1
                dur = (j - i) / fps
                if 0.05 <= dur <= 0.45:
                    hops.append(float(times[i + int(np.argmax(lift[i:j]))]))
                i = j + int(fps * 0.5)
            else:
                i += 1

    # 挥拍候选：右腕速度峰（右手持拍）
    swings = []
    wr_ok = vis(R_WRIST)
    wr = xy(R_WRIST)
    wr_idx = np.where(wr_ok)[0]
    if len(wr_idx) > fps:
        wx = np.interp(np.arange(len(times)), wr_idx, wr[wr_idx, 0])
        wy = np.interp(np.arange(len(times)), wr_idx, wr[wr_idx, 1])
        spd = np.hypot(np.gradient(wx), np.gradient(wy)) * fps / max(np.nanmedian(shin), 1)
        th = np.nanpercentile(spd, 99) * 0.5
        i = 1
        while i < len(spd):
            if spd[i] > th and wr_ok[i]:
                peak = i + int(np.argmax(spd[i:i + int(fps)]))
                swings.append(float(times[peak]))
                i = peak + int(fps * 1.2)
            else:
                i += 1

    # 安全占位口径：膝内扣筛查——屈膝时（任一膝角<150°）双膝间距/双踝间距 < 0.70。
    # 背面机位对这个比值恰好敏感；保守占位阈值，非诊断，正式口径 C 阶前统一定。
    knee_sep = np.abs(xy(L_KNEE)[:, 0] - xy(R_KNEE)[:, 0])
    sep_ok = vis(L_KNEE) & vis(R_KNEE) & vis(L_ANKLE) & vis(R_ANKLE) & (ankle_dx > 5)
    sep_ratio = np.where(sep_ok, knee_sep / np.maximum(ankle_dx, 1e-3), np.nan)
    # 只在双脚站定时评估（排除跨步换位时双腿交叉的假阳性）：双踝水平速度 < 0.5 小腿长/秒
    med_shin_v = max(np.nanmedian(shin), 1)
    ax_l = np.gradient(np.nan_to_num(xy(L_ANKLE)[:, 0])) * fps / med_shin_v
    ax_r = np.gradient(np.nan_to_num(xy(R_ANKLE)[:, 0])) * fps / med_shin_v
    planted = (np.abs(ax_l) < 0.5) & (np.abs(ax_r) < 0.5)
    flexed = ((np.nan_to_num(knee_l, nan=180) < 150) | (np.nan_to_num(knee_r, nan=180) < 150)) & ~np.isnan(sep_ratio)
    low = flexed & planted & (sep_ratio < 0.70)
    valgus_events, last_t = [], -1e9
    for i in np.where(low)[0]:
        t = float(times[i])
        if t - last_t > 0.5:
            valgus_events.append(round(t, 2))
        last_t = t
    m["knee_valgus_events"] = valgus_events[:30]
    m["knee_valgus_note"] = "屈膝时双膝间距/双踝间距<0.70 的时刻；保守占位口径，非诊断"

    m["frames_total"] = int(len(times))
    m["frames_with_pose"] = int(np.sum(~np.isnan(pts[:, L_ANKLE, 0])))
    m["lower_body_visible_ratio"] = round(float(np.mean(vis(L_ANKLE) & vis(R_ANKLE) & vis(L_KNEE) & vis(R_KNEE))), 3)
    m["upper_body_visible_ratio"] = round(float(np.mean(vis(L_SHOULDER) & vis(R_SHOULDER))), 3)
    m["knee_angle_left_median"] = round(float(np.nanmedian(knee_l)), 1)
    m["knee_angle_right_median"] = round(float(np.nanmedian(knee_r)), 1)
    m["knee_angle_left_p10"] = round(float(np.nanpercentile(knee_l, 10)), 1)  # 最弯的 10% 时刻
    m["knee_angle_right_p10"] = round(float(np.nanpercentile(knee_r, 10)), 1)
    m["stance_width_median"] = round(float(np.nanmedian(stance)), 2)  # 单位：小腿长
    m["stance_width_p90"] = round(float(np.nanpercentile(stance, 90)), 2)
    m["hop_events"] = [round(t, 2) for t in hops]
    m["swing_candidates"] = [round(t, 2) for t in swings]
    return m, dict(knee_l=knee_l, knee_r=knee_r, stance=stance, hip_y=hip_y)


def plot(series, times, metrics, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "sans-serif"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(times, series["knee_l"], lw=0.8, label="左膝角", color="#1D9E75")
    axes[0].plot(times, series["knee_r"], lw=0.8, label="右膝角", color="#D85A30")
    axes[0].axhspan(120, 150, alpha=0.08, color="green")
    axes[0].set_ylabel("膝角 (°)\n180=伸直")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].plot(times, series["stance"], lw=0.8, color="#534AB7", label="站位宽度")
    axes[1].axhline(2.0, ls="--", lw=0.8, color="gray")
    axes[1].set_ylabel("踝间距\n(小腿长倍数)")
    axes[1].legend(loc="upper right", fontsize=8)

    hip = series["hip_y"]
    axes[2].plot(times, -hip, lw=0.8, color="#0C447C", label="重心高度(髋)")
    for t in metrics["hop_events"]:
        axes[2].axvline(t, color="#BA7517", lw=0.8, alpha=0.7)
    for t in metrics["swing_candidates"]:
        axes[2].axvline(t, color="#A32D2D", lw=0.8, alpha=0.5, ls=":")
    axes[2].set_ylabel("髋高度\n(像素,上=高)")
    axes[2].set_xlabel("时间 (s)")
    axes[2].legend(["重心高度(髋)", f"垫步/起跳 ×{len(metrics['hop_events'])}",
                    f"挥拍候选 ×{len(metrics['swing_candidates'])}"], loc="lower right", fontsize=8)

    fig.suptitle("脚步时间线", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--cut-swings", action="store_true", help="按挥拍候选切出 ±2s 叠加片段到 out/swings/")
    ap.add_argument("--det-conf", type=float, default=0.5, help="姿态检测置信度；人物在画面里较小（<30%%）时降到 0.15-0.3")
    args = ap.parse_args()

    video = Path(args.video).expanduser()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    name = video.stem + (f"_{int(args.start)}s" if args.start else "")

    print(f"处理 {video.name} ...", flush=True)
    pts, times, fps = run_pose(video, args.start, args.end, out_dir, name, args.width, args.det_conf)
    np.savez_compressed(out_dir / f"{name}_landmarks.npz", pts=pts, times=times, fps=fps)

    metrics, series = compute_metrics(pts, times, fps)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    plot(series, times, metrics, out_dir / f"{name}_footwork.png")

    if args.cut_swings and metrics["swing_candidates"]:
        swing_dir = out_dir / "swings"
        swing_dir.mkdir(exist_ok=True)
        for k, t in enumerate(metrics["swing_candidates"], 1):
            clip = swing_dir / f"{name}_swing{k:02d}_{t:.1f}s.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error",
                            "-ss", str(max(0, t - 2 - args.start)), "-to", str(t + 2 - args.start),
                            "-i", str(out_dir / f"{name}_overlay.mp4"),
                            "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-an", str(clip)],
                           check=False)
        print(f"挥拍片段: {swing_dir}/ 共 {len(metrics['swing_candidates'])} 段")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n输出: {out_dir}/{name}_overlay.mp4 / _footwork.png / _metrics.json")


if __name__ == "__main__":
    main()
