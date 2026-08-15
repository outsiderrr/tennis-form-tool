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


def angle_deg(a, b, c):
    """b 处的夹角（度），a/b/c 为 (x, y)"""
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return np.nan
    cosv = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def run_pose(video, start, end, out_dir, name, overlay_width):
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
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

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
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            for j, p in enumerate(lm):
                pts[i, j] = (p.x * w, p.y * h, p.visibility)
            draw = {j: (int(pts[i, j, 0] * scale), int(pts[i, j, 1] * scale))
                    for j in range(33) if pts[i, j, 2] > VIS_TH and 0 <= pts[i, j, 1] < h}
            for a, b in CONNECTIONS:
                if a in draw and b in draw:
                    cv2.line(small, draw[a], draw[b], (80, 220, 80), 2, cv2.LINE_AA)
            for j, p in draw.items():
                cv2.circle(small, p, 4, (60, 120, 255) if j in LEFT_IDS else (255, 160, 40), -1, cv2.LINE_AA)
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

    # 分腿垫步候选：髋中点的短促上-下振荡（速度过零 + 幅度阈值）
    hops = []
    hy = hip_y.copy()
    idx = np.where(~np.isnan(hy))[0]
    if len(idx) > fps:  # 至少 1 秒有效数据
        hy_i = np.interp(np.arange(len(hy)), idx, hy[idx])
        k = max(3, int(fps * 0.08))
        kernel = np.ones(k) / k
        smooth = np.convolve(hy_i, kernel, mode="same")
        vel = np.gradient(smooth) * fps  # px/s，向下为正
        med_shin = np.nanmedian(shin)
        up_th = -0.6 * med_shin  # 上升速度阈值：0.6 小腿长/秒
        i = 1
        while i < len(vel) - 1:
            if vel[i] < up_th:  # 快速上升开始
                j = i
                while j < len(vel) - 1 and vel[j] < 0:
                    j += 1
                rise = smooth[i] - smooth[min(j, len(smooth) - 1)]
                if rise > 0.10 * med_shin:  # 髋上抬超过 0.1 小腿长 → 记一次垫步/起跳
                    hops.append(float(times[i]))
                    i = j + int(fps * 0.4)  # 0.4s 内不重复计
                    continue
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
    args = ap.parse_args()

    video = Path(args.video).expanduser()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    name = video.stem + (f"_{int(args.start)}s" if args.start else "")

    print(f"处理 {video.name} ...", flush=True)
    pts, times, fps = run_pose(video, args.start, args.end, out_dir, name, args.width)
    np.savez_compressed(out_dir / f"{name}_landmarks.npz", pts=pts, times=times, fps=fps)

    metrics, series = compute_metrics(pts, times, fps)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    plot(series, times, metrics, out_dir / f"{name}_footwork.png")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n输出: {out_dir}/{name}_overlay.mp4 / _footwork.png / _metrics.json")


if __name__ == "__main__":
    main()
