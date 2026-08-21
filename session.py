#!/usr/bin/env python3
"""一条命令跑完一次训练的全部分析，并和上一次并排成进度表。

用法：
  session.py <视频> --player me [--segments 0-180:定点,180-360:变化] [--label "8/22 背面"]
产出（sessions/<player>/<日期时间>_<视频名>/）：
  landmarks.npz / overlay.mp4 / report.json / progress.md
  card.png  关键时刻卡（C 阶）：四个关键帧 + 口令 + 达标率
  annot/    四个维度的示意视频（取「典型挥拍」）
并打印进度表：本次 vs 上一次（同 player 的最近一次）vs 目标门槛。
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from analyze import run_pose
from report import report_one, parse_segments

# 进阶门槛（原理占位口径；C 阶前统一正式化）
GOALS = {
    "contact_stance_ok_ratio": ("站位 1.3–1.8 达标率", 0.70, "≥"),
    "contact_knee_lt150_ratio": ("击球屈膝 <150° 达标率", 0.70, "≥"),
    "pre_hop_ratio": ("击球前有垫步占比", 0.50, "≥"),
    "stay_down_ratio": ("击球后留在下面占比", 0.70, "≥"),
}
AUX = [("swings_measured", "可测挥拍数"), ("contact_knee_median", "击球膝角中位(°)"),
       ("contact_stance_median", "击球站位中位"), ("pose_ratio", "有效帧占比")]


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v:.0%}" if pct else str(v)


def find_previous(player_dir, current_name):
    runs = sorted(p for p in player_dir.iterdir() if p.is_dir() and p.name != current_name and (p / "report.json").exists())
    return runs[-1] if runs else None


def typical_swing(rep):
    ps = [p for p in rep["per_swing"] if p.get("knee") is not None and p.get("stance") is not None]
    if not ps:
        return None
    km = np.median([p["knee"] for p in ps])
    sm = np.median([p["stance"] for p in ps])
    return min(ps, key=lambda p: abs(p["knee"] - km) / 10 + abs(p["stance"] - sm))["t"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--player", required=True, help="me / friend / 任意标识，用于找上一次记录")
    ap.add_argument("--segments", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--no-annot", action="store_true")
    ap.add_argument("--quick", action="store_true", help="场边快速模式：隔帧推理、不出叠加视频/卡片/示意视频，只出达标率表（约快 2.5 倍）")
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()

    video = Path(args.video).expanduser()
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"{stamp}_{video.stem}"
    player_dir = ROOT / "sessions" / args.player
    run_dir = player_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 姿态提取 {video.name} ...", flush=True)
    pts, times, fps = run_pose(video, args.start, args.end, run_dir, "overlay", args.width,
                               write_overlay=not args.quick, stride=2 if args.quick else 1)
    if args.quick and fps < 25:
        print(f"  注意：快速模式下有效帧率 {fps:.0f}fps，垫步占比会低估，只看站位/屈膝/留在下面")
    # run_pose 写的是 overlay_overlay.mp4，改个名
    ov = run_dir / "overlay_overlay.mp4"
    if ov.exists():
        ov.rename(run_dir / "overlay.mp4")
    npz = run_dir / "landmarks.npz"
    np.savez_compressed(npz, pts=pts, times=times, fps=fps)

    print("[2/5] 指标与达标率 ...", flush=True)
    segs = parse_segments(args.segments)
    rep = report_one(npz, None, segs)
    rep["video"] = str(video)
    rep["label"] = args.label
    rep["player"] = args.player
    rep["run"] = run_name
    (run_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))

    print("[3/5] 进度表 ...", flush=True)
    prev_dir = find_previous(player_dir, run_name)
    prev = json.loads((prev_dir / "report.json").read_text()) if prev_dir else None
    lines = [f"# 进度表 · {args.player} · {run_name}  {args.label}", ""]
    hdr = ["维度", "上一次" + (f"（{prev['run'][:13]}）" if prev else "（无）"), "本次", "门槛", "状态"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for key, (label, goal, op) in GOALS.items():
        cur, old = rep.get(key), (prev.get(key) if prev else None)
        ok = cur is not None and cur >= goal
        trend = ""
        if cur is not None and old is not None:
            trend = " ↑" if cur > old + 0.02 else (" ↓" if cur < old - 0.02 else " →")
        lines.append(f"| {label} | {fmt(old, True)} | {fmt(cur, True)}{trend} | {op}{goal:.0%} | {'✅' if ok else '·'} |")
    for key, label in AUX:
        cur, old = rep.get(key), (prev.get(key) if prev else None)
        lines.append(f"| {label} | {fmt(old)} | {fmt(cur)} |  |  |")
    if rep.get("by_segment"):
        lines += ["", "## 分段（同一维度在不同球下的达标率）", ""]
        segl = list(rep["by_segment"].keys())
        lines.append("| 维度 | " + " | ".join(segl) + " |")
        lines.append("|---|" + "---|" * len(segl))
        for key, (label, _, _) in GOALS.items():
            lines.append(f"| {label} | " + " | ".join(fmt(rep["by_segment"][s].get(key), True) for s in segl) + " |")
        lines.append("| 可测挥拍数 | " + " | ".join(str(rep["by_segment"][s]["n"]) for s in segl) + " |")
    md = "\n".join(lines)
    (run_dir / "progress.md").write_text(md)
    print("\n" + md + "\n")

    if args.quick:
        print(f"\n完成（快速模式）：{run_dir}")
        return

    print("[4/5] 关键时刻卡 ...", flush=True)
    cmd = [sys.executable, str(ROOT / "card.py"), str(video), str(npz), "--report", str(run_dir / "report.json"),
           "--out", str(run_dir / "card.png"), "--label", f"· {args.player} · {args.label}"]
    if prev_dir:
        cmd += ["--prev", str(prev_dir / "report.json")]
    subprocess.run(cmd, cwd=str(ROOT), check=False)

    if not args.no_annot:
        t = typical_swing(rep)
        if t is not None:
            print(f"[5/5] 示意视频（典型挥拍 t={t:.2f}s）...", flush=True)
            subprocess.run([sys.executable, str(ROOT / "annotate.py"), str(video), str(npz),
                            "--t", f"{t:.2f}", "--all", "--label", f"· {args.player} · {args.label}",
                            "--out", str(run_dir / "annot")], cwd=str(ROOT), check=False)
        else:
            print("[5/5] 无可用挥拍，跳过示意视频")
    print(f"\n完成：{run_dir}")


if __name__ == "__main__":
    main()
