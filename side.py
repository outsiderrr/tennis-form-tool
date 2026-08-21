#!/usr/bin/env python3
"""侧面机位报告（v0）：只算侧面视角才可靠的三件事。

  1. 击球瞬间膝角 —— 侧面是测屈膝最准的视角（背面会低估）
  2. 击球点在体前的距离 —— 手腕相对髋中心沿击球方向的水平距离（小腿长归一）
  3. 弯腰还是弯膝（安全项）—— 躯干前倾角 vs 屈膝量；前倾>25° 且膝角>160° = 弯腰代偿
  附: 引拍深度 —— 引拍窗口内手腕到髋后方的最大水平距离

不算站位宽度/垫步（侧面视角下这些量的几何意义变了，见 CONTEXT.md 口径）。

用法：side.py <视频> <landmarks.npz> [--out-prefix out/side_IMG_3339]
输出：<prefix>_report.json + <prefix>_contact.png（典型挥拍击球帧标注图）
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from analyze import (CONNECTIONS, LEFT_IDS, VIS_TH, angle_deg,
                     L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE,
                     L_ANKLE, R_ANKLE, L_HEEL, R_HEEL, L_FOOT, R_FOOT, R_WRIST)
from report import series, swing_events, identity_mask
from card import put, render_panel

GREEN, AMBER, WHITE, GRAY = (80, 220, 80), (40, 170, 255), (240, 240, 240), (170, 170, 170)


def side_series(pts, times, fps):
    n = len(times)
    vis = lambda j: pts[:, j, 2] > VIS_TH
    x = lambda j: pts[:, j, 0]
    y = lambda j: pts[:, j, 1]

    hip_ok = vis(L_HIP) & vis(R_HIP)
    sh_ok = vis(L_SHOULDER) & vis(R_SHOULDER)
    hip_cx = np.where(hip_ok, (x(L_HIP) + x(R_HIP)) / 2, np.nan)
    hip_cy = np.where(hip_ok, (y(L_HIP) + y(R_HIP)) / 2, np.nan)
    sh_cx = np.where(sh_ok, (x(L_SHOULDER) + x(R_SHOULDER)) / 2, np.nan)
    sh_cy = np.where(sh_ok, (y(L_SHOULDER) + y(R_SHOULDER)) / 2, np.nan)

    # 躯干前倾角：髋中点→肩中点连线与竖直方向的夹角（度，0=直立）
    lean = np.degrees(np.arctan2(np.abs(sh_cx - hip_cx), np.maximum(hip_cy - sh_cy, 1e-6)))
    lean = np.where(hip_ok & sh_ok, lean, np.nan)

    # 朝向符号：脚尖相对脚跟的 x 方向（两脚平均），+1=画面右为「前」
    fdir = np.zeros(n)
    cnt = np.zeros(n)
    for heel, toe in ((L_HEEL, L_FOOT), (R_HEEL, R_FOOT)):
        ok = vis(heel) & vis(toe)
        fdir += np.where(ok, np.sign(x(toe) - x(heel)), 0)
        cnt += ok
    facing = np.where(cnt > 0, np.sign(fdir), np.nan)

    # 手腕相对髋中心的带符号前向距离（像素，之后除以小腿长）
    wr_ok = vis(R_WRIST)
    wrist_fwd = np.where(wr_ok & hip_ok, (x(R_WRIST) - hip_cx) * facing, np.nan)
    return dict(lean=lean, facing=facing, wrist_fwd=wrist_fwd, hip_cx=hip_cx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("npz")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    prefix = Path(args.out_prefix or ("out/side_" + Path(args.npz).stem.replace("_landmarks", "")))

    d = np.load(args.npz)
    pts, times, fps = d["pts"], d["times"], float(d["fps"])
    pts, _ = identity_mask(pts)
    s = series(pts, times, fps)
    ss = side_series(pts, times, fps)
    swings = swing_events(pts, times, fps, s["med_shin"])
    shin = s["med_shin"]
    n = len(times)

    def med_at(arr, t, w=0.06):
        m = (times >= t - w) & (times <= t + w)
        v = arr[m]
        return float(np.nanmedian(v)) if np.any(~np.isnan(v)) else np.nan

    per = []
    for t in swings:
        knee = med_at(s["knee_min"], t)
        fwd = med_at(ss["wrist_fwd"], t) / shin if shin > 1 else np.nan
        lean = med_at(ss["lean"], t)
        # 引拍深度：窗口内手腕在髋后方的最大距离
        a = int(np.clip(np.searchsorted(times, t - 0.9), 0, n - 1))
        b = int(np.clip(np.searchsorted(times, t - 0.15), 0, n - 1))
        seg = ss["wrist_fwd"][a:b]
        back = (-np.nanmin(seg) / shin) if len(seg) and np.any(~np.isnan(seg)) and shin > 1 else np.nan
        bend_over = (not np.isnan(lean)) and (not np.isnan(knee)) and lean > 25 and knee > 160
        per.append({"t": round(float(t), 2),
                    "knee": None if np.isnan(knee) else round(knee, 1),
                    "contact_fwd": None if np.isnan(fwd) else round(fwd, 2),
                    "lean": None if np.isnan(lean) else round(lean, 1),
                    "takeback": None if np.isnan(back) else round(back, 2),
                    "bend_over": bool(bend_over)})

    valid = [p for p in per if p["knee"] is not None]
    rep = {"file": Path(args.npz).name, "view": "side", "swings_measured": len(valid), "label": args.label}
    if valid:
        ks = [p["knee"] for p in valid]
        rep["contact_knee_median"] = round(float(np.median(ks)), 1)
        rep["contact_knee_lt150_ratio"] = round(float(np.mean([k < 150 for k in ks])), 2)
        fw = [p["contact_fwd"] for p in valid if p["contact_fwd"] is not None]
        rep["contact_fwd_median"] = round(float(np.median(fw)), 2) if fw else None
        rep["contact_in_front_ratio"] = round(float(np.mean([f > 0.2 for f in fw])), 2) if fw else None
        ln = [p["lean"] for p in valid if p["lean"] is not None]
        rep["lean_median"] = round(float(np.median(ln)), 1) if ln else None
        rep["bend_over_ratio"] = round(float(np.mean([p["bend_over"] for p in valid])), 2)
        tb = [p["takeback"] for p in valid if p["takeback"] is not None]
        rep["takeback_median"] = round(float(np.median(tb)), 2) if tb else None
    rep["per_swing"] = per
    rep["notes"] = ["侧面口径：膝角最准；击球点>0.2 小腿长=在体前；前倾>25°且膝角>160°=弯腰代偿",
                    "不算站位/垫步（侧面几何意义不同）；与背面数据不可直接比膝角绝对值"]
    Path(str(prefix) + "_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))

    # 典型挥拍击球帧标注图
    if valid:
        km = np.median([p["knee"] for p in valid])
        best = min(valid, key=lambda p: abs(p["knee"] - km))
        ic = int(np.argmin(np.abs(times - best["t"])))
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, ic)
        ok, frame = cap.read()
        cap.release()
        if ok:
            hips = np.where(((pts[:, L_HIP, 2] > VIS_TH) & (pts[:, R_HIP, 2] > VIS_TH))[:, None],
                            (pts[:, L_HIP, :2] + pts[:, R_HIP, :2]) / 2, np.nan)
            i0, i1 = max(0, ic - int(fps)), min(n, ic + int(fps))
            cxy = np.nanmedian(hips[i0:i1], axis=0)
            panel = render_panel(frame, pts[ic], cxy[0], cxy[1], shin)
            H, W = panel.shape[:2]
            sx = W / (5.0 * shin)
            # 竖直参考线（过髋）+ 髋→腕水平箭头
            px_hip = int(W * 0.5)
            for yy in range(40, H - 20, 18):
                cv2.line(panel, (px_hip, yy), (px_hip, yy + 9), GRAY, 1, cv2.LINE_AA)
            if best["contact_fwd"] is not None and not np.isnan(ss["facing"][ic]):
                dx = int(best["contact_fwd"] * shin * sx * ss["facing"][ic])
                yw = int(H * 0.45)
                cv2.arrowedLine(panel, (px_hip, yw), (px_hip + dx, yw), AMBER := (40, 170, 255), 3, cv2.LINE_AA, tipLength=0.12)
                panel = put(panel, f"击球点 {best['contact_fwd']:+.2f} 小腿长", (min(px_hip, px_hip + dx), yw - 30), 20,
                            GREEN if best["contact_fwd"] > 0.2 else AMBER)
            panel = put(panel, f"膝角 {best['knee']:.0f}°  前倾 {best['lean']:.0f}°", (14, 12), 22,
                        GREEN if best["knee"] < 150 else AMBER)
            bar = np.full((70, W, 3), 24, np.uint8)
            bar = put(bar, f"侧面 · 典型挥拍 t={best['t']:.1f}s  {args.label}", (14, 8), 20, WHITE)
            bar = put(bar, "灰虚线=髋的竖直参考 · 箭头=击球点在体前的距离", (14, 38), 16, GRAY)
            cv2.imwrite(str(prefix) + "_contact.png", np.vstack([panel, bar]))

    print(json.dumps({k: v for k, v in rep.items() if k != "per_swing"}, ensure_ascii=False, indent=2))
    print(f"输出: {prefix}_report.json / _contact.png")


if __name__ == "__main__":
    main()
