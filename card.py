#!/usr/bin/env python3
"""关键时刻卡（C 阶）：一次训练 → 一张卡。

四个关键时刻（每拍自动定位）：
  ① 分腿垫步   击球前 -1.2~-0.4s 内双脚离地峰（没有则取 -0.8s 并标「缺」）
  ② 引拍完成   击球前 0.15~0.9s 内手腕相对髋中心横向外展最远处
  ③ 击球瞬间   手腕速度峰
  ④ 随挥收尾   击球后 +0.4s
每格：截图 + 火柴人 + 测量（●达标/○未达标）+ 一句外部焦点口令。
底部：四个关键时刻在本次全部挥拍上的达标率（与上次并排）。
展示的那一拍 = 本次「最接近达标」的一拍，让你看到对的样子长在自己身上是什么样。

用法：card.py <视频> <landmarks.npz> [--report report.json] [--prev 上次report.json] [--out card.png] [--label ...]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

from analyze import (CONNECTIONS, LEFT_IDS, VIS_TH, angle_deg,
                     L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, R_WRIST)
from report import series, swing_events, identity_mask

FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
GREEN, AMBER, WHITE, GRAY, DIM = (80, 220, 80), (40, 170, 255), (240, 240, 240), (170, 170, 170), (110, 110, 110)
PW, PH = 380, 500          # 每格截图尺寸
TEXT_H = 180               # 每格文字区
GAP = 18

CHECKS = [
    ("hop",     "① 分腿垫步", "对方拍子碰球那一下，双脚小跳，落地膝盖弯"),
    ("load",    "② 引拍完成", "转肩引拍时人已经沉下去，眼睛和网带一样高"),
    ("contact", "③ 击球瞬间", "在体前碰球，双脚比肩宽，腿是弯的"),
    ("finish",  "④ 随挥收尾", "头顶着一本书，一直到拍子停下来"),
]


def put(img, text, xy, size=20, color=WHITE, anchor="la"):
    pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(FONT, size)
    except OSError:
        font = ImageFont.load_default()
    d.text(xy, text, fill=(color[2], color[1], color[0]), font=font, anchor=anchor)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def foot_lift(pts, j, ank_base, shin):
    n = len(pts)
    y = pts[:, j, 1]
    ok = pts[:, j, 2] > VIS_TH
    ii = np.where(ok)[0]
    if len(ii) < 2:
        return np.full(n, np.nan)
    yi = np.interp(np.arange(n), ii, y[ii])
    ypad = np.pad(yi, 2, mode="edge")
    ym = np.array([np.median(ypad[t:t + 5]) for t in range(n)])
    return np.where(ok, (ank_base - ym) / shin, np.nan)


def keyframes(pts, times, fps, s, tc):
    """一拍的四个关键时刻：返回 {key: (frame_idx, metrics_dict, passed_bool_or_None)}"""
    n = len(times)
    ic = int(np.argmin(np.abs(times - tc)))
    shin = s["med_shin"]
    hip_h = s["hip_h"]

    def idx_at(t):
        return int(np.clip(np.argmin(np.abs(times - t)), 0, n - 1))

    def med(arr, i, w=3):
        seg = arr[max(0, i - w):i + w + 1]
        return float(np.nanmedian(seg)) if np.any(~np.isnan(seg)) else np.nan

    out = {}
    # ① 垫步
    ank_ok = (pts[:, L_ANKLE, 2] > VIS_TH) & (pts[:, R_ANKLE, 2] > VIS_TH)
    ank_y = np.where(ank_ok, (pts[:, L_ANKLE, 1] + pts[:, R_ANKLE, 1]) / 2, np.nan)
    ii = np.where(~np.isnan(ank_y))[0]
    if len(ii) > 2:
        ay = np.interp(np.arange(n), ii, ank_y[ii])
        win = int(fps * 1.5) | 1
        pad = np.pad(ay, win // 2, mode="edge")
        base = np.array([np.median(pad[i:i + win]) for i in range(n)])
        ll, lr = foot_lift(pts, L_ANKLE, base, shin), foot_lift(pts, R_ANKLE, base, shin)
        both = np.fmin(ll, lr)
        a, b = idx_at(tc - 1.2), idx_at(tc - 0.4)
        seg = both[a:b]
        if len(seg) and np.any(~np.isnan(seg)):
            j = a + int(np.nanargmax(seg))
            peak = float(np.nanmax(seg))
        else:
            j, peak = idx_at(tc - 0.8), np.nan
        hop_ok = (not np.isnan(peak)) and peak > 0.10
        land_st = med(s["stance"], min(j + int(fps * 0.15), n - 1))
        out["hop"] = (j if hop_ok else idx_at(tc - 0.8),
                      {"双脚离地": (f"{peak:.2f}" if not np.isnan(peak) else "-", hop_ok),
                       "落地站位": (f"{land_st:.2f}" if not np.isnan(land_st) else "-", (not np.isnan(land_st)) and 1.3 <= land_st <= 1.8)},
                      hop_ok)
    else:
        out["hop"] = (idx_at(tc - 0.8), {"双脚离地": ("-", False)}, None)

    # ② 引拍完成：手腕相对髋中心横向外展最远（右手：x 更大）
    a, b = idx_at(tc - 0.9), idx_at(tc - 0.15)
    hip_cx = np.where((pts[:, L_HIP, 2] > VIS_TH) & (pts[:, R_HIP, 2] > VIS_TH), (pts[:, L_HIP, 0] + pts[:, R_HIP, 0]) / 2, np.nan)
    wr_ok = pts[:, R_WRIST, 2] > VIS_TH
    off = np.where(wr_ok, pts[:, R_WRIST, 0] - hip_cx, np.nan)
    seg = off[a:b]
    if len(seg) and np.any(~np.isnan(seg)):
        j = a + int(np.nanargmax(seg))
    else:
        j = idx_at(tc - 0.4)
    knee = med(s["knee_min"], j)
    st = med(s["stance"], j)
    # 转肩粗估：肩线视宽 / 髋线视宽（背面机位，越小=转得越多）；<0.75 视为已转肩
    sh_ok = all(pts[j, k, 2] > VIS_TH for k in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP))
    turn = np.nan
    if sh_ok:
        sw = abs(pts[j, L_SHOULDER, 0] - pts[j, R_SHOULDER, 0])
        hw = abs(pts[j, L_HIP, 0] - pts[j, R_HIP, 0])
        turn = sw / hw if hw > 1 else np.nan
    m = {"膝角": (f"{knee:.0f}°" if not np.isnan(knee) else "-", (not np.isnan(knee)) and knee < 155),
         "站位": (f"{st:.2f}" if not np.isnan(st) else "-", (not np.isnan(st)) and 1.3 <= st <= 1.8)}
    # 转肩粗估在背面机位不可靠（转体后肩线视宽反而可能变大），暂不展示；侧面机位到位后再做正式指标
    _ = turn
    out["load"] = (j, m, m["膝角"][1])

    # ③ 击球
    knee = med(s["knee_min"], ic)
    st = med(s["stance"], ic)
    out["contact"] = (ic, {"膝角": (f"{knee:.0f}°" if not np.isnan(knee) else "-", (not np.isnan(knee)) and knee < 150),
                           "站位": (f"{st:.2f}" if not np.isnan(st) else "-", (not np.isnan(st)) and 1.3 <= st <= 1.8)},
                      (not np.isnan(knee)) and knee < 150)

    # ④ 随挥收尾：+0.4s，髋相对击球时上浮量
    jf = idx_at(tc + 0.4)
    h0, h1 = med(hip_h, ic), med(hip_h, jf)
    rise = (h1 - h0) if not (np.isnan(h0) or np.isnan(h1)) else np.nan
    st = med(s["stance"], jf)
    out["finish"] = (jf, {"髋上浮": (f"{rise:+.2f}" if not np.isnan(rise) else "-", (not np.isnan(rise)) and rise < 0.08),
                          "站位": (f"{st:.2f}" if not np.isnan(st) else "-", (not np.isnan(st)) and st >= 1.0)},
                     (not np.isnan(rise)) and rise < 0.08)
    return out


def render_panel(frame, p, cx, cy, shin):
    bw, bh = 5.0 * shin, 7.0 * shin
    x0, y0 = cx - bw / 2, cy - bh * 0.42
    H, W = frame.shape[:2]
    pad_l, pad_t = max(0, -int(x0)), max(0, -int(y0))
    pad_r, pad_b = max(0, int(x0 + bw) - W), max(0, int(y0 + bh) - H)
    img = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    crop = img[int(y0) + pad_t:int(y0 + bh) + pad_t, int(x0) + pad_l:int(x0 + bw) + pad_l]
    canvas = cv2.resize(crop, (PW, PH))
    sx, sy = PW / bw, PH / bh
    drw = {j: (int((p[j, 0] - x0) * sx), int((p[j, 1] - y0) * sy)) for j in range(33) if p[j, 2] > VIS_TH}
    for a, b in CONNECTIONS:
        if a in drw and b in drw:
            cv2.line(canvas, drw[a], drw[b], GREEN, 2, cv2.LINE_AA)
    for j, q in drw.items():
        cv2.circle(canvas, q, 4, (60, 120, 255) if j in LEFT_IDS else (255, 160, 40), -1, cv2.LINE_AA)
    return canvas


def score_swing(kf):
    return sum(1 for k, (_, _, ok) in kf.items() if ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("npz")
    ap.add_argument("--report", default=None, help="report.json（含 per_swing 与达标率）")
    ap.add_argument("--prev", default=None, help="上一次 report.json，用于箭头")
    ap.add_argument("--out", default="out/card.png")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    d = np.load(args.npz)
    pts, times, fps = d["pts"], d["times"], float(d["fps"])
    pts, _ = identity_mask(pts)
    s = series(pts, times, fps)
    def load_rep(pth):
        if not pth:
            return None
        obj = json.loads(Path(pth).read_text())
        return obj[0] if isinstance(obj, list) else obj  # report.py 多文件输出是列表
    rep, prev = load_rep(args.report), load_rep(args.prev)
    swings = [p["t"] for p in rep["per_swing"] if p.get("knee") is not None] if rep else swing_events(pts, times, fps, s["med_shin"])
    if not swings:
        raise SystemExit("没有可用挥拍")

    # 每拍关键帧 → 选最接近达标的一拍（并列取更晚的：更接近训练末状态）
    kfs = [(t, keyframes(pts, times, fps, s, t)) for t in swings]
    rates = {k: np.mean([1 if kf[k][2] else 0 for _, kf in kfs if kf[k][2] is not None]) if any(kf[k][2] is not None for _, kf in kfs) else np.nan
             for k, _, _ in CHECKS}
    best_t, best_kf = max(kfs, key=lambda x: (score_swing(x[1]), x[0]))

    # 取帧
    cap = cv2.VideoCapture(args.video)
    frames = {}
    for k, (fi, _, _) in best_kf.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        frames[k] = f if ok else np.full((int(cap.get(4)), int(cap.get(3)), 3), 24, np.uint8)
    cap.release()

    # 固定裁剪框：四帧共用（该拍窗口内髋中点中位）
    i0, i1 = int(np.argmin(np.abs(times - (best_t - 1.2)))), int(np.argmin(np.abs(times - (best_t + 0.6))))
    hip = np.where(((pts[:, L_HIP, 2] > VIS_TH) & (pts[:, R_HIP, 2] > VIS_TH))[:, None], (pts[:, L_HIP, :2] + pts[:, R_HIP, :2]) / 2, np.nan)
    cxy = np.nanmedian(hip[i0:i1], axis=0)
    # 裁剪尺度用该拍窗口内的局部小腿长（人远近不同，全段中位会让远处的人显得很小）
    ok4 = np.all(pts[i0:i1][:, [L_KNEE, R_KNEE, L_ANKLE, R_ANKLE], 2] > VIS_TH, axis=1)
    loc = (np.linalg.norm(pts[i0:i1, L_KNEE, :2] - pts[i0:i1, L_ANKLE, :2], axis=1)
           + np.linalg.norm(pts[i0:i1, R_KNEE, :2] - pts[i0:i1, R_ANKLE, :2], axis=1)) / 2
    shin = float(np.nanmedian(np.where(ok4, loc, np.nan))) if ok4.any() else s["med_shin"]

    # 画布
    W = GAP + 4 * (PW + GAP)
    HEAD, FOOT = 90, 150
    H = HEAD + PH + TEXT_H + FOOT
    card = np.full((H, W, 3), 24, np.uint8)
    title = f"关键时刻卡  {args.label}".strip()
    card = put(card, title, (GAP, 18), 30, WHITE)
    card = put(card, f"展示的是本次最接近达标的一拍（t={best_t:.1f}s，{score_swing(best_kf)}/4 项达标）· 练之前看一眼，带着这四个画面上场", (GAP, 58), 18, GRAY)

    for col, (key, name, cue) in enumerate(CHECKS):
        x = GAP + col * (PW + GAP)
        fi, metrics, ok = best_kf[key]
        panel = render_panel(frames[key], pts[fi], cxy[0], cxy[1], shin)
        rel = times[fi] - best_t
        panel = put(panel, f"t={rel:+.2f}s", (PW - 10, 8), 16, GRAY, "ra")
        card[HEAD:HEAD + PH, x:x + PW] = panel
        # 边框颜色 = 该项是否达标
        col_bd = GREEN if ok else (AMBER if ok is not None else DIM)
        cv2.rectangle(card, (x - 2, HEAD - 2), (x + PW + 1, HEAD + PH + 1), col_bd, 2)
        # 文字区
        y = HEAD + PH + 10
        card = put(card, name, (x, y), 24, WHITE)
        y += 34
        for mk, (mv, mok) in metrics.items():
            card = put(card, f"{'●' if mok else '○'} {mk} {mv}", (x, y), 18, GREEN if mok else AMBER)
            y += 24
        # 口令（自动折行）
        maxc = 17
        chunks = [cue[c:c + maxc] for c in range(0, len(cue), maxc)]
        yy = HEAD + PH + TEXT_H - 22 * len(chunks) - 4
        for ch in chunks:
            card = put(card, ch, (x, yy), 17, WHITE)
            yy += 22

    # 底部：达标率
    y0 = HEAD + PH + TEXT_H + 12
    cv2.line(card, (GAP, y0 - 6), (W - GAP, y0 - 6), DIM, 1)
    card = put(card, "本次全部挥拍的达标率" + (f"（共 {len(kfs)} 拍）" if kfs else ""), (GAP, y0), 18, GRAY)
    prev_rates = None
    if prev:
        prev_rates = {"hop": prev.get("pre_hop_ratio"), "load": None,
                      "contact": prev.get("contact_knee_lt150_ratio"), "finish": prev.get("stay_down_ratio")}
    for col, (key, name, _) in enumerate(CHECKS):
        x = GAP + col * (PW + GAP)
        r = rates.get(key, np.nan)
        txt = f"{name[2:]}  {r:.0%}" if not np.isnan(r) else f"{name[2:]}  -"
        arrow = ""
        if prev_rates and prev_rates.get(key) is not None and not np.isnan(r):
            pr = prev_rates[key]
            arrow = "  ↑" if r > pr + 0.02 else ("  ↓" if r < pr - 0.02 else "  →")
            txt += f"{arrow}（上次 {pr:.0%}）"
        good = (not np.isnan(r)) and r >= (0.5 if key == "hop" else 0.7)
        card = put(card, txt, (x, y0 + 30), 22, GREEN if good else WHITE)
    card = put(card, "门槛：垫步 ≥50%，其余 ≥70% · 阈值为原理占位口径，看趋势", (GAP, y0 + 70), 15, DIM)
    card = put(card, "上半身指标（转肩/击球点/躯干）待侧面机位；本卡为背面机位版", (GAP, y0 + 92), 15, DIM)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), card)
    summary = {"best_swing_t": best_t, "best_score": score_swing(best_kf),
               "rates": {k: (None if np.isnan(v) else round(float(v), 2)) for k, v in rates.items()},
               "swings": len(kfs)}
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))
    print(f"卡片: {out}")


if __name__ == "__main__":
    main()
