#!/usr/bin/env python3
"""脚步体检报告：从 analyze.py 存下的关键点（npz）计算全套背面机位指标。

用法：report.py out/IMG_3266_landmarks.npz [out/IMG_0010_landmarks.npz ...] [--swings 39.5,50.9]
多个 npz 一起给时输出对照表。

指标（全部按小腿长归一化、视角无关或背面机位可靠）：
  准备/常态：膝角中位、膝角 p10、站位宽度中位/p90、重心高度稳定性
  击球瞬间：膝角、站位、重心相对常态的下沉量、屈膝达标率(<150°)
  动态：垫步次数/分钟、击球前 0.4s 内有无垫步、脚踝横向移动速度、
        击球后 0.5s 重心是否上浮（=站起来了，动力链没往前走）
  安全：膝内扣时刻数、躯干侧倾（肩线-髋线夹角，仅在上半身可见时）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import (VIS_TH, angle_deg, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
                     L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, R_WRIST)


def series(pts, times, fps):
    n = len(times)

    def vis(j):
        return pts[:, j, 2] > VIS_TH

    def xy(j):
        return pts[:, j, :2]

    def ang(h, k, a):
        out = np.full(n, np.nan)
        ok = vis(h) & vis(k) & vis(a)
        for i in np.where(ok)[0]:
            out[i] = angle_deg(xy(h)[i], xy(k)[i], xy(a)[i])
        return out

    s = {}
    s["knee_l"], s["knee_r"] = ang(L_HIP, L_KNEE, L_ANKLE), ang(R_HIP, R_KNEE, R_ANKLE)
    s["knee_min"] = np.fmin(s["knee_l"], s["knee_r"])
    shin = (np.linalg.norm(xy(L_KNEE) - xy(L_ANKLE), axis=1)
            + np.linalg.norm(xy(R_KNEE) - xy(R_ANKLE), axis=1)) / 2
    med_shin = float(np.nanmedian(np.where(vis(L_KNEE) & vis(L_ANKLE) & vis(R_KNEE) & vis(R_ANKLE), shin, np.nan)))
    s["med_shin"] = med_shin
    ok4 = vis(L_ANKLE) & vis(R_ANKLE) & vis(L_KNEE) & vis(R_KNEE)
    s["stance"] = np.where(ok4 & (shin > 1), np.abs(xy(L_ANKLE)[:, 0] - xy(R_ANKLE)[:, 0]) / med_shin, np.nan)

    hip_ok = vis(L_HIP) & vis(R_HIP)
    hip = np.where(hip_ok[:, None], (xy(L_HIP) + xy(R_HIP)) / 2, np.nan)
    ank_ok = vis(L_ANKLE) & vis(R_ANKLE)
    ank = np.where(ank_ok[:, None], (xy(L_ANKLE) + xy(R_ANKLE)) / 2, np.nan)
    # 重心高度：髋中点到踝中点的垂直距离 / 小腿长（透视稳健、越小越低）
    s["hip_h"] = np.where(hip_ok & ank_ok, (ank[:, 1] - hip[:, 1]) / med_shin, np.nan)

    # 踝部横向速度（小腿长/秒）：脚步活跃度
    def spd(j):
        x = xy(j)[:, 0].copy()
        idx = np.where(vis(j))[0]
        if len(idx) < 2:
            return np.full(n, np.nan)
        xi = np.interp(np.arange(n), idx, x[idx])
        k = max(3, int(fps * 0.1)) | 1
        xi = np.convolve(np.pad(xi, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:n]
        v = np.abs(np.gradient(xi)) * fps / med_shin
        return np.where(vis(j), v, np.nan)
    s["ank_spd"] = np.fmax(spd(L_ANKLE), spd(R_ANKLE))

    # 躯干侧倾：肩线与髋线夹角（度），背面视角可见时
    sh_ok = vis(L_SHOULDER) & vis(R_SHOULDER) & hip_ok
    sh_vec = xy(R_SHOULDER) - xy(L_SHOULDER)
    hp_vec = xy(R_HIP) - xy(L_HIP)
    a1 = np.degrees(np.arctan2(sh_vec[:, 1], sh_vec[:, 0]))
    a2 = np.degrees(np.arctan2(hp_vec[:, 1], hp_vec[:, 0]))
    tilt = np.abs(((a1 - a2) + 180) % 360 - 180)
    s["trunk_tilt"] = np.where(sh_ok, tilt, np.nan)

    # 垫步（同 analyze.py v1 口径）
    hops = []
    ank_y = np.where(ank_ok, ank[:, 1], np.nan)
    idx = np.where(~np.isnan(ank_y))[0]
    if len(idx) > fps and med_shin > 1:
        ay = np.interp(np.arange(n), idx, ank_y[idx])
        win = int(fps * 1.5) | 1
        pad = win // 2
        padded = np.pad(ay, pad, mode="edge")
        base = np.array([np.median(padded[i:i + win]) for i in range(n)])
        lift = (base - ay) / med_shin
        i = 0
        while i < n:
            if lift[i] > 0.12 and ank_ok[i]:
                j = i
                while j < n and lift[j] > 0.05:
                    j += 1
                if 0.05 <= (j - i) / fps <= 0.45:
                    hops.append(float(times[i + int(np.argmax(lift[i:j]))]))
                i = j + int(fps * 0.5)
            else:
                i += 1
    s["hops"] = hops
    return s


def swing_events(pts, times, fps, med_shin, return_rejected=False):
    """右腕速度峰 → 挥拍候选，再用视觉规则剔除假挥拍（捡球/弯腰/走动甩手）。

    剔除条件（击球时刻 ±0.1s 内取中位）：
      1. 身份：人物像素身高 < 0.5×全段中位 → 锁到了远处别人
      2. 深蹲：髋高于膝的高度/大腿长 < 0.55（尺度无关）→ 捡球
      3. 手腕低于膝盖且双脚并拢 → 捡球
    音频击球声方案已测试：室内回声+他人击球+球落地声干扰，不可靠，弃用。
    """
    n = len(times)
    ok = pts[:, R_WRIST, 2] > VIS_TH
    idx = np.where(ok)[0]
    if len(idx) < fps:
        return ([], []) if return_rejected else []
    wx = np.interp(np.arange(n), idx, pts[idx, R_WRIST, 0])
    wy = np.interp(np.arange(n), idx, pts[idx, R_WRIST, 1])
    spd = np.hypot(np.gradient(wx), np.gradient(wy)) * fps / max(med_shin, 1)
    th = np.nanpercentile(spd, 99) * 0.5

    knee_y = np.where((pts[:, L_KNEE, 2] > VIS_TH) & (pts[:, R_KNEE, 2] > VIS_TH),
                      (pts[:, L_KNEE, 1] + pts[:, R_KNEE, 1]) / 2, np.nan)
    hip_ok = (pts[:, L_HIP, 2] > VIS_TH) & (pts[:, R_HIP, 2] > VIS_TH)
    ank_ok = (pts[:, L_ANKLE, 2] > VIS_TH) & (pts[:, R_ANKLE, 2] > VIS_TH)
    hip_y = np.where(hip_ok, (pts[:, L_HIP, 1] + pts[:, R_HIP, 1]) / 2, np.nan)
    ank_y = np.where(ank_ok, (pts[:, L_ANKLE, 1] + pts[:, R_ANKLE, 1]) / 2, np.nan)
    # 逐帧小腿长做尺度归一（人远近变化时像素距离会变）
    shin_f = (np.linalg.norm(pts[:, L_KNEE, :2] - pts[:, L_ANKLE, :2], axis=1)
              + np.linalg.norm(pts[:, R_KNEE, :2] - pts[:, R_ANKLE, :2], axis=1)) / 2
    shin_f = np.where(ank_ok & (pts[:, L_KNEE, 2] > VIS_TH) & (pts[:, R_KNEE, 2] > VIS_TH) & (shin_f > 1), shin_f, np.nan)
    stance_f = np.where(ank_ok, np.abs(pts[:, L_ANKLE, 0] - pts[:, R_ANKLE, 0]) / shin_f, np.nan)
    # 深蹲判据（尺度无关）：髋在膝上方的高度 / 大腿长。站立≈1.0，深蹲≤0.5
    sq = []
    for h, k in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
        thigh = np.linalg.norm(pts[:, h, :2] - pts[:, k, :2], axis=1)
        sq.append(np.where((pts[:, h, 2] > VIS_TH) & (pts[:, k, 2] > VIS_TH) & (thigh > 1),
                           (pts[:, k, 1] - pts[:, h, 1]) / thigh, np.nan))
    with np.errstate(all="ignore"):
        squat = np.nanmax(np.vstack(sq), axis=0)
    # 身份判据：目标人物像素身高（髋-踝）远小于全段中位 → 锁到了别人
    body_px = ank_y - hip_y
    body_med = np.nanmedian(body_px)

    win = max(1, int(fps * 0.1))
    ev, rejected, i = [], [], 1
    while i < n:
        if spd[i] > th and ok[i]:
            peak = i + int(np.argmax(spd[i:i + int(fps)]))
            a, b = max(0, peak - win), min(n, peak + win + 1)
            def med(arr):
                seg = arr[a:b]
                return float(np.nanmedian(seg)) if np.any(~np.isnan(seg)) else np.nan
            w_y, k_y, st, sqv, bpx = med(wy), med(knee_y), med(stance_f), med(squat), med(body_px)
            reason = None
            if not np.isnan(bpx) and not np.isnan(body_med) and bpx < 0.65 * body_med:
                reason = "疑似锁到别人（人物过小）"
            elif not np.isnan(sqv) and sqv < 0.55:
                reason = "深蹲（捡球）"
            elif (not np.isnan(k_y)) and w_y > k_y and (not np.isnan(st)) and st < 0.6:
                reason = "手腕低于膝盖且双脚并拢（捡球）"
            (rejected if reason else ev).append((float(times[peak]), reason) if reason else float(times[peak]))
            i = peak + int(fps * 1.2)
        else:
            i += 1
    return (ev, rejected) if return_rejected else ev


def at(series_arr, times, t, win=0.05):
    m = (times >= t - win) & (times <= t + win)
    v = series_arr[m]
    return float(np.nanmedian(v)) if np.any(~np.isnan(v)) else np.nan


def identity_mask(pts):
    """把「锁到别人」的帧整体置 NaN：人物像素身高（髋-踝）< 0.65×全段中位"""
    hip_ok = (pts[:, L_HIP, 2] > VIS_TH) & (pts[:, R_HIP, 2] > VIS_TH)
    ank_ok = (pts[:, L_ANKLE, 2] > VIS_TH) & (pts[:, R_ANKLE, 2] > VIS_TH)
    body = np.where(hip_ok & ank_ok,
                    (pts[:, L_ANKLE, 1] + pts[:, R_ANKLE, 1]) / 2 - (pts[:, L_HIP, 1] + pts[:, R_HIP, 1]) / 2, np.nan)
    med = np.nanmedian(body)
    bad = ~np.isnan(body) & (body < 0.65 * med)
    out = pts.copy()
    out[bad] = np.nan
    return out, int(bad.sum())


def parse_segments(spec):
    """'0-180:定点,180-360:变化' → [(0,180,'定点'),(180,360,'变化')]；None → 全段一个标签"""
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        rng, label = part.split(":")
        a, b = rng.split("-")
        out.append((float(a), float(b), label.strip()))
    return out


def contact_stats(valid):
    """一组挥拍的击球瞬间统计（达标率）"""
    o = {"n": len(valid)}
    if not valid:
        return o
    ks = [p["knee"] for p in valid]
    o["contact_knee_median"] = round(float(np.median(ks)), 1)
    o["contact_knee_lt150_ratio"] = round(float(np.mean([k < 150 for k in ks])), 2)
    sts = [p["stance"] for p in valid if p["stance"] is not None]
    o["contact_stance_median"] = round(float(np.median(sts)), 2) if sts else None
    o["contact_stance_ok_ratio"] = round(float(np.mean([1.3 <= x <= 1.8 for x in sts])), 2) if sts else None
    sinks = [p["sink"] for p in valid if p["sink"] is not None]
    o["contact_sink_median"] = round(float(np.median(sinks)), 2) if sinks else None
    rises = [p["rise_after"] for p in valid if p["rise_after"] is not None]
    o["post_contact_rise_ratio"] = round(float(np.mean([x > 0.08 for x in rises])), 2) if rises else None
    o["stay_down_ratio"] = round(1 - o["post_contact_rise_ratio"], 2) if rises else None
    o["pre_hop_ratio"] = round(float(np.mean([p["pre_hop"] for p in valid])), 2)
    return o


def report_one(npz_path, swings=None, segments=None):
    d = np.load(npz_path)
    pts, times, fps = d["pts"], d["times"], float(d["fps"])
    pts, n_bad = identity_mask(pts)
    s = series(pts, times, fps)
    if swings is None:
        swings = swing_events(pts, times, fps, s["med_shin"])
    dur_min = (times[-1] - times[0]) / 60 if len(times) > 1 else 1

    def seg_label(t):
        if not segments:
            return "全段"
        for a, b, lab in segments:
            if a <= t < b:
                return lab
        return "未标注"

    r = {"file": Path(npz_path).name, "duration_min": round(dur_min, 2),
         "pose_ratio": round(float(np.mean(~np.isnan(pts[:, L_HIP, 0]))), 2),
         "frames_masked_other_person": n_bad}
    # 常态
    r["knee_median"] = round(float(np.nanmedian(s["knee_min"])), 1)
    r["knee_p10"] = round(float(np.nanpercentile(s["knee_min"], 10)), 1)
    r["stance_median"] = round(float(np.nanmedian(s["stance"])), 2)
    r["stance_p90"] = round(float(np.nanpercentile(s["stance"], 90)), 2)
    r["hip_h_median"] = round(float(np.nanmedian(s["hip_h"])), 2)
    r["trunk_tilt_median"] = round(float(np.nanmedian(s["trunk_tilt"])), 1) if np.any(~np.isnan(s["trunk_tilt"])) else None
    r["trunk_tilt_p90"] = round(float(np.nanpercentile(s["trunk_tilt"], 90)), 1) if np.any(~np.isnan(s["trunk_tilt"])) else None
    r["ankle_speed_median"] = round(float(np.nanmedian(s["ank_spd"])), 2)
    r["hops_per_min"] = round(len(s["hops"]) / dur_min, 1)

    # 击球瞬间
    per = []
    for t in swings:
        k = at(s["knee_min"], times, t)
        st = at(s["stance"], times, t)
        h0 = at(s["hip_h"], times, t)
        h_after = at(s["hip_h"], times, t + 0.5)
        h_norm = r["hip_h_median"]
        pre_hop = any(t - 1.2 <= hp <= t - 0.4 for hp in s["hops"])
        per.append({"t": round(t, 2), "label": seg_label(t), "knee": None if np.isnan(k) else round(k, 1),
                    "stance": None if np.isnan(st) else round(st, 2),
                    "sink": None if np.isnan(h0) else round(h_norm - h0, 2),
                    "rise_after": None if (np.isnan(h0) or np.isnan(h_after)) else round(h_after - h0, 2),
                    "pre_hop": pre_hop})
    valid = [p for p in per if p["knee"] is not None]
    r["swings"] = len(swings)
    r["swings_measured"] = len(valid)
    r.update(contact_stats(valid))
    if segments:
        r["by_segment"] = {lab: contact_stats([p for p in valid if p["label"] == lab])
                           for lab in dict.fromkeys(l for _, _, l in segments)}
    r["per_swing"] = per
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--swings", default=None, help="仅单文件时可手动指定挥拍时刻，逗号分隔")
    ap.add_argument("--segments", default=None, help="段落标签，如 0-180:定点,180-360:变化（仅单文件）")
    args = ap.parse_args()
    sw = [float(x) for x in args.swings.split(",")] if args.swings else None
    segs = parse_segments(args.segments) if len(args.npz) == 1 else None
    reps = [report_one(p, sw if len(args.npz) == 1 else None, segs) for p in args.npz]

    keys = [("pose_ratio", "有效帧占比"), ("knee_median", "膝角中位(°)"), ("knee_p10", "膝角p10(°)"),
            ("stance_median", "站位中位(小腿长)"), ("stance_p90", "站位p90"),
            ("hip_h_median", "重心高度中位(髋-踝/小腿长)"), ("trunk_tilt_median", "躯干侧倾中位(°)"),
            ("trunk_tilt_p90", "躯干侧倾p90(°)"), ("ankle_speed_median", "脚步活跃度(小腿长/s)"),
            ("hops_per_min", "垫步/分钟"), ("swings_measured", "可测挥拍数"),
            ("contact_knee_median", "击球瞬间膝角中位(°)"), ("contact_knee_lt150_ratio", "击球屈膝<150°占比"),
            ("contact_stance_median", "击球站位中位"), ("contact_stance_ok_ratio", "击球站位达标(1.3–1.8)占比"),
            ("contact_sink_median", "击球重心下沉量"),
            ("pre_hop_ratio", "击球前有垫步占比"), ("stay_down_ratio", "击球后留在下面占比")]
    w = max(len(k[1]) for k in keys) + 2
    cols = [(r["file"].replace("_landmarks.npz", ""), r) for r in reps]
    if len(reps) == 1 and reps[0].get("by_segment"):
        cols += [(f"  ·{lab}", {**{kk: None for kk, _ in keys}, **st, "swings_measured": st["n"]})
                 for lab, st in reps[0]["by_segment"].items()]
    print("指标".ljust(w) + "".join(c.ljust(18) for c, _ in cols))
    for k, label in keys:
        print(label.ljust(w) + "".join(str(r.get(k, "-") if r.get(k) is not None else "-").ljust(18) for _, r in cols))
    out = Path("out") / ("report_" + "_vs_".join(r["file"].replace("_landmarks.npz", "") for r in reps) + ".json")
    out.write_text(json.dumps(reps, ensure_ascii=False, indent=2))
    print(f"\n详细: {out}")


if __name__ == "__main__":
    main()
