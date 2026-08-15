#!/usr/bin/env python3
"""两机位事后同步：不靠网络、不靠时钟，靠「两台手机都看见了同一串挥拍」。

原理：各自提取持拍手手腕速度序列（或挥拍时刻序列），做互相关，峰值处的偏移就是两段视频的时间差。
精度：一帧以内（60fps 下 ~17ms）。开拍时在两台手机都能看见的位置举手拍一下，会让峰更尖。

用法：
  sync.py A_landmarks.npz B_landmarks.npz            → 打印 offset（B 比 A 晚多少秒）与置信度
  sync.py A.npz B.npz --write out/sync_A_B.json     → 存结果供 report/card 合并两视角
自测：
  sync.py --selftest out/IMG_0011_landmarks.npz     → 人为切两段错开 3.37s，看能否恢复
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import VIS_TH, R_WRIST, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE


def wrist_speed(pts, times, fps):
    """手腕速度（小腿长/秒），插值填洞、无效段置 0"""
    n = len(times)
    ok = pts[:, R_WRIST, 2] > VIS_TH
    idx = np.where(ok)[0]
    if len(idx) < fps:
        return np.zeros(n)
    shin = (np.linalg.norm(pts[:, L_KNEE, :2] - pts[:, L_ANKLE, :2], axis=1)
            + np.linalg.norm(pts[:, R_KNEE, :2] - pts[:, R_ANKLE, :2], axis=1)) / 2
    med_shin = float(np.nanmedian(np.where(ok, shin, np.nan))) or 1.0
    wx = np.interp(np.arange(n), idx, pts[idx, R_WRIST, 0])
    wy = np.interp(np.arange(n), idx, pts[idx, R_WRIST, 1])
    spd = np.hypot(np.gradient(wx), np.gradient(wy)) * fps / max(med_shin, 1)
    spd[~ok] = 0
    # 压缩动态范围，避免个别大峰主导；再做零均值
    s = np.log1p(spd)
    return s - s.mean()


def resample(sig, src_fps, dst_fps):
    if abs(src_fps - dst_fps) < 1e-6:
        return sig
    t_src = np.arange(len(sig)) / src_fps
    t_dst = np.arange(0, t_src[-1], 1 / dst_fps)
    return np.interp(t_dst, t_src, sig)


def find_offset(sig_a, sig_b, fps, max_offset_s=600):
    """返回 (offset_s, confidence)：offset>0 表示 B 比 A 晚开拍 offset 秒（B 的 t=0 对应 A 的 t=offset）"""
    n = len(sig_a) + len(sig_b)
    fa = np.fft.rfft(sig_a, n)
    fb = np.fft.rfft(sig_b, n)
    cc = np.fft.irfft(fa * np.conj(fb), n)          # cc[k] = sum_t a[t] * b[t-k]
    lags = np.arange(n)
    lags[lags > n // 2] -= n                          # 负滞后
    m = np.abs(lags) <= max_offset_s * fps
    cc, lags = cc[m], lags[m]
    k = int(np.argmax(cc))
    peak = cc[k]
    # 置信度：峰值相对第二高（排除峰邻域 ±0.5s）
    mask = np.abs(lags - lags[k]) > 0.5 * fps
    second = cc[mask].max() if mask.any() else 0
    conf = float(peak / max(second, 1e-9))
    return lags[k] / fps, conf


def sync_npz(path_a, path_b):
    A, B = np.load(path_a), np.load(path_b)
    fa, fb = float(A["fps"]), float(B["fps"])
    fps = min(fa, fb)
    sa = resample(wrist_speed(A["pts"], A["times"], fa), fa, fps)
    sb = resample(wrist_speed(B["pts"], B["times"], fb), fb, fps)
    off, conf = find_offset(sa, sb, fps)
    # 视频自身的 times 起点（--start 裁剪过的话不为 0）
    off_total = off + float(A["times"][0]) - float(B["times"][0])
    return {"a": Path(path_a).name, "b": Path(path_b).name,
            "offset_b_minus_a_s": round(float(off_total), 3),
            "confidence": round(conf, 2),
            "verdict": "可信" if conf >= 2.0 else ("勉强" if conf >= 1.4 else "不可信：换一段或开拍时拍手做标记"),
            "how_to_use": "B 视频中的时刻 tB 对应 A 视频中的 tA = tB + offset"}


def selftest(path):
    d = np.load(path)
    pts, times, fps = d["pts"], d["times"], float(d["fps"])
    true_off = 3.37
    ia = int(20 * fps)                # A: 从 20s 开始
    ib = int((20 + true_off) * fps)   # B: 从 23.37s 开始（B 比 A 晚 3.37s）
    L = int(80 * fps)
    tmp = Path("out/_synctest")
    tmp.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp / "A.npz", pts=pts[ia:ia + L], times=times[ia:ia + L] - times[ia], fps=fps)
    np.savez_compressed(tmp / "B.npz", pts=pts[ib:ib + L], times=times[ib:ib + L] - times[ib], fps=fps)
    r = sync_npz(tmp / "A.npz", tmp / "B.npz")
    err = abs(r["offset_b_minus_a_s"] - true_off)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print(f"真实偏移 {true_off}s，恢复 {r['offset_b_minus_a_s']}s，误差 {err*1000:.0f}ms（{err*fps:.1f} 帧） → {'PASS' if err < 1.5 / fps else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--write", default=None)
    ap.add_argument("--selftest", default=None, help="用一段 landmarks.npz 自测")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.selftest)
        return
    if not (args.a and args.b):
        ap.error("需要两个 landmarks.npz")
    r = sync_npz(args.a, args.b)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if args.write:
        Path(args.write).write_text(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
