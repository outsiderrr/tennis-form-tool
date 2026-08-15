#!/usr/bin/env python3
"""单维度教学示意视频：在球员火柴人上叠加该维度的辅助线/角度弧/参考线 + 建议字幕。

用法：
  annotate.py <视频> <landmarks.npz> --t 78.55 --dim knee|stance|hop|rise [--all]
输出：
  out/annot/<name>_<t>s_<dim>.mp4   示意短片（击球前 1.0s ~ 后 0.8s）
  out/annot/<name>_<t>s_<dim>.png   该维度最有说明力的一帧

复用 analyze.py 存下的关键点，不重跑姿态。
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

from analyze import (CONNECTIONS, LEFT_IDS, VIS_TH, angle_deg,
                     L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)

W, H = 640, 880           # 画布
CAP_H = 96                # 底部字幕条
BOX_W_SHIN, BOX_H_SHIN = 5.0, 7.0
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"

GREEN, AMBER, CORAL, WHITE, GRAY = (80, 220, 80), (40, 170, 255), (60, 90, 240), (240, 240, 240), (170, 170, 170)

DIMS = {
    "stance": {"title": "站位宽度", "advice": "已达标（目标 1.3–1.6 小腿长）· 这是地基，保持住"},
    "knee":   {"title": "击球瞬间屈膝", "advice": "膝盖先弯 · 目标击球时 <150°（现在中位 160°）"},
    "hop":    {"title": "分腿垫步", "advice": "教练触球瞬间小跳一下 · 目标 >50% 击球前有垫步（现在 3%）",
               "lines": [
                   ("标准", "教练击球的一瞬间起跳，双脚同时落地、膝盖弯——落在你击球前 0.5–1.2 秒（图中绿色区间）"),
                   ("她的情况", "绿色区间里两只脚都贴地（蓝橙两线都在 0 附近）；击球附近的抬起是前脚跨步 / 后脚脚跟，单脚动作，不是垫步"),
                   ("为什么", "落地那一下把腿部肌肉预先拉紧（压弹簧），第一步更快；双脚同时重置 = 每球回到平衡、屈膝的准备姿势；跟着对方节奏，不会反应晚"),
               ]},
    "rise":   {"title": "击球后重心", "advice": "头和髋的高度到随挥结束都不变 · 现在 49% 的球会站起来"},
}


def put_cjk(img_bgr, text, xy, size=22, color=(240, 240, 240), anchor="la"):
    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(FONT, size)
    except OSError:
        font = ImageFont.load_default()
    d.text(xy, text, fill=(color[2], color[1], color[0]), font=font, anchor=anchor)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_dashed(img, p1, p2, color, gap=10, thick=2):
    p1, p2 = np.array(p1, float), np.array(p2, float)
    L = np.linalg.norm(p2 - p1)
    if L < 1:
        return
    n = int(L // gap)
    for k in range(0, n, 2):
        a = p1 + (p2 - p1) * (k / n)
        b = p1 + (p2 - p1) * (min(k + 1, n) / n)
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thick, cv2.LINE_AA)


def draw_angle_arc(img, hip, knee, ankle, color, radius=44):
    """在膝点画厚-小腿夹角弧"""
    v1, v2 = np.array(hip) - np.array(knee), np.array(ankle) - np.array(knee)
    a1 = np.degrees(np.arctan2(v1[1], v1[0]))
    a2 = np.degrees(np.arctan2(v2[1], v2[0]))
    d = (a2 - a1 + 540) % 360 - 180  # 最短方向
    start, end = (a1, a1 + d) if d > 0 else (a1 + d, a1)
    cv2.ellipse(img, tuple(int(x) for x in knee), (radius, radius), 0, start, end, color, 3, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("npz")
    ap.add_argument("--t", type=float, required=True, help="击球时刻（秒）")
    ap.add_argument("--dim", choices=list(DIMS), default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--before", type=float, default=1.0)
    ap.add_argument("--after", type=float, default=0.8)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    dims = list(DIMS) if args.all or not args.dim else [args.dim]

    d = np.load(args.npz)
    pts, times, fps = d["pts"], d["times"], float(d["fps"])
    n = len(times)
    vis = lambda j: pts[:, j, 2] > VIS_TH
    xy = lambda j: pts[:, j, :2]

    # 全局量：小腿长、踝基线（1.5s 滚动中位）、髋中点、踝中点
    shin_all = (np.linalg.norm(xy(L_KNEE) - xy(L_ANKLE), axis=1) + np.linalg.norm(xy(R_KNEE) - xy(R_ANKLE), axis=1)) / 2
    shin = float(np.nanmedian(np.where(vis(L_KNEE) & vis(L_ANKLE) & vis(R_KNEE) & vis(R_ANKLE), shin_all, np.nan)))
    hip_ok, ank_ok = vis(L_HIP) & vis(R_HIP), vis(L_ANKLE) & vis(R_ANKLE)
    hip = np.where(hip_ok[:, None], (xy(L_HIP) + xy(R_HIP)) / 2, np.nan)
    ank = np.where(ank_ok[:, None], (xy(L_ANKLE) + xy(R_ANKLE)) / 2, np.nan)
    idx = np.where(ank_ok)[0]
    ank_y_i = np.interp(np.arange(n), idx, ank[idx, 1])
    win = int(fps * 1.5) | 1
    padded = np.pad(ank_y_i, win // 2, mode="edge")
    ank_base = np.array([np.median(padded[i:i + win]) for i in range(n)])

    def foot_lift(j):
        y = xy(j)[:, 1]
        ok = vis(j)
        ii = np.where(ok)[0]
        yi = np.interp(np.arange(n), ii, y[ii])
        # 5 帧中值滤波去单帧抖动
        k5 = 5
        ypad = np.pad(yi, k5 // 2, mode="edge")
        ym = np.array([np.median(ypad[t:t + k5]) for t in range(n)])
        return np.where(ok, (ank_base - ym) / shin, np.nan)
    lift_l, lift_r = foot_lift(L_ANKLE), foot_lift(R_ANKLE)
    BLUE_L, ORANGE_R = (255, 160, 40), (60, 120, 255)  # 与火柴人左/右点同色（BGR）

    CHART_H, HOP_CAP_H = 300, 200

    def hop_chart(cur_i, t_c):
        """双脚离地曲线面板：横轴=距击球时间(s)，纵轴=离地高度(小腿长倍数)"""
        ch = np.full((CHART_H, W, 3), 24, np.uint8)
        gx0, gx1, gy0, gy1 = 70, W - 20, 36, CHART_H - 58
        t_lo, t_hi, y_max = -args.before, args.after, 0.4
        tx = lambda t: int(gx0 + (t - t_lo) / (t_hi - t_lo) * (gx1 - gx0))
        ty = lambda v: int(gy1 - np.clip(v, 0, y_max) / y_max * (gy1 - gy0))
        # 标准区间（绿色带）
        cv2.rectangle(ch, (tx(max(t_lo, -1.2)), gy0), (tx(-0.5), gy1), (40, 70, 40), -1)
        ch = put_cjk(ch, "标准：垫步应落在这里", ((tx(max(t_lo, -1.2)) + tx(-0.5)) // 2, gy0 + 4), 16, GREEN, "ma")
        # 坐标轴
        cv2.line(ch, (gx0, gy1), (gx1, gy1), GRAY, 1)
        cv2.line(ch, (gx0, gy0), (gx0, gy1), GRAY, 1)
        for tt in np.arange(np.ceil(t_lo * 2) / 2, t_hi + 0.01, 0.5):
            cv2.line(ch, (tx(tt), gy1), (tx(tt), gy1 + 5), GRAY, 1)
            if abs(tt) > 1e-6:
                ch = put_cjk(ch, f"{tt:+.1f}", (tx(tt), gy1 + 8), 15, GRAY, "ma")
        for vv in (0.0, 0.1, 0.2, 0.3, 0.4):
            cv2.line(ch, (gx0 - 5, ty(vv)), (gx0, ty(vv)), GRAY, 1)
            ch = put_cjk(ch, f"{vv:.1f}", (gx0 - 9, ty(vv) - 9), 15, GRAY, "ra")
        ch = put_cjk(ch, "横轴：距击球的时间（秒）", ((gx0 + gx1) // 2, gy1 + 28), 16, GRAY, "ma")
        ch = put_cjk(ch, "纵轴：脚离地高度（小腿长倍数）", (gx0 - 62, 8), 15, GRAY, "la")
        # 击球线
        cv2.line(ch, (tx(0), gy0), (tx(0), gy1), WHITE, 1)
        ch = put_cjk(ch, "击球", (tx(0), gy1 + 8), 15, WHITE, "ma")
        # 两条曲线
        seg = np.arange(i0, i1)
        for series, col in ((lift_l, BLUE_L), (lift_r, ORANGE_R)):
            pts_g = [(tx(times[s] - t_c), ty(series[s])) for s in seg if not np.isnan(series[s])]
            if len(pts_g) > 1:
                cv2.polylines(ch, [np.array(pts_g, np.int32)], False, col, 2, cv2.LINE_AA)
        # 双脚同时离地 = 垫步 → 绿点
        both = [(tx(times[s] - t_c), ty(min(lift_l[s], lift_r[s]))) for s in seg
                if not np.isnan(lift_l[s]) and not np.isnan(lift_r[s]) and lift_l[s] > 0.10 and lift_r[s] > 0.10]
        for pt in both:
            cv2.circle(ch, pt, 4, GREEN, -1)
        # 图例
        cv2.line(ch, (gx1 - 300, gy0 + 12), (gx1 - 275, gy0 + 12), BLUE_L, 2)
        ch = put_cjk(ch, "左脚", (gx1 - 270, gy0 + 3), 15, GRAY, "la")
        cv2.line(ch, (gx1 - 230, gy0 + 12), (gx1 - 205, gy0 + 12), ORANGE_R, 2)
        ch = put_cjk(ch, "右脚", (gx1 - 200, gy0 + 3), 15, GRAY, "la")
        cv2.circle(ch, (gx1 - 160, gy0 + 12), 4, GREEN, -1)
        ch = put_cjk(ch, "双脚同时离地=垫步", (gx1 - 152, gy0 + 3), 15, GRAY, "la")
        # 当前时刻游标
        xc = tx(times[cur_i] - t_c)
        cv2.line(ch, (xc, gy0), (xc, gy1), (120, 120, 120), 1)
        return ch

    # 片段窗口 + 固定裁剪框（窗口内髋中点中位为中心，垂直运动可见）
    i0 = int(np.searchsorted(times, args.t - args.before))
    i1 = int(np.searchsorted(times, args.t + args.after))
    ic = int(np.argmin(np.abs(times - args.t)))
    cxy = np.nanmedian(hip[i0:i1], axis=0)
    bw, bh = BOX_W_SHIN * shin, BOX_H_SHIN * shin
    x0, y0 = cxy[0] - bw / 2, cxy[1] - bh * 0.42
    sx, sy = W / bw, H / bh
    to_c = lambda p: (int((p[0] - x0) * sx), int((p[1] - y0) * sy))

    cap = cv2.VideoCapture(args.video)
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    frames = []
    for _ in range(i0, i1):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    out_dir = Path("out/annot")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{Path(args.video).stem}_{args.t:.1f}s"

    # 击球帧的髋/踝高度（rise 维度参考线）
    hip_c = hip[ic] if hip_ok[ic] else np.nanmedian(hip[max(ic - 3, 0):ic + 4], axis=0)

    for dim in dims:
        out_h = H + (CHART_H + HOP_CAP_H if dim == "hop" else CAP_H)
        writer = cv2.VideoWriter(str(out_dir / f"{name}_{dim}.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 60.0, (W, out_h))
        if not writer.isOpened():
            writer = cv2.VideoWriter(str(out_dir / f"{name}_{dim}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (W, out_h))
        key_png, key_score = None, -1e9

        for k, frame in enumerate(frames):
            i = i0 + k
            rel = times[i] - args.t
            p = pts[i]
            # 裁剪 + 缩放
            pad_l, pad_t = max(0, -int(x0)), max(0, -int(y0))
            pad_r, pad_b = max(0, int(x0 + bw) - fw), max(0, int(y0 + bh) - fh)
            img = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(24, 24, 24))
            crop = img[int(y0) + pad_t:int(y0 + bh) + pad_t, int(x0) + pad_l:int(x0 + bw) + pad_l]
            canvas = cv2.resize(crop, (W, H))

            # 火柴人
            drw = {j: to_c(p[j, :2]) for j in range(33) if p[j, 2] > VIS_TH}
            for a, b in CONNECTIONS:
                if a in drw and b in drw:
                    cv2.line(canvas, drw[a], drw[b], GREEN, 2, cv2.LINE_AA)
            for j, q in drw.items():
                cv2.circle(canvas, q, 4, (60, 120, 255) if j in LEFT_IDS else (255, 160, 40), -1, cv2.LINE_AA)

            readout, score = "", -1e8
            # ---------- 维度专属叠加 ----------
            if dim == "stance" and all(j in drw for j in (L_ANKLE, R_ANKLE)):
                la, ra = drw[L_ANKLE], drw[R_ANKLE]
                yline = max(la[1], ra[1]) + 22
                wdt = abs(p[L_ANKLE, 0] - p[R_ANKLE, 0]) / shin
                col = GREEN if 1.3 <= wdt <= 1.8 else AMBER
                cv2.arrowedLine(canvas, (la[0], yline), (ra[0], yline), col, 3, cv2.LINE_AA, tipLength=0.06)
                cv2.arrowedLine(canvas, (ra[0], yline), (la[0], yline), col, 3, cv2.LINE_AA, tipLength=0.06)
                for a in (la, ra):
                    draw_dashed(canvas, a, (a[0], yline), col, gap=6, thick=1)
                if all(j in drw for j in (L_SHOULDER, R_SHOULDER)):
                    ls, rs = drw[L_SHOULDER], drw[R_SHOULDER]
                    for s_ in (ls, rs):
                        draw_dashed(canvas, (s_[0], s_[1]), (s_[0], yline + 34), GRAY, gap=8, thick=1)
                    canvas = put_cjk(canvas, "肩宽参考", ((ls[0] + rs[0]) // 2, yline + 40), 18, GRAY, "ma")
                canvas = put_cjk(canvas, f"站位 {wdt:.2f} 小腿长" + ("  · 达标" if col == GREEN else "  · 偏窄"), (W // 2, 24), 26, col, "ma")
                canvas = put_cjk(canvas, "两踝间距 ÷ 小腿长", (W // 2, 60), 18, GRAY, "ma")
                readout, score = f"站位 {wdt:.2f}", -abs(rel)

            elif dim == "knee":
                for hj, kj, aj, side in ((L_HIP, L_KNEE, L_ANKLE, "L"), (R_HIP, R_KNEE, R_ANKLE, "R")):
                    if all(j in drw for j in (hj, kj, aj)):
                        ang = angle_deg(p[hj, :2], p[kj, :2], p[aj, :2])
                        col = GREEN if ang < 150 else (AMBER if ang < 160 else CORAL)
                        draw_angle_arc(canvas, drw[hj], drw[kj], drw[aj], col)
                        kx, ky = drw[kj]
                        off = -92 if side == "L" else 26
                        canvas = put_cjk(canvas, f"{ang:.0f}°", (kx + off, ky - 14), 26, col, "la")
                        readout += f"{side}{ang:.0f}° "
                        score = -abs(rel)
                if abs(rel) < 0.02:
                    canvas = put_cjk(canvas, "击球瞬间", (W // 2, 24), 26, WHITE, "ma")
                    canvas = put_cjk(canvas, "目标 <150°", (W // 2, 58), 20, GREEN, "ma")

            elif dim == "hop":
                by = to_c((0, ank_base[i]))[1]
                draw_dashed(canvas, (30, by), (W - 30, by), GRAY, gap=12, thick=1)
                canvas = put_cjk(canvas, "站定基线", (34, by - 26), 18, GRAY, "la")
                for j, series, col, nm in ((L_ANKLE, lift_l, BLUE_L, "左脚"), (R_ANKLE, lift_r, ORANGE_R, "右脚")):
                    if j in drw and not np.isnan(series[i]):
                        ax_, ay_ = drw[j]
                        cv2.circle(canvas, (ax_, ay_), 8, col, -1, cv2.LINE_AA)
                        cv2.line(canvas, (ax_, ay_), (ax_, by), col, 2, cv2.LINE_AA)
                        canvas = put_cjk(canvas, f"{nm} {max(series[i], 0):.2f}", (ax_ + 12, ay_ - 30), 18, col, "la")
                        score = -abs(rel + 0.8)
                if -1.2 <= rel <= -0.5:
                    canvas = put_cjk(canvas, "标准区间：这里该有一个双脚小跳", (W // 2, 24), 22, GREEN, "ma")
                elif -0.5 < rel < 0.05:
                    canvas = put_cjk(canvas, "跨步进球 → 击球（单脚动作，不算垫步）", (W // 2, 24), 22, GRAY, "ma")

            elif dim == "rise":
                hy = to_c(hip_c)[1]
                draw_dashed(canvas, (30, hy), (W - 30, hy), WHITE, gap=12, thick=2)
                canvas = put_cjk(canvas, "击球时髋的高度", (34, hy - 26), 18, WHITE, "la")
                if hip_ok[i]:
                    hp = to_c(hip[i])
                    delta = (hip_c[1] - hip[i, 1]) / shin  # 正=上浮
                    col = (AMBER if delta > 0.08 else GREEN) if rel > 0 else GRAY
                    cv2.circle(canvas, hp, 9, col, -1, cv2.LINE_AA)
                    if rel > 0.02:
                        cv2.arrowedLine(canvas, (hp[0] + 60, hy), (hp[0] + 60, hp[1]), col, 3, cv2.LINE_AA, tipLength=0.3)
                        canvas = put_cjk(canvas, f"{'上浮' if delta > 0 else '下沉'} {abs(delta):.2f}", (hp[0] + 74, (hp[1] + hy) // 2 - 12), 22, col, "la")
                        readout = f"髋 {'+' if delta > 0 else ''}{delta:.2f}"
                        score = -abs(rel - 0.4)
                if rel > 0.02:
                    canvas = put_cjk(canvas, "击球后：留在下面，别站起来", (W // 2, 24), 22, WHITE, "ma")

            # 字幕条
            if dim == "hop":
                cap_bar = np.full((HOP_CAP_H, W, 3), 24, np.uint8)
                cap_bar = put_cjk(cap_bar, f"{DIMS[dim]['title']}  {args.label}", (16, 6), 22, WHITE)
                cap_bar = put_cjk(cap_bar, f"t={rel:+.2f}s", (W - 16, 8), 18, GRAY, "ra")
                yy = 38
                for head, body in DIMS[dim]["lines"]:
                    col = GREEN if head == "标准" else (AMBER if head == "她的情况" else WHITE)
                    cap_bar = put_cjk(cap_bar, f"{head}：", (16, yy), 15, col)
                    # 简单折行（按字数）
                    maxc = 37
                    chunks = [body[c:c + maxc] for c in range(0, len(body), maxc)]
                    for ci, chunk in enumerate(chunks):
                        cap_bar = put_cjk(cap_bar, chunk, (16 + 72, yy + ci * 18), 15, (215, 215, 215))
                    yy += 18 * len(chunks) + 4
                frame_out = np.vstack([canvas, hop_chart(i, args.t), cap_bar])
            else:
                cap_bar = np.full((CAP_H, W, 3), 24, np.uint8)
                cap_bar = put_cjk(cap_bar, f"{DIMS[dim]['title']}  {args.label}", (16, 10), 24, WHITE)
                cap_bar = put_cjk(cap_bar, DIMS[dim]["advice"], (16, 48), 20, AMBER if dim != "stance" else GREEN)
                cap_bar = put_cjk(cap_bar, f"t={rel:+.2f}s", (W - 16, 12), 20, GRAY, "ra")
                frame_out = np.vstack([canvas, cap_bar])
            writer.write(frame_out)
            if score > key_score:
                key_score, key_png = score, frame_out.copy()
        writer.release()
        if key_png is not None:
            cv2.imwrite(str(out_dir / f"{name}_{dim}.png"), key_png)
        print(f"{dim}: {out_dir}/{name}_{dim}.mp4 / .png")


if __name__ == "__main__":
    main()
