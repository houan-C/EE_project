#!/usr/bin/env python3
"""
fec_sim.py — 封包遺失模型 + 封包層 FEC（抹除碼）模擬

用途
----
手上只有「封包成功率」、沒有逐包紀錄時，用模型造出各種遺失型態
（隨機遺失、連續遺失），看封包層 FEC 在每一種型態下的效果。

重要前提（寫報告時要照實寫）
--------------------------
1. 成功率只決定「平均掉多少」，決定不了「怎麼掉」。所以這裡把
   平均連續遺失長度 L 當成「未知參數」去掃描，結果的意思是：
   「如果通道的 L 是 X，FEC 效果是 Y」，而不是「我們的通道就是這樣」。
   之後用 link_test 量到真正的 L，就能在圖上找到對應的那一點。
2. 封包層 FEC 用的是 MDS 抹除碼（例如 Reed-Solomon 抹除碼）：
   一個區塊送 n = k + r 包，只要掉的包數 ≤ r，k 個資料包就能全部還原。
   這裡直接用這個性質判斷成功或失敗，沒有真的做編解碼。
3. 遺失是以「封包序號」為時間軸，不考慮封包之間的實際時間間隔。

遺失模型
--------
- iid（隨機遺失）：每包獨立，以機率 p 遺失。
- Gilbert（連續遺失）：兩狀態馬可夫鏈。Good 狀態不掉包，Bad 狀態必掉包。
      P(Bad→Good) = 1/L                 → 平均連續遺失長度 = L
      P(Good→Bad) = (1/L)·p/(1−p)        → 長期平均遺失率 = p
  L = 1 時很接近 iid；L 越大，掉包越成串。

兩種實驗
--------
block : 不需要影像資料。對每個 (成功率, L, FEC 參數 k+r) 算「區塊失敗率」。
video : 需要 H.265 的 I／P 幀大小。模擬 GOP 依賴（P 幀依賴前一幀、
        I 幀遺失整個 GOP 都壞）、AES-GCM（一幀只要有資料包沒救回就整幀失敗）、
        以及空中速率上限，算出「有效 FPS」。

用法範例
--------
  python fec_sim.py selftest
  python fec_sim.py block --psr 0.99,0.95,0.87,0.65 --bursts 1,2,4,8,16 --codes 10+0,10+1,10+2,10+4,20+4
  python fec_sim.py video --i-bytes 6000 --p-bytes 600 --gop 30 --src-fps 30 --pkt-rate 100 \
                          --psr 0.999,0.99,0.97,0.95,0.9,0.87 --bursts 1,4,16 --rho 0,0.1,0.25,0.5
  （video 的幀大小請換成你們實際量到的數字，上面只是示範格式）
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime

import numpy as np


# ===========================================================================
# 遺失模型
# ===========================================================================
def gen_iid(n, p_loss, rng):
    """回傳長度 n 的 bool 陣列，True = 該包遺失。"""
    return rng.random(n) < p_loss


def gilbert_params(p_loss, mean_burst):
    r = 1.0 / mean_burst                 # Bad → Good
    q = r * p_loss / (1.0 - p_loss)      # Good → Bad
    return q, r


def gilbert_valid(p_loss, mean_burst):
    if not (0 < p_loss < 1) or mean_burst < 1:
        return False
    q, _ = gilbert_params(p_loss, mean_burst)
    return q <= 1.0


def gen_gilbert(n, p_loss, mean_burst, rng):
    """用「交替的 Good 段 / Bad 段」產生 Gilbert 序列（每段長度服從幾何分布）。"""
    if p_loss <= 0:
        return np.zeros(n, dtype=bool)
    if p_loss >= 1:
        return np.ones(n, dtype=bool)
    q, r = gilbert_params(p_loss, mean_burst)
    if q > 1:
        raise ValueError(f"遺失率 {p_loss:.3f} 下，平均連續遺失長度 L={mean_burst} 太短，"
                         f"Gilbert 模型無解（需要 L ≥ {p_loss / (1 - p_loss):.2f}）")
    start_bad = rng.random() < p_loss
    cycle_len = 1.0 / q + 1.0 / r
    out, total = [], 0
    while total < n:
        m = int(math.ceil((n - total) / cycle_len * 1.2)) + 16
        good = rng.geometric(q, m)
        bad = rng.geometric(r, m)
        lens = np.empty(2 * m, dtype=np.int64)
        states = np.empty(2 * m, dtype=bool)
        if start_bad:
            lens[0::2], lens[1::2] = bad, good
            states[0::2], states[1::2] = True, False
        else:
            lens[0::2], lens[1::2] = good, bad
            states[0::2], states[1::2] = False, True
        seg = np.repeat(states, lens)
        out.append(seg)
        total += len(seg)
    return np.concatenate(out)[:n]


def gen_loss(n, p_loss, mean_burst, rng):
    """mean_burst = 1 視為 iid，其餘用 Gilbert。"""
    if mean_burst <= 1:
        return gen_iid(n, p_loss, rng)
    return gen_gilbert(n, p_loss, mean_burst, rng)


def trace_stats(mask):
    """實際產生的序列：遺失率、平均連續遺失長度、最長連續遺失。"""
    n = len(mask)
    lost = int(mask.sum())
    if lost == 0:
        return 0.0, 0.0, 0
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    runs = ends - starts
    return lost / n, float(runs.mean()), int(runs.max())


# ===========================================================================
# FEC 判斷（MDS 抹除碼：掉的包數 ≤ r 就能還原）
# ===========================================================================
def block_fail_rate(p_loss, mean_burst, k, r, n_blocks, rng):
    n = k + r
    mask = gen_loss(n_blocks * n, p_loss, mean_burst, rng)
    losses = mask.reshape(n_blocks, n).sum(axis=1)
    if r == 0:
        fail = losses > 0
    else:
        fail = losses > r
    return float(fail.mean()), trace_stats(mask)


def binom_tail(n, r, p):
    """iid 理論值：n 包中掉超過 r 包的機率。"""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(r + 1, n + 1))


# ===========================================================================
# 實驗：selftest
# ===========================================================================
def cmd_selftest(args):
    rng = np.random.default_rng(args.seed)
    print("=== 自我檢查 1：iid 模擬值 vs 二項分布理論值 ===")
    print(f"{'k+r':>6} {'p_loss':>7} {'模擬':>10} {'理論':>10}")
    for (k, r) in [(10, 0), (10, 2), (20, 4)]:
        for p in [0.01, 0.05, 0.13]:
            sim, _ = block_fail_rate(p, 1, k, r, 200_000, rng)
            th = binom_tail(k + r, r, p)
            print(f"{k:>3}+{r:<2} {p:>7.2f} {sim:>10.5f} {th:>10.5f}")
    print("\n=== 自我檢查 2：Gilbert 產生的序列是否符合設定的 p 與 L ===")
    print(f"{'設定 p':>7} {'設定 L':>7} {'實際 p':>8} {'實際 L':>8} {'最長':>6}")
    for p in [0.01, 0.05, 0.35]:
        for L in [2, 4, 16]:
            if not gilbert_valid(p, L):
                print(f"{p:>7.2f} {L:>7} {'（此組合無解）':>20}")
                continue
            m = gen_gilbert(2_000_000, p, L, rng)
            ap, aL, mx = trace_stats(m)
            print(f"{p:>7.2f} {L:>7} {ap:>8.4f} {aL:>8.2f} {mx:>6}")


# ===========================================================================
# 實驗：block
# ===========================================================================
def parse_list(s, typ=float):
    return [typ(x) for x in s.split(',') if x.strip()]


def parse_codes(s):
    codes = []
    for item in s.split(','):
        item = item.strip()
        if not item:
            continue
        k, r = item.split('+')
        codes.append((int(k), int(r)))
    return codes


def make_outdir(base, tag):
    d = os.path.join(base, f"{tag}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(d, exist_ok=True)
    return d


def setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Noto Sans CJK TC',
                                       'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def cmd_block(args):
    rng = np.random.default_rng(args.seed)
    psrs = parse_list(args.psr)
    bursts = parse_list(args.bursts)
    codes = parse_codes(args.codes)
    outdir = make_outdir(args.outdir, 'block')

    rows = []
    for psr in psrs:
        p = 1 - psr
        for L in bursts:
            if L > 1 and not gilbert_valid(p, L):
                print(f"[略過] 成功率 {psr:.3f}、L={L:g}：Gilbert 模型無解")
                continue
            for (k, r) in codes:
                fail, (ap, aL, mx) = block_fail_rate(p, L, k, r, args.blocks, rng)
                rows.append(dict(psr=psr, p_loss=p, mean_burst=L, k=k, r=r,
                                 overhead=r / k, block_fail=fail,
                                 trace_p=ap, trace_L=aL, trace_maxrun=mx))
                print(f"成功率 {psr:6.3f}  L={L:>4g}  {k:>3}+{r:<2} (開銷 {r / k:5.0%})"
                      f"  區塊失敗率 {fail:.5f}   [序列實際 p={ap:.4f}, L={aL:.2f}]")

    csv_path = os.path.join(outdir, 'block_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    plt = setup_mpl()
    ncol = min(2, len(psrs))
    nrow = math.ceil(len(psrs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.5 * nrow), squeeze=False)
    floor = 0.5 / args.blocks
    for ax, psr in zip(axes.flat, psrs):
        for (k, r) in codes:
            pts = [(row['mean_burst'], max(row['block_fail'], floor)) for row in rows
                   if row['psr'] == psr and row['k'] == k and row['r'] == r]
            if not pts:
                continue
            xs, ys = zip(*pts)
            label = f"k={k}, r={r}（開銷 {r / k:.0%}）" if r else f"k={k}, 無 FEC"
            ax.plot(xs, ys, marker='o', label=label)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_title(f"封包成功率 {psr:.1%}")
        ax.set_xlabel("平均連續遺失長度 L（包）")
        ax.set_ylabel("區塊失敗率")
        ax.axhline(floor, color='gray', lw=0.6, ls=':')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(psrs):]:
        ax.axis('off')
    fig.suptitle("封包層 FEC：區塊失敗率 vs 連續遺失長度（L=1 為隨機遺失）")
    fig.tight_layout()
    png = os.path.join(outdir, 'block_fail_vs_burst.png')
    fig.savefig(png, dpi=130)

    with open(os.path.join(outdir, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)
    print(f"\n輸出：{csv_path}\n      {png}\n（圖上灰色虛線以下 = 模擬次數不足以分辨，視為 < {floor:.0e}）")


# ===========================================================================
# 實驗：video（H.265 GOP + AES-GCM + 空中速率上限）
# ===========================================================================
def frame_packets(frame_bytes, payload):
    return max(1, math.ceil(frame_bytes / payload))


def cmd_video(args):
    if args.i_bytes is None or args.p_bytes is None:
        raise SystemExit("[錯誤] video 實驗需要 --i-bytes 和 --p-bytes（H.265 I 幀、P 幀的平均大小，"
                         "含 AES-GCM 的 nonce 與 tag）。請用你們 TX 實際量到的數字。")
    rng = np.random.default_rng(args.seed)
    psrs = parse_list(args.psr)
    bursts = parse_list(args.bursts)
    rhos = parse_list(args.rho)
    outdir = make_outdir(args.outdir, 'video')

    m_i = frame_packets(args.i_bytes, args.payload)
    m_p = frame_packets(args.p_bytes, args.payload)
    print(f"每包可用資料 {args.payload} B → I 幀 {m_i} 包，P 幀 {m_p} 包，GOP={args.gop}")

    # 每一種 FEC 設定 = (圖例名稱, rho 欄位值, I 幀校驗包數, P 幀校驗包數)
    schemes = []
    if args.parity:
        for r in parse_list(args.parity, int):
            which = "開銷" if args.gop == 1 else "I 幀開銷"
            label = "無 FEC" if r == 0 else f"每幀 +{r} 包（{which} {r / m_i:.0%}）"
            schemes.append((label, r / m_i, r, r))
    else:
        for rho in rhos:
            r_i = math.ceil(rho * m_i) if rho > 0 else 0
            r_p = math.ceil(rho * m_p) if rho > 0 else 0
            schemes.append(("無 FEC" if rho == 0 else f"ρ={rho:.0%}", rho, r_i, r_p))

    rows = []
    for (label, rho, r_i, r_p) in schemes:
        pkts_per_gop = (m_i + r_i) + (args.gop - 1) * (m_p + r_p)
        avg_pkts = pkts_per_gop / args.gop
        tx_fps = min(args.src_fps, args.pkt_rate / avg_pkts)
        sizes = np.array([m_i + r_i] + [m_p + r_p] * (args.gop - 1))
        rs = np.array([r_i] + [r_p] * (args.gop - 1))
        bounds = np.concatenate(([0], np.cumsum(sizes)))
        print(f"\n{label}：I 幀 {m_i}+{r_i}，P 幀 {m_p}+{r_p}，平均每幀 {avg_pkts:.2f} 包，"
              f"送得出的 FPS 上限 {tx_fps:.2f}")
        for psr in psrs:
            p = 1 - psr
            for L in bursts:
                if L > 1 and not gilbert_valid(p, L):
                    continue
                mask = gen_loss(args.gops * pkts_per_gop, p, L, rng)
                lost = np.add.reduceat(mask.reshape(args.gops, pkts_per_gop).astype(np.int32),
                                       bounds[:-1], axis=1)
                frame_ok = lost <= rs[None, :]          # 掉的包數 ≤ 校驗包數 → 整幀救回
                decodable = np.cumprod(frame_ok, axis=1)  # P 幀依賴前一幀
                survival = float(decodable.mean())
                eff_fps = tx_fps * survival
                rows.append(dict(scheme=label, parity_i=r_i, parity_p=r_p, rho=rho, psr=psr, p_loss=p, mean_burst=L,
                                 pkts_per_frame=avg_pkts, tx_fps=tx_fps,
                                 frame_survival=survival, effective_fps=eff_fps))
                print(f"  成功率 {psr:6.3f}  L={L:>4g}  可解碼比例 {survival:6.3f}  有效 FPS {eff_fps:6.2f}")

    csv_path = os.path.join(outdir, 'video_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    plt = setup_mpl()
    ncol = min(3, len(bursts))
    nrow = math.ceil(len(bursts) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.3 * nrow), squeeze=False,
                             sharey=True)
    for ax in list(axes.flat)[len(bursts):]:
        ax.axis('off')
    for ax, L in zip(axes.flat, bursts):
        ax.tick_params(labelleft=True)
        for (label, _, _, _) in schemes:
            pts = sorted((row['psr'], row['effective_fps']) for row in rows
                         if row['scheme'] == label and row['mean_burst'] == L)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot([x * 100 for x in xs], ys, marker='o', label=label)
        ax.set_title("隨機遺失 (L=1)" if L <= 1 else f"連續遺失 L={L:g}")
        ax.set_xlabel("封包成功率 (%)")
        ax.set_ylabel("有效 FPS")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    size_txt = (f"每幀 {args.i_bytes:g} B" if args.gop == 1
                else f"I={args.i_bytes:g} B, P={args.p_bytes:g} B")
    fig.suptitle(f"H.265 有效 FPS（{size_txt}, GOP={args.gop}, "
                 f"空中上限 {args.pkt_rate:g} 包/秒，各圖縱軸相同）")
    fig.tight_layout()
    png = os.path.join(outdir, 'video_fps_vs_psr.png')
    fig.savefig(png, dpi=130)
    with open(os.path.join(outdir, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)
    print(f"\n輸出：{csv_path}\n      {png}")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="封包遺失模型 + 封包層 FEC 模擬")
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--outdir', default='fec_results')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('selftest', help='檢查模擬器本身是否正確')

    b = sub.add_parser('block', help='區塊失敗率 vs 連續遺失長度')
    b.add_argument('--psr', default='0.99,0.95,0.87,0.65', help='封包成功率清單')
    b.add_argument('--bursts', default='1,2,4,8,16', help='平均連續遺失長度 L 清單（1 = 隨機遺失）')
    b.add_argument('--codes', default='10+0,10+1,10+2,10+4,20+4', help='FEC 參數 k+r 清單')
    b.add_argument('--blocks', type=int, default=200_000, help='每組模擬幾個區塊')

    v = sub.add_parser('video', help='H.265 有效 FPS')
    v.add_argument('--i-bytes', type=float, help='I 幀平均大小（bytes，含加密 overhead）')
    v.add_argument('--p-bytes', type=float, help='P 幀平均大小（bytes，含加密 overhead）')
    v.add_argument('--gop', type=int, default=30)
    v.add_argument('--src-fps', type=float, default=30.0)
    v.add_argument('--pkt-rate', type=float, default=100.0,
                   help='空中每秒最多送幾包（請用 link_test 速率掃描量到的值）')
    v.add_argument('--payload', type=int, default=222,
                   help='每個 RF 封包可放的影像資料 bytes（230 B 扣掉 FEC 標頭）')
    v.add_argument('--psr', default='0.999,0.99,0.97,0.95,0.9,0.87')
    v.add_argument('--bursts', default='1,4,16')
    v.add_argument('--rho', default='0,0.1,0.25,0.5', help='校驗包比例 r/k 清單（0 = 無 FEC）')
    v.add_argument('--parity', help='直接指定每幀加幾個校驗包，例如 0,1,2,3,4（給了就不用 --rho）')
    v.add_argument('--gops', type=int, default=20_000, help='模擬幾個 GOP')

    args = ap.parse_args()
    {'selftest': cmd_selftest, 'block': cmd_block, 'video': cmd_video}[args.cmd](args)


if __name__ == '__main__':
    main()
