#!/usr/bin/env python3
"""
分析 MoE 模型的專家使用頻率，估算「熱門專家放記憶體、冷門專家留在 SSD」時，
每個 token 還要從 SSD 讀多少資料。

步驟：
  1. 用 llama.cpp 收集專家選中次數（用你自己平常的問題/文章當 calib.txt）：
       llama-imatrix -m model.gguf -f calib.txt -o imatrix.gguf
  2. 分析：
       python3 expert_usage.py --model model-00001-of-0000N.gguf --imatrix imatrix.gguf

需要：pip install gguf numpy
"""

import argparse
import glob
import json
import re
import sys

import numpy as np
from gguf import GGUFReader

EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight$")
GB = 1024**3
MB = 1024**2


def model_shards(path):
    """llama.cpp 分割檔 xxx-00001-of-00004.gguf：自動找出全部分割檔。"""
    m = re.match(r"^(.*)-\d{5}-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    shards = sorted(glob.glob(f"{m.group(1)}-*-of-{m.group(2)}.gguf"))
    if len(shards) != int(m.group(2)):
        sys.exit(f"找不到全部分割檔：預期 {m.group(2)} 個，只找到 {len(shards)} 個")
    return shards


def read_model(path):
    """回傳 (每層每個專家的位元組數 {layer: bytes}, 專家數, 每 token 選幾個專家, 非專家權重總位元組, 全部位元組)"""
    expert_bytes = {}
    n_expert = n_used = None
    other_bytes = 0
    for shard in model_shards(path):
        r = GGUFReader(shard)
        for name, field in r.fields.items():
            if name.endswith(".expert_count"):
                n_expert = int(field.contents())
            elif name.endswith(".expert_used_count"):
                n_used = int(field.contents())
        for t in r.tensors:
            m = EXPS_RE.match(t.name)
            if m:
                layer = int(m.group(1))
                n_as = int(t.shape[-1])
                expert_bytes[layer] = expert_bytes.get(layer, 0) + int(t.n_bytes) // n_as
            else:
                other_bytes += int(t.n_bytes)
    if not expert_bytes or not n_expert or not n_used:
        sys.exit("這個模型不是 MoE，或讀不到 expert_count / expert_used_count")
    total = other_bytes + sum(b * n_expert for b in expert_bytes.values())
    return expert_bytes, n_expert, n_used, other_bytes, total


def read_counts(path):
    """從 imatrix GGUF 讀出 {layer: 每個專家被選中次數}"""
    counts = {}
    r = GGUFReader(path)
    for t in r.tensors:
        if not t.name.endswith(".counts"):
            continue
        m = EXPS_RE.match(t.name[: -len(".counts")])
        if not m:
            continue
        layer = int(m.group(1))
        c = np.asarray(t.data, dtype=np.float64).ravel()
        if c.size > 1:
            counts[layer] = c  # gate/up/down 三個張量的次數相同，取一個即可
    if not counts:
        sys.exit("imatrix 裡沒有專家次數。請用新版 llama-imatrix 以 GGUF 格式輸出（-o xxx.gguf）")
    return counts


def build_table(expert_bytes, counts, n_used):
    """所有專家攤平成一張表：層、專家編號、大小、平均每 token 被選中的機率。"""
    rows = []
    for layer, c in counts.items():
        if layer not in expert_bytes:
            continue
        tokens = c.sum() / n_used
        for e, n in enumerate(c):
            rows.append((layer, e, expert_bytes[layer], n / tokens))
    return rows


def hot_set(rows, budget_bytes):
    """在預算內，依「每位元組省下的讀取量」由高到低挑熱門專家（同層大小相同時就是依使用率）。"""
    order = sorted(rows, key=lambda r: r[3], reverse=True)
    picked, used = [], 0
    for r in order:
        if used + r[2] > budget_bytes:
            continue
        picked.append(r)
        used += r[2]
    return picked, used


def summarize(rows, picked):
    hot = {(r[0], r[1]) for r in picked}
    total_sel = sum(r[3] for r in rows)
    hot_sel = sum(r[3] for r in picked)
    # 冷門專家每 token 平均要讀的量（假設冷門專家沒有快取，保守估計）
    cold_bytes_per_tok = sum(r[2] * r[3] for r in rows if (r[0], r[1]) not in hot)
    return hot_sel / total_sel, cold_bytes_per_tok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="模型 GGUF（分割檔給第一個即可）")
    ap.add_argument("--imatrix", required=True, help="llama-imatrix 輸出的 imatrix.gguf")
    ap.add_argument("--budget-gb", type=float, nargs="*", default=[20, 30, 40],
                    help="可放專家的記憶體預算（GB，顯存+RAM 扣掉非專家權重、KV cache、系統）")
    ap.add_argument("--ssd-gbps", type=float, default=3.0, help="SSD 實際隨機讀取速度 GB/s（用來估算速度上限）")
    ap.add_argument("--hot-out", help="把最大預算的熱門專家清單存成 JSON")
    args = ap.parse_args()

    expert_bytes, n_expert, n_used, other_bytes, total = read_model(args.model)
    counts = read_counts(args.imatrix)
    rows = build_table(expert_bytes, counts, n_used)

    all_expert_bytes = sum(r[2] for r in rows)
    active_bytes = other_bytes + sum(expert_bytes[l] * n_used for l in counts if l in expert_bytes)
    print(f"模型總大小          : {total / GB:8.2f} GB")
    print(f"非專家權重（必讀）  : {other_bytes / GB:8.2f} GB")
    print(f"全部專家            : {all_expert_bytes / GB:8.2f} GB（{len(counts)} 層 × {n_expert} 個，每 token 選 {n_used} 個）")
    print(f"每 token 啟用權重   : {active_bytes / GB:8.2f} GB")
    print()

    # 使用集中度：每層要幾個專家才能涵蓋 50/90/99% 的路由
    print("每層需要多少比例的專家，才能涵蓋 X% 的路由（平均 / 最差層）")
    for q in (0.5, 0.9, 0.99):
        need = []
        for c in counts.values():
            s = np.sort(c)[::-1].cumsum() / c.sum()
            need.append((np.searchsorted(s, q) + 1) / c.size)
        print(f"  {q:4.0%} : 平均 {np.mean(need):6.1%}   最差 {np.max(need):6.1%}")
    unused = sum(int((c == 0).sum()) for c in counts.values())
    print(f"  完全沒被選中的專家：{unused} / {sum(c.size for c in counts.values())}")
    print()

    print(f"{'預算':>8} {'熱門專家數':>10} {'命中率':>8} {'SSD 讀取/token':>16} {'SSD 上限 tok/s':>16}")
    picked_last = []
    for b in args.budget_gb:
        picked, used = hot_set(rows, b * GB)
        hit, cold = summarize(rows, picked)
        tps = (args.ssd_gbps * GB / cold) if cold > 0 else float("inf")
        print(f"{b:6.1f}GB {len(picked):>10} {hit:>8.1%} {cold / MB:>13.1f} MB {tps:>16.1f}")
        picked_last = picked
    print()
    print("註：SSD 讀取量假設冷門專家完全不在快取（保守估計）；實際上系統快取會再吸收一部分。")
    print("    命中率只代表你的 calib 文本；換用途時分布可能不同。")

    if args.hot_out:
        out = {}
        for layer, e, _, p in picked_last:
            out.setdefault(str(layer), []).append(e)
        with open(args.hot_out, "w") as f:
            json.dump({k: sorted(v) for k, v in sorted(out.items(), key=lambda x: int(x[0]))}, f)
        print(f"熱門專家清單已存到 {args.hot_out}")


if __name__ == "__main__":
    main()
