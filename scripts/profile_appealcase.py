#!/usr/bin/env python3
"""步骤 3-A: AppealCase 原始数据盘点。

用法:
  python scripts/profile_appealcase.py \
      --data data/raw/appealcase/AppealCase.json \
      --meta data/raw/appealcase/AppealCase.metadata.json \
      --out-json reports/step3_appealcases_profile.json \
      --out-cause-csv reports/step3_appealcases_cause_freq.csv
"""
import argparse, json, re, statistics, collections, sys

def pct(vals, p):
    if not vals: return None
    vals = sorted(vals)
    k = (len(vals) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 1)

def stats(vals):
    vals = [v for v in vals if v]
    return {"n": len(vals), "mean": round(statistics.mean(vals), 1) if vals else None,
            "P50": pct(vals, 50), "P90": pct(vals, 90), "P99": pct(vals, 99),
            "max": max(vals) if vals else None}

MARKERS = {
    "原告诉称": ["原告诉称", "原告起诉称", "原告向本院提出诉讼请求", "向本院提出诉讼请求"],
    "被告辩称": ["被告辩称", "被告答辩称", "被告辩称意见"],
    "经审理查明": ["经审理查明", "本院查明", "本院经审理认定", "另查明"],
    "本院认为": ["本院认为"],
    "判决结果": ["判决如下", "裁定如下"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-cause-csv", required=True)
    args = ap.parse_args()

    data = json.load(open(args.data, encoding="utf-8"))
    meta = json.load(open(args.meta, encoding="utf-8"))
    assert len(data) == len(meta), (len(data), len(meta))

    cause_l1 = collections.Counter()
    cause_l2 = collections.Counter()
    mismatch = 0
    years = collections.Counter()
    year_missing = 0
    lens_claim, lens_fact, lens_input, lens_court, lens_judg = [], [], [], [], []
    marker_hits = {k: 0 for k in MARKERS}
    both_claim_fact = 0
    missing_claim = 0
    missing_fact = 0
    pair_type = collections.Counter()
    n = len(data)

    for rec, m in zip(data, meta):
        f = rec["first-instance"]
        claim = f.get("claim") or ""
        fact = f.get("fact_description") or ""
        court = f.get("court_view") or ""
        judg = f.get("judgment") or ""
        c1 = (m["first-instance"].get("cause_of_action") or "").strip()
        c2 = (m["second-instance"].get("cause_of_action") or "").strip()
        cause_l1[c1] += 1
        cause_l2[c2] += 1
        if c1 and c2 and c1 != c2: mismatch += 1
        pair_type["|".join(m["first-instance"].get("type") or [])] += 1

        no = (m["first-instance"].get("case") or {}).get("no") or ""
        mm = re.search(r"[（(](\d{4})[）)]", no)
        if mm:
            years[int(mm.group(1))] += 1
        else:
            year_missing += 1

        if claim.strip(): lens_claim.append(len(claim))
        else: missing_claim += 1
        if fact.strip(): lens_fact.append(len(fact))
        else: missing_fact += 1
        if claim.strip() and fact.strip(): both_claim_fact += 1
        lens_input.append(len(claim) + len(fact))
        lens_court.append(len(court)); lens_judg.append(len(judg))

        whole = claim + fact + court + judg
        for k, pats in MARKERS.items():
            if any(p in whole for p in pats): marker_hits[k] += 1

    out = {
        "n_records": n,
        "fields_first_instance": list(data[0]["first-instance"].keys()),
        "fields_second_instance": list(data[0]["second-instance"].keys()),
        "fields_top": list(data[0].keys()),
        "cause_field": "metadata.first-instance.cause_of_action / metadata.second-instance.cause_of_action",
        "n_distinct_cause_first": len([c for c in cause_l1 if c]),
        "n_distinct_cause_second": len([c for c in cause_l2 if c]),
        "cause_empty_first": cause_l1.get("", 0),
        "cause_empty_second": cause_l2.get("", 0),
        "cause_mismatch_l1_vs_l2": mismatch,
        "cause_mismatch_ratio": round(mismatch / n, 4),
        "cause_mismatch_ratio_of_both_nonempty": round(mismatch / max(1, n - cause_l1.get("", 0)), 4),
        "year_dist": dict(sorted(years.items())),
        "year_missing": year_missing,
        "len_claim": stats(lens_claim),
        "len_fact_description": stats(lens_fact),
        "len_input_claim_plus_fact": stats(lens_input),
        "len_court_view": stats(lens_court),
        "len_judgment": stats(lens_judg),
        "missing_claim": missing_claim,
        "missing_fact": missing_fact,
        "both_claim_and_fact": both_claim_fact,
        "both_claim_and_fact_ratio": round(both_claim_fact / n, 4),
        "marker_presence": {k: {"n": v, "pct": round(v / n * 100, 1)} for k, v in marker_hits.items()},
        "pair_type": dict(pair_type),
    }
    json.dump(out, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    with open(args.out_cause_csv, "w", encoding="utf-8-sig") as fh:
        fh.write("案由,条数,占比\n")
        for c, k in cause_l1.most_common():
            fh.write(f"{c},{k},{k/n*100:.2f}%\n")
    print(json.dumps(out, ensure_ascii=False, indent=1)[:3000])

if __name__ == "__main__":
    main()
