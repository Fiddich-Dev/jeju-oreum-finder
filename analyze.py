"""제주관광빅데이터플랫폼 '오름별 차량 도착 수'(티맵 내비게이션) 분석과 대체 오름 추천 데이터 생성.

입력: data/oreum_2021_2026.csv (플랫폼 데이터 픽 regSn=91 CSV, 2021-01~2026-09)
출력: out/summary.json, out/recommend.json, out/fig_*.png, docs/index.html(추천 페이지)
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
(ROOT / "docs").mkdir(exist_ok=True)

o = pd.read_csv(ROOT / "data/oreum_2021_2026.csv", encoding="utf-8-sig")
o.columns = ["ym", "name", "addr", "res", "tour", "total"]
o["y"] = o.ym // 100
o["m"] = o.ym % 100
o["emd"] = o.addr.map(lambda a: re.split(r"\s+", a)[2] if len(re.split(r"\s+", a)) > 2 else "")
o["si"] = o.addr.map(lambda a: re.split(r"\s+", a)[1] if len(re.split(r"\s+", a)) > 1 else "")

summary = {"rows": int(len(o)), "oreum_count": int(o.name.nunique()),
           "period": [int(o.ym.min()), int(o.ym.max())]}

# 1) 관광객 도착의 상위 5개 오름 집중도 (연도별, 집계 로직이 바뀐 2026년은 별도 표기)
conc = []
for y, g in o.groupby("y"):
    d = g.groupby("name").tour.sum().sort_values(ascending=False)
    conc.append({"year": int(y), "months": int(g.ym.nunique()), "oreum": int((d > 0).sum()),
                 "top5": list(d.head(5).index), "top5_share": round(float(d.head(5).sum() / d.sum()), 4),
                 "top1_share": round(float(d.iloc[0] / d.sum()), 4)})
summary["concentration"] = conc

# 2) 2025년 오름별 지표
base = o[o.y == 2025].groupby(["name", "si", "emd"])[["res", "tour", "total"]].sum().reset_index()
base["tour_share"] = base.tour / base.total.where(base.total > 0)
top5 = conc[[c["year"] for c in conc].index(2025)]["top5"]

# 월별 계절성(2023~2025 평균, 관광객 도착 기준)
season = o[o.y.between(2023, 2025)]
mon = season.assign(grp=season.name.isin(top5).map({True: "상위 5개 오름", False: "나머지 오름"})) \
    .groupby(["grp", "m"]).tour.sum().unstack(0)
mon_share = (mon / mon.sum()).round(4)
summary["month_share_2023_2025"] = {k: {int(m): float(v) for m, v in mon_share[k].items()} for k in mon_share}

# 3) 대체 오름 추천: 같은 읍면에서 연간 전체 도착 1,000건 이상(접근성이 확인된 곳)이면서
#    관광객 비중이 기준 오름보다 낮은 오름. 도민이 많이 찾는 곳을 '검증된 대안'으로 본다.
MIN_TOTAL = 1000
# 행정구역이 맞닿은 읍면(같은 읍면에 대안이 없을 때만 씀)
NEIGHBORS = {"한림읍": ["한경면", "애월읍"], "조천읍": ["구좌읍"], "안덕면": ["대정읍"], "애월읍": ["한림읍"],
             "구좌읍": ["조천읍", "성산읍"], "표선면": ["성산읍", "남원읍"], "남원읍": ["표선면"]}
rec = []
for _, r in base.sort_values("tour", ascending=False).head(15).iterrows():
    ok = (base.name != r["name"]) & (base.total >= MIN_TOTAL) & (base.tour_share < r.tour_share) & (base.tour < r.tour)
    cand = base[ok & (base.emd == r.emd)].sort_values("tour_share")
    scope = "같은 읍면"
    if cand.empty:
        cand = base[ok & base.emd.isin(NEIGHBORS.get(r.emd, []))].sort_values("tour_share")
        scope = "맞닿은 읍면"
    months = o[(o.name == r["name"]) & o.y.between(2023, 2025)].groupby("m").tour.sum()
    quiet = [int(m) for m in months.sort_values().index[:3]]
    rec.append({
        "name": r["name"], "emd": f"{r.si} {r.emd}", "tour_2025": int(r.tour), "res_2025": int(r.res),
        "tour_share": round(float(r.tour_share), 3),
        "monthly_tour_2023_2025": {int(m): int(v) for m, v in months.items()},
        "quiet_months": sorted(quiet), "alt_scope": scope if not cand.empty else "",
        "alternatives": [{"name": c["name"], "total_2025": int(c.total), "tour_2025": int(c.tour),
                          "res_2025": int(c.res), "tour_share": round(float(c.tour_share), 3)}
                         for _, c in cand.head(3).iterrows()],
    })
summary["recommend_rule"] = (f"같은 읍면(없으면 맞닿은 읍면)에서 2025년 전체 도착 {MIN_TOTAL}건 이상이고 관광객 비중과 "
                             "관광객 도착 수가 모두 기준 오름보다 낮은 오름을 관광객 비중 낮은 순으로 3곳")
summary["top15_with_alternative"] = sum(1 for r in rec if r["alternatives"])
summary["top15_without_alternative"] = [r["name"] for r in rec if not r["alternatives"]]

json.dump(rec, open(OUT / "recommend.json", "w"), ensure_ascii=False, indent=1)

# 그림 1: 상위 5개 오름 집중도
fig, ax = plt.subplots(figsize=(7, 3.6))
ys = [c["year"] for c in conc]
ax.bar([str(y) + ("*" if y == 2026 else "") for y in ys], [c["top5_share"] * 100 for c in conc], color="#2b6cb0")
for i, c in enumerate(conc):
    ax.text(i, c["top5_share"] * 100 + 1, f'{c["top5_share"]*100:.1f}%', ha="center", fontsize=9)
ax.set_ylim(0, 100)
ax.set_ylabel("관광객 도착 중 상위 5개 오름 비중(%)")
ax.set_title("오름 관광객 내비 도착의 상위 5개 집중도 (*2026은 1~9월, 집계 로직 변경)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig1_concentration.png", dpi=160); plt.close(fig)

# 그림 2: 월별 분포
fig, ax = plt.subplots(figsize=(7, 3.6))
for k, c in [("상위 5개 오름", "#c53030"), ("나머지 오름", "#2f855a")]:
    ax.plot(range(1, 13), [mon_share[k].get(m, 0) * 100 for m in range(1, 13)], marker="o", label=k, color=c)
ax.set_xticks(range(1, 13)); ax.set_xlabel("월"); ax.set_ylabel("연간 관광객 도착 중 해당 월 비중(%)")
ax.set_title("월별 관광객 도착 분포 (2023~2025 합계)", fontsize=10); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "fig2_month.png", dpi=160); plt.close(fig)

# 그림 3: 2025년 오름별 전체 도착 vs 관광객 비중
b = base[base.total >= 300]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(b.total, b.tour_share * 100, s=14, color="#718096")
for _, r in b.iterrows():
    if r["name"] in top5 or any(r["name"] in [a["name"] for a in x["alternatives"]] for x in rec[:5]):
        ax.scatter([r.total], [r.tour_share * 100], s=24, color="#c53030" if r["name"] in top5 else "#2f855a")
        ax.annotate(r["name"], (r.total, r.tour_share * 100), fontsize=8, xytext=(3, 3), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("2025년 전체 도착 수(로그)"); ax.set_ylabel("관광객 비중(%)")
ax.set_title("2025년 오름별 도착 규모와 관광객 비중 (빨강: 상위 5개, 초록: 추천 대안)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig3_scatter.png", dpi=160); plt.close(fig)

# 대안 오름들의 관광객 도착 합 vs 상위 5개
alt_names = {a["name"] for r in rec[:5] for a in r["alternatives"]}
summary["alt_names_for_top5"] = sorted(alt_names)
summary["alt_tour_2025"] = int(base[base.name.isin(alt_names)].tour.sum())
summary["top5_tour_2025"] = int(base[base.name.isin(top5)].tour.sum())
json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)

# 추천 페이지(정적 HTML 한 장, 외부 요청 없음)
html = (ROOT / "page_template.html").read_text().replace("/*DATA*/", json.dumps(rec, ensure_ascii=False))
(ROOT / "docs/index.html").write_text(html)
print(json.dumps(summary, ensure_ascii=False, indent=1))
