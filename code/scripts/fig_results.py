import json, numpy as np
from common import *
S = json.load(open(f"{RES}/summary_seeds.json")); B = json.load(open(f"{RES}/bench.json"))
Q = json.load(open(f"{RES}/r2_seeds.json")); BT = json.load(open(f"{RES}/build_times.json"))
F = S["full"]
# 7. 시간 분할 vs 무작위 분할 (5번 반복 평균)
fig, ax = plt.subplots(figsize=(9, 4.2)); names = ["R0","R1"]; lab = ["R0 (코드 안 봄)","R1 (+ 변경 통계)"]; x = np.arange(2); w = 0.36
tv = [F[n]["time_auc"]["mean"] for n in names]; rv = [F[n]["random_auc"]["mean"] for n in names]
ax.bar(x-w/2, rv, w-0.03, color="#86b6ef", label="무작위 분할 (미래 PR이 학습에 섞임)", edgecolor=SURF)
ax.bar(x+w/2, tv, w-0.03, color="#2a78d6", label="시간 분할 (과거로 학습 → 미래로 평가)", edgecolor=SURF)
for i in range(2):
    ax.text(x[i]-w/2, rv[i], f"{rv[i]:.3f}", ha="center", va="bottom", fontsize=10, color=INK2)
    ax.text(x[i]+w/2, tv[i], f"{tv[i]:.3f}", ha="center", va="bottom", fontsize=10, color=INK2)
ax.set_xticks(x, lab); ax.set_ylim(0.5, 0.9); ax.set_ylabel("AUC (5번 반복 평균)"); ax.grid(axis="x", visible=False)
ax.set_title("무작위로 나누면 점수가 약 0.08 부풀려진다"); ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig7_split_compare.png"); plt.close(fig)
# 8. pandas vs DuckDB
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for k,(pre,title) in enumerate([("pr","PR 테이블 (362MB)"),("cd","파일 변경 테이블 (12.4GB)")]):
    rows = [("pandas\n전체 읽기", B[f"{pre}_pandas_all"]), ("pandas\n필요한 열만", B[f"{pre}_pandas_cols"]), ("DuckDB", B[f"{pre}_duckdb"])]
    for i,(lb,r) in enumerate(rows):
        if r["ok"]:
            ax[k].bar(i, r["peak_rss_mb"], color=["#eb6834","#eda100","#2a78d6"][i], width=0.6)
            ax[k].text(i, r["peak_rss_mb"], f"{r['peak_rss_mb']:,} MB\n{r['sec_median']:.2f}초", ha="center", va="bottom", fontsize=9.5, color=INK2)
        else:
            ax[k].bar(i, 2200, color=SURF, edgecolor="#d03b3b", hatch="//", width=0.6)
            ax[k].text(i, 2200, "실패\n(2.2GB 초과)", ha="center", va="bottom", fontsize=9.5, color="#d03b3b")
    ax[k].set_xticks(range(3), [r[0] for r in rows]); ax[k].set_ylim(0, 2900); ax[k].set_ylabel("최대 메모리 (MB)"); ax[k].set_title(title); ax[k].grid(axis="x", visible=False)
fig.suptitle("같은 집계를 세 방식으로: 최대 메모리와 시간 (메모리 약 4GB 가상 환경, 2.2GB를 넘으면 중단)", fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(f"{FIG}/fig8_bench.png"); plt.close(fig)
# 9. 표현별 성능 vs 비용 (같은 표본, 5번 반복 평균)
med = lambda v: float(np.median(v))
r0_cost = med(BT["build_r0_sec"]) / BT["n_pr_r0_input"] * 1000
r1_cost = r0_cost + med(BT["build_r1_sec"]) / BT["n_pr_r1_input"] * 1000
T = Q["times"]; n_s = Q["n_train"] + Q["n_test"]
r2_cost = r1_cost + (T["extract_sec"] + T["tfidf_sec"] + T["svd_sec"]) / n_s * 1000
r2_est = r1_cost + (T["extract_sec"] / 6 + T["tfidf_sec"] + T["svd_sec"]) / n_s * 1000
A = {n: S["sample"][n]["auc"]["mean"] for n in ["R0","R1","R2"]}
pts = [("R0 코드 안 봄", r0_cost, A["R0"]), ("R1 + 변경 통계", r1_cost, A["R1"]), ("R2 + diff 단어 빈도", r2_cost, A["R2"])]
json.dump({"cost_s_per_1k": {p[0]: p[1] for p in pts}, "r2_cost_est_single_scan": r2_est,
           "ratio_r2_r1_measured": r2_cost / r1_cost, "ratio_r2_r1_est": r2_est / r1_cost}, open(f"{RES}/cost_per_1k.json","w"), ensure_ascii=False, indent=1)
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot([p[1] for p in pts], [p[2] for p in pts], color=GRID, lw=2, zorder=1)
for (lb,c,a),col in zip(pts, ["#86b6ef","#2a78d6","#184f95"]):
    ax.scatter([c],[a], s=110, color=col, zorder=3, edgecolor=SURF, linewidth=2)
    ax.annotate(f"{lb}\nAUC {a:.3f} · {c:.2f}초/1천 PR", (c,a), textcoords="offset points", xytext=(10,8), fontsize=9.5, color=INK)
ax.scatter([r2_est],[A["R2"]], s=110, facecolor="none", edgecolor="#184f95", linewidth=1.5, linestyle="--", zorder=3)
ax.annotate(f"R2 추정\n(파일을 한 번만 읽는다면 ≈{r2_est:.2f}초)", (r2_est, A["R2"]), textcoords="offset points", xytext=(-20,-38), fontsize=8.5, color=INK2, ha="center")
ax.set_xscale("log"); ax.set_xlabel("PR 1,000개당 표현 만드는 시간 (초, 로그 눈금)"); ax.set_ylabel("AUC (5번 반복 평균)")
ax.set_ylim(0.66, 0.75); ax.set_xlim(0.01, 30); ax.set_title("diff를 단어 빈도로 읽으면 비용은 15~60배, 성능은 R1과 비슷했다 (표본 3만+3만)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig9_cost_vs_auc.png"); plt.close(fig)
print(json.dumps(json.load(open(f"{RES}/cost_per_1k.json")), ensure_ascii=False))
