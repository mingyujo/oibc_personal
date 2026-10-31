"""Standalone aggregate HTML. No row predictions or site identifiers are embedded."""
from html import escape
import json
from pathlib import Path
import numpy as np
from .io import write


def compare(results, baseline, seeds, minimum_hours, repetitions=1000, block_days=7):
    ids = sorted({r["candidate"] for r in results})
    by_key = {(r["candidate"], r["seed"]): r for r in results}
    output = []
    for candidate in ids:
        rows = [by_key[(candidate, seed)] for seed in seeds]
        base = [by_key[(baseline, seed)] for seed in seeds]
        reference = by_key[(baseline,seeds[0])]
        for a, b in zip(rows, base):
            if any(r["daily_keys"] != reference["daily_keys"] or r["support_hours"] != reference["support_hours"] for r in (a,b)):
                raise ValueError("Candidates must use identical calendar and truth support")
            if not np.allclose(a["daily_R"], b["daily_R"], rtol=1e-12, atol=1e-6):
                raise ValueError("Revenue differs on supposedly identical support")
        support = rows[0]["settlement"]["common_hours"]
        eligible = support >= minimum_hours
        def mean_money(name):
            values = [r["settlement"][name] for r in rows]
            return float(np.mean(values)) if all(v is not None for v in values) else None
        a = np.mean([r["daily_W"] for r in rows], axis=0)
        b = np.mean([r["daily_W"] for r in base], axis=0)
        gains = b-a
        interval = None
        if eligible and len(gains):
            rng = np.random.default_rng(20261101); n = len(gains); width = min(block_days, n)
            samples = []
            for _ in range(repetitions):
                starts = rng.integers(0, n-width+1, size=int(np.ceil(n/width)))
                indices = (starts[:, None]+np.arange(width)).reshape(-1)[:n]
                samples.append(float(gains[indices].sum()))
            interval = np.quantile(samples, [.025, .975]).tolist()
        wins = sum(float(np.sum(rb["daily_W"])) > float(np.sum(ra["daily_W"])) for ra, rb in zip(rows, base))
        improvement = float(gains.sum()) if support else None
        status = "reference" if candidate == baseline else "insufficient_support" if not eligible else "descriptive_comparison_only"
        if candidate != baseline and eligible and len(seeds) >= 3 and wins >= 2 and interval and interval[0] > 0:
            status = "research_candidate_requires_further_validation"
        output.append({"candidate": candidate, "mean_S": mean_money("S"), "mean_W": mean_money("W"), "mean_R": mean_money("R"),
                       "mean_E": float(np.mean([r["all"]["E"] for r in rows])),
                       "mean_annual_complete_E": float(np.mean([r["annual_complete"]["E"] for r in rows])) if rows[0]["annual_complete"]["labels"] else None,
                       "complete_cohort_sites": rows[0]["complete_cohort_sites"], "common_hours": support,
                       "delta_S_vs_baseline": improvement, "delta_S_exploratory_95pct": interval,
                       "seed_wins": wins, "seeds": len(seeds), "ranking_supported": eligible, "status": status})
    output.sort(key=lambda r: (not r["ranking_supported"], -(r["mean_S"] if r["mean_S"] is not None else -float("inf")), r["candidate"]))
    return output


def render(folder, result):
    folder = Path(folder)
    def cell(value):
        if value is None: return "—"
        if isinstance(value, float): return f"{value:,.4f}"
        return escape(str(value))
    fields = [("candidate", "후보"), ("mean_S", "정산금 S ↑"), ("mean_W", "페널티 W ↓"),
              ("delta_S_vs_baseline", "기준 대비 ΔS"), ("common_hours", "공통 정답 시간"),
              ("mean_E", "전체 관측 MAE"), ("mean_annual_complete_E", "완비 코호트 MAE"), ("status", "판정 범위")]
    rows = "".join("<tr>"+"".join("<td>"+cell(row[key])+"</td>" for key, _ in fields)+"</tr>" for row in result["comparison"])
    cards = "".join(f"<p><strong>{escape(label)}</strong> {cell(value)}</p>" for label, value in (
        ("평가 기간", result["split"]["test_start"]+" ~ "+result["split"]["test_end"]),
        ("학습 마감", result["split"]["cutoff"]), ("프로필", result["profile"]),
        ("완비 코호트 발전소 수", result["complete_cohort_sites"])))
    details = []
    for job in result["results"]:
        monthly = "".join("<tr>"+"".join("<td>"+cell(month.get(k))+"</td>" for k in ("month", "labels", "E", "R", "W", "S", "common_hours"))+"</tr>" for month in job["monthly"])
        details.append(f'<details><summary>{escape(job["candidate"])} · seed {job["seed"]}</summary><table><thead><tr><th>월</th><th>관측 수</th><th>MAE</th><th>R</th><th>W</th><th>S</th><th>공통 시간</th></tr></thead><tbody>{monthly}</tbody></table></details>')
    intervals = "".join(f'<li>{escape(row["candidate"])}: ΔS 95% 탐색 구간 {cell(row["delta_S_exploratory_95pct"])}; 시드 개선 {row["seed_wins"]}/{row["seeds"]}</li>' for row in result["comparison"])
    evaluation_note = "가상 자료로 만든 단기 동작 검증입니다. 실제 대회 성능 점수가 아닙니다." if result["profile"] == "synthetic" else "과거 고정 1년 평가입니다. 이미 조회한 기간이므로 새 독립 최종시험이나 공식 리더보드 점수가 아닙니다."
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>OIBC 테스트베드</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1400px;margin:36px auto;padding:0 22px;color:#172c3b;background:#f5f7fa}}header,section{{background:white;padding:24px;margin:18px 0;border-radius:12px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dde4eb}}th{{background:#eef4fa}}.scroll{{overflow-x:auto}}input{{padding:10px;min-width:250px}}summary{{cursor:pointer;padding:12px}}.note{{color:#754a06}}</style>
<header><h1>OIBC 테스트베드</h1>{cards}<p>S = R − W, W = 0.10 × Σ SMP × k × (|G1 오차| + |G2 오차|)</p>
<p class="note">{evaluation_note} 정산은 모든 발전소의 정답이 있는 시간만 계산합니다. 시드별 점수 평균이며 예측 평균 앙상블이 아닙니다.</p></header>
<section><h2>후보 비교</h2><label>후보 검색 <input id="query" placeholder="후보 이름"></label><div class="scroll"><table id="ranking"><thead><tr>{''.join('<th>'+escape(label)+'</th>' for _,label in fields)}</tr></thead><tbody>{rows}</tbody></table></div>
<p>정답 지원이 부족한 행은 순위 근거가 없습니다. 자동 제출·모델 승격은 수행하지 않습니다. 같은 표본에서 R은 같으므로 S 증가는 W 감소와 같습니다.</p></section>
<section><h2>날짜 블록 불확실성</h2><p>{result['statistics']['block_days']}일 이동 블록 · {result['statistics']['replicates']}회 · 다중 비교 보정 없는 탐색 구간. 시드 평균 일별 차이를 날짜 단위로 재표집합니다.</p><ul>{intervals}</ul></section>
<section><h2>월별·시드별 결과</h2>{''.join(details)}</section>
<script>document.getElementById('query').addEventListener('input',e=>{{const q=e.target.value.toLowerCase();document.querySelectorAll('#ranking tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));}});</script></html>'''
    (folder/"report.html").write_text(html, encoding="utf-8")
    write(folder/"summary.json", result)
    return folder/"report.html"
