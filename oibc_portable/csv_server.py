"""Loopback-only CSV testbed. Uploaded rows stay under .local-only/details/."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
from pathlib import Path
import secrets
import traceback
import pandas as pd
from .csv_testbed import Truth, Store
from .io import RunLock, event
from .research_archive import archive

PAGE = r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OIBC CSV 테스트베드</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#182e43;font:15px/1.65 system-ui,sans-serif}header{background:#142c45;color:white;padding:30px max(24px,calc((100vw - 1300px)/2))}main{max-width:1350px;margin:auto;padding:20px}h1{margin:0 0 6px;font-size:28px}h2{font-size:19px}.sub{color:#a8c4dd}.card{background:white;border:1px solid #dfe6ee;border-radius:12px;padding:22px;margin-bottom:18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}label{display:block;font-weight:600}input,select,button{font:inherit;padding:9px;border:1px solid #c8d4e1;border-radius:6px;max-width:100%}input{width:100%}button{cursor:pointer;background:#175a8d;color:white;border:0}button:disabled{opacity:.5}.hint{font-size:13px;color:#52667d}.scroll{overflow:auto;max-height:560px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #e0e8ef;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf3f8;position:sticky;top:0;cursor:pointer}#status{white-space:pre-wrap;color:#165b8f}.warning{background:#fff7df;padding:10px;border-radius:7px}details{margin:12px 0}summary{cursor:pointer}a{color:#175a8d}#historical td{white-space:normal;min-width:100px}nav button{margin:0 8px 16px 0}section[hidden]{display:none}.money{font-variant-numeric:tabular-nums;font-weight:600}</style>
<header><h1>OIBC CSV 테스트베드</h1><div class="sub">예측 CSV를 넣고 마지막 1년의 W·S를 바로 비교</div></header><main>
<div class="card"><div class="grid" id="context"></div><p class="hint">W·S·R은 원 단위로 반올림 표시합니다. 계산과 정렬에는 소수점 원래 값을 유지합니다. 외부 서버로 파일을 전송하지 않습니다.</p></div>
<nav><button id="scoresTab">CSV 채점·비교</button><button id="historyTab">기존 전체 실험·결과</button></nav>
<section id="scoring"><div class="card"><h2>예측 결과 추가</h2><div class="grid"><label>예측 CSV · 여러 파일 가능<input id="forecast" type="file" accept=".csv" multiple></label><label>배정 CSV · 참조 배정이면 생략<input id="composition" type="file" accept=".csv"></label><label>결과 이름 · 단일 파일일 때<input id="name" placeholder="예: LightGBM RAW6"></label><label>학습 마감 · 알고 있을 때만<input id="cutoff" type="date" max="2024-05-31"></label></div>
<p class="hint">집합 예측: group, ts, forecast_kwh · 발전소 예측: site_id, ts, forecast_kwh. 값 단위는 kWh, 시각에는 타임존이 필요합니다. 배정 CSV는 site_id, group입니다.</p><button id="score">CSV 채점</button><p id="status" role="status"></p><div class="warning">테스트 대상일은 2024-06-01~2025-05-31입니다. 공식 제출 기간 CSV는 이 화면의 채점 대상이 아닙니다. CSV만으로 학습 시점의 누수 여부를 증명할 수 없으므로 미확인·선언 여부를 함께 표시합니다.</div></div>
<div class="card"><h2>채점 결과</h2><button onclick="location.href='/api/export'">비교표 CSV 저장</button><label>결과 검색<input id="search" placeholder="모델·실험 이름"></label><div class="scroll"><table id="scores"><thead><tr><th data-sort="label">결과</th><th data-sort="S">S (원) ↑</th><th data-sort="W">W (원) ↓</th><th data-sort="R">R (원)</th><th>공통 정답 시간</th><th>예측 범위</th><th>학습 시점</th><th>출처</th></tr></thead><tbody></tbody></table></div><p class="hint">253개소가 모두 관측된 같은 시간으로 정산합니다. 없는 정답을0으로 채우지 않으며 서로 다른 집합의 오차를 상쇄하지 않습니다. 과거 레시피의 입력·학습 조건은 다를 수 있어 이 순위를 모델 구조만의 우열로 해석하지 않습니다.</p><div id="monthly"></div></div></section>
<section id="history" hidden><div class="card"><h2>지금까지의 전체 연구 기록</h2><p>과거 실험의 기간·입력·코호트가 다를 수 있어 새 CSV 순위와 분리합니다. 원래 기록의 수치는 보존하고 금액 표기만 원 단위로 맞췄습니다.</p><div class="grid"><label>실험<select id="experiment"><option value="">전체</option></select></label><label>검색<input id="historySearch" placeholder="모델·지표·실험"></label></div><p id="historyCount"></p><div id="known"></div><div id="historical"></div></div></section>
<script>
const TOKEN='__TOKEN__'; let results=[],historyData=null,sortKey='S',ascending=false;
const el=id=>document.getElementById(id),esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path,options={}){const r=await fetch(path,options);const d=await r.json();if(!r.ok)throw Error(d.error||'처리 실패');return d;}
function draw(){const q=el('search').value.toLowerCase();const rows=results.filter(r=>r.label.toLowerCase().includes(q));rows.sort((a,b)=>{const x=sortKey==='label'?a.label:a.settlement[sortKey],y=sortKey==='label'?b.label:b.settlement[sortKey];if(x==null)return 1;if(y==null)return -1;const d=typeof x==='string'?x.localeCompare(y):x-y;return ascending?d:-d;});
el('scores').querySelector('tbody').innerHTML=rows.map(r=>`<tr><td>${esc(r.label)}</td><td class="money">${esc(r.display.S)}</td><td class="money">${esc(r.display.W)}</td><td>${esc(r.display.R)}</td><td>${r.settlement.common_hours.toLocaleString()}</td><td>${r.full_test_grid?'전체 시각':'채점 가능 시각 포함'}</td><td>${r.training_time_check==='unverified'?'미확인':'마감 이전으로 선언'}</td><td>${r.bundled?'기존 저장 예측':'새 CSV'}</td></tr>`).join('');
el('monthly').innerHTML=rows.map(r=>`<details><summary>${esc(r.label)} · 월별 결과</summary><div class="scroll"><table><thead><tr><th>대상월</th><th>S (원)</th><th>W (원)</th><th>R (원)</th><th>정답 시간</th></tr></thead><tbody>${r.monthly.map(m=>`<tr><td>${esc(m.month)}</td><td>${esc(m.display.S)}</td><td>${esc(m.display.W)}</td><td>${esc(m.display.R)}</td><td>${m.common_hours}</td></tr>`).join('')}</tbody></table></div></details>`).join('');}
async function refresh(){results=await api('/api/results');draw();}
el('score').onclick=async()=>{const files=el('forecast').files;if(!files.length){el('status').textContent='예측 CSV를 선택해 주세요.';return;}el('score').disabled=true;try{const comp=el('composition').files[0];const composition_csv=comp?await comp.text():null;let n=0;for(const file of files){el('status').textContent=`채점 중 ${++n}/${files.length} · ${file.name}`;await api('/api/score',{method:'POST',headers:{'Content-Type':'application/json','X-OIBC-Token':TOKEN},body:JSON.stringify({forecast_csv:await file.text(),composition_csv,name:files.length===1&&el('name').value?el('name').value:file.name.replace(/\.csv$/i,''),trained_through:el('cutoff').value||null})});}await refresh();el('status').textContent=`${files.length}개 CSV 채점 완료`;}catch(e){el('status').textContent=e.message;}finally{el('score').disabled=false;}};
el('search').oninput=draw;document.querySelectorAll('[data-sort]').forEach(th=>th.onclick=()=>{const key=th.dataset.sort;ascending=key===sortKey?!ascending:key==='W';sortKey=key;draw();});
el('scoresTab').onclick=()=>{el('scoring').hidden=false;el('history').hidden=true;};el('historyTab').onclick=async()=>{el('scoring').hidden=true;el('history').hidden=false;if(!historyData){historyData=await api('/api/history');historyData.experiments.forEach(x=>{const o=document.createElement('option');o.value=x.id;o.textContent=x.id+' · '+x.title;el('experiment').appendChild(o);});}drawHistory();};
function drawHistory(){const id=el('experiment').value,q=el('historySearch').value.toLowerCase();const tables=historyData.tables.filter(t=>(!id||t.experiment_ids.includes(id))&&JSON.stringify(t).toLowerCase().includes(q));el('historyCount').textContent=`실험 ${historyData.experiments.length}개 · 기록 표 ${tables.length}/${historyData.tables.length}개`;
el('known').innerHTML=!id?'<h3>저장 예측 CSV 재채점 · '+historyData.rescored_results.length+'건</h3><div class="scroll"><table><thead><tr><th>정책</th><th>S (원)</th><th>W (원)</th><th>지원 시간</th></tr></thead><tbody>'+historyData.rescored_results.map(r=>`<tr><td>${esc(r.label)}</td><td>${esc(r.display.S)}</td><td>${esc(r.display.W)}</td><td>${r.settlement.common_hours}</td></tr>`).join('')+'</tbody></table></div><p class="hint">원본 데이터 지문이 같으면 CSV 비교 탭에도 기존 재채점 기록을 함께 표시합니다. EXP-026은 진행 중인 기록으로 최종 승격 결과를 가정하지 않습니다.</p>':'';
el('historical').innerHTML=tables.map(t=>`<details><summary>${esc(t.experiment_ids.join(', ')||'공통 기록')} · ${esc(t.heading)} · ${t.rows.length}행</summary><a target="_blank" rel="noreferrer" href="${esc(t.source_url)}">원래 기록 확인</a><div class="scroll"><table><thead><tr>${t.headers.map(h=>'<th>'+esc(h)+'</th>').join('')}</tr></thead><tbody>${t.rows.map(row=>'<tr>'+row.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')}</tbody></table></div></details>`).join('');}
el('experiment').onchange=drawHistory;el('historySearch').oninput=()=>historyData&&drawHistory();
api('/api/context').then(d=>{el('context').innerHTML=[['평가 대상일',d.test_start+' ~ '+d.test_end],['전체 시간',d.hours.toLocaleString()],['전수 정산 가능 시간',d.common_hours.toLocaleString()],['연간 완비 발전소',d.complete_cohort_sites+'개소']].map(([k,v])=>`<div><strong>${esc(k)}</strong><div>${esc(v)}</div></div>`).join('');}).then(refresh).catch(e=>el('status').textContent=e.message);
</script></main></html>'''


def handler(store,token,port):
    hosts={f"127.0.0.1:{port}",f"localhost:{port}"}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass  # No uploaded content, identifiers or row keys in access logs.
        def send(self,status,body,kind="application/json; charset=utf-8",download=None):
            raw=body if isinstance(body,bytes) else body.encode("utf-8") if isinstance(body,str) else json.dumps(body,ensure_ascii=False,allow_nan=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type",kind);self.send_header("Content-Length",str(len(raw)))
            if download: self.send_header("Content-Disposition",'attachment; filename="'+download+'"')
            self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff");self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if self.headers.get("Host") not in hosts: return self.send(403,{"error":"Local host required"})
            try:
                if self.path=="/": return self.send(200,PAGE.replace("__TOKEN__",token),"text/html; charset=utf-8")
                if self.path=="/api/context": return self.send(200,store.truth.context)
                if self.path=="/api/results": return self.send(200,store.results())
                if self.path=="/api/export": return self.send(200,store.export_scores().read_bytes(),"text/csv; charset=utf-8","oibc-scoreboard.csv")
                if self.path=="/api/history": return self.send(200,archive())
                return self.send(404,{"error":"Not found"})
            except Exception:
                with (store.folder/"server-errors.log").open("a",encoding="utf-8") as log: traceback.print_exc(file=log)
                return self.send(500,{"error":"저장된 결과나 데이터가 변경됐는지 로컬 로그를 확인해 주세요."})
        def do_POST(self):
            origin=self.headers.get("Origin")
            if self.headers.get("Host") not in hosts or self.headers.get("X-OIBC-Token")!=token or origin is not None and origin not in {"http://"+h for h in hosts}:
                return self.send(403,{"error":"Local authenticated page required"})
            if self.path!="/api/score": return self.send(404,{"error":"Not found"})
            try:
                size=int(self.headers.get("Content-Length","0"))
                if not 0<size<=256*1024**2: return self.send(413,{"error":"CSV 입력 크기를 확인해 주세요."})
                body=json.loads(self.rfile.read(size)); text=body["forecast_csv"]
                forecast=pd.read_csv(StringIO(text)); comp=pd.read_csv(StringIO(body["composition_csv"])) if body.get("composition_csv") else None
                row=store.add(forecast,comp,body.get("name") or "CSV result",body.get("trained_through"),text.encode("utf-8"))
                return self.send(200,{"ok":True,"id":row["id"]})
            except Exception:
                with (store.folder/"server-errors.log").open("a",encoding="utf-8") as log: traceback.print_exc(file=log)
                return self.send(400,{"error":"CSV의 열·kWh 단위·타임존·배정·테스트 기간·채점 가능 시각 누락을 확인해 주세요. 상세 오류는 로컬 로그에 저장했습니다."})
    return Handler


def serve(root,work,port=8765):
    truth=Truth(root);store=Store(truth,work); token=secrets.token_urlsafe(32)
    server=ThreadingHTTPServer(("127.0.0.1",port),handler(store,token,port))
    with RunLock(store.folder,resume=True):
        event("csv_testbed_ready",url=f"http://127.0.0.1:{port}",test_start=truth.start,test_end=truth.end,common_hours=int(truth.common.sum()))
        try: server.serve_forever()
        finally: server.server_close()
