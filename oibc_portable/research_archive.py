"""Read the complete aggregate-table history bundled with the testbed."""
from importlib.resources import files
import json
import re
from decimal import Decimal
from .display import won


def monetary(header):
    header=str(header)
    if re.search(r"pct|%|ratio|fraction|gate|WAPE|weight",header,re.I): return False
    return bool(re.search(r"(?<![A-Za-z])[WRS](?![A-Za-z])|Wtotal|(?:penalty|settlement|revenue).*KRW|손실.*원|정산.*원|\((?:백만|만|천)?원\)",header))


def archived_cell(value,header):
    if not monetary(header): return value
    raw=re.sub(r"[*`원\s]","",str(value))
    cleaned=raw.replace(",","")
    scale=1000000 if "백만원" in header else 10000 if "만원" in header else 1000 if "천원" in header else 1
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",cleaned): return won(Decimal(cleaned)*scale)
    if raw.startswith("[") and raw.endswith("]"):
        try:
            interval=json.loads(raw,parse_float=Decimal,parse_int=Decimal)
            if isinstance(interval,list) and len(interval)==2 and all(v is None or isinstance(v,(Decimal,int,float)) for v in interval):
                return "["+" ~ ".join(won(v*scale if v is not None else None) for v in interval)+"]"
        except (ValueError,TypeError): pass
    return value


def archive():
    data=json.loads(files("oibc_portable").joinpath("research_history.json").read_text(encoding="utf-8"))
    for table in data["tables"]:
        table["rows"]=[[archived_cell(value,table["headers"][i] if i<len(table["headers"]) else "") for i,value in enumerate(row)] for row in table["rows"]]
        table["headers"]=[re.sub("백만원|만원|천원","원",h) if monetary(h) else h for h in table["headers"]]
    for row in data["known_settlement"]:
        row["display"]={key:won(row[key]) for key in ("R","W","S")}
    scores=json.loads(files("oibc_portable").joinpath("historical_scores.json").read_text(encoding="utf-8"))
    data["rescored_results"]=scores["results"]
    for row in data["rescored_results"]: row["display"]={key:won(row["settlement"][key]) for key in ("R","W","S")}
    return data


def bundled_results(context):
    data=json.loads(files("oibc_portable").joinpath("historical_scores.json").read_text(encoding="utf-8"))
    original=data["source_context"]
    if any(context.get(key)!=original.get(key) for key in ("version","profile","test_start","test_end","source_hashes","common_hours")):
        return []
    rows=data["results"]
    for row in rows: row.update(bundled=True,id="history-"+row["id"])
    return rows
