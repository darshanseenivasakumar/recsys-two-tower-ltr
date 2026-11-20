"""Latency by layer from the API's JSON logs: service steps, middleware, steady-state only."""

import json
import sys

import numpy as np
import pandas as pd

rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    try:
        rows.append(json.loads(line))
    except ValueError:
        pass
df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["ts"])
acc = df[(df.logger == "recsys.access") & (df.path == "/recommend")]
# steady state: 10 s buckets with at least 80% of the peak rate, minus the edges
per = acc.set_index("ts").resample("10s").size()
busy = per[per >= 0.8 * per.max()]
lo, hi = busy.index.min() + pd.Timedelta("10s"), busy.index.max()
acc = acc[(acc.ts >= lo) & (acc.ts < hi)]
rec = df[(df.msg == "recommend") & (df.ts >= lo) & (df.ts < hi)]
t = pd.json_normalize(rec["timings_ms"].tolist())


def pct(s: pd.Series) -> str:
    return "p50 %5.1f  p95 %5.1f  p99 %5.1f" % tuple(np.percentile(s.dropna(), [50, 95, 99]))


print(f"steady window {lo:%H:%M:%S}-{hi:%H:%M:%S}: {len(acc) / (hi - lo).total_seconds():.0f} req/s")
print("middleware (handler + threadpool wait)", pct(acc.duration_ms))
for c in ["total", "store", "retrieve_and_features", "rank"]:
    print(f"service {c:22s}        ", pct(t[c]))
personal = rec[rec.strategy == "personalized"]
tp = pd.json_normalize(personal["timings_ms"].tolist())
print(f"personalized only ({len(personal) / len(rec):.0%}): total", pct(tp["total"]))
m = acc.set_index("request_id").duration_ms.rename("mw").to_frame()
m = m.join(rec.set_index("request_id").timings_ms.apply(lambda x: x["total"]).rename("svc"))
m["outside"] = m.mw - m.svc
slow = m[m.mw >= m.mw.quantile(0.95)]
print(
    "slowest 5%%: median %.0f ms in middleware = %.0f in the service + %.0f waiting outside it"
    % (slow.mw.median(), slow.svc.median(), slow.outside.median())
)
