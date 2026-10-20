"""Print the selected features, the best screened candidates and the log of a feature search.

python scripts/inspect_search.py reports/results/kion__autofe_search.json [top_n]
"""
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("selected:")
for n in d["selected_names"]:
    print("  ", n)
print("top screened candidates:")
for a in d["archive"][: int(sys.argv[2]) if len(sys.argv) > 2 else 25]:
    print(f"  {a['gain']:.6f} g{a['gen']} {a['name']}")
print("log:")
for r in d["log"]:
    print("  ", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != "accepted"})
print("timing", {k: round(v) for k, v in d["timing"].items()})
