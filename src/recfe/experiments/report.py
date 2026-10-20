"""Collect result JSONs into markdown tables (reports/tables.md) and figures (reports/figures)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from recfe.paths import FIGURES, PROCESSED, REPORTS, RESULTS  # noqa: E402

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "lines.linewidth": 2, "legend.frameon": False,
})

METRICS = {"yambda": ["logloss", "auc", "gauc", "ndcg@10"], "kion": ["logloss", "auc", "ndcg@10", "recall@10", "map@10"]}
ROWS = [
    ("base", "Только статические признаки (+ скоры кандидатов)"),
    ("manual", "**Бустинг на ручных признаках (baseline)**"),
    ("grid", "Сетка: все ключи × события × окна {1,7,30,all}"),
    ("random", "Случайные K признаков грамматики (3 сида, среднее)"),
    ("autofe", "**AutoFE** (поиск с нуля)"),
    ("manual_autofe", "Ручные + AutoFE-добавки"),
    ("gnn", "**Графовая модель** (GNN, полная)"),
    ("gnn_emb", "GNN, только эмбеддинги"),
    ("base+lgcn", "Статические + LightGCN-скоры"),
    ("manual+lgcn", "Ручные + LightGCN-скоры"),
    ("autofe+lgcn", "AutoFE + LightGCN-скоры"),
    ("blend_manual_gnn", "Бленд: бустинг(ручные) ⊕ GNN"),
    ("blend_autofe_gnn", "Бленд: бустинг(AutoFE) ⊕ GNN"),
]


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def collect(ds):
    out = {}
    for tag in ("base", "manual", "grid", "autofe", "manual_autofe", "gnn", "gnn_emb", "base+lgcn",
                "manual+lgcn", "autofe+lgcn"):
        r = load(f"{ds}__{tag}.json")
        if r and "test" in r:
            out[tag] = r
    rnd = load(f"{ds}__random_summary.json")
    if rnd:
        runs = rnd["runs"]
        out["random"] = {s: {m: float(np.mean([r[s][m] for r in runs])) for m in runs[0][s] if m != "groups"}
                         for s in ("val", "test")}
        n_static = out["base"]["n_features"] if "base" in out else 0
        out["random"]["n_features"] = rnd["k"] + n_static
    hyb = load(f"{ds}__hybrid_summary.json") or {}
    for k, v in hyb.items():
        if k.startswith("blend"):
            out[k] = v
    ref = (load(f"{ds}__baseline_summary.json") or {}).get("reference", {})
    return out, ref


def fmt(x, m):
    if x is None:
        return "–"
    return f"{x:.4f}"


def table(ds):
    res, ref = collect(ds)
    ms = METRICS[ds]
    head = "| Модель | " + " | ".join(m.upper() if m != "logloss" else "LogLoss" for m in ms) + " | признаков |"
    lines = [head, "|" + "---|" * (len(ms) + 2)]
    for k, v in ref.items():
        lines.append(f"| _эталон: {k}_ | " + " | ".join(fmt(v["test"].get(m), m) for m in ms) + " | – |")
    for tag, label in ROWS:
        if tag not in res:
            continue
        r = res[tag]
        nf = r.get("n_features", "–")
        lines.append(f"| {label} | " + " | ".join(fmt(r["test"].get(m), m) for m in ms) + f" | {nf} |")
    lines.append("")
    lines.append("Валидация (по ней выбирались признаки и останавливалось обучение):")
    lines.append("")
    lines.append(head)
    lines.append("|" + "---|" * (len(ms) + 2))
    for tag, label in ROWS:
        if tag in res:
            r = res[tag]
            lines.append(f"| {label} | " + " | ".join(fmt(r["val"].get(m), m) for m in ms) + f" | {r.get('n_features', '–')} |")
    return "\n".join(lines)


def sig_table(ds):
    s = load(f"{ds}__significance.json")
    if not s:
        return ""
    labels = dict(ROWS)
    ms = list(next(iter(s.values())))
    lines = ["| Модель − бейзлайн | " + " | ".join(f"Δ {m} [95% ДИ]" for m in ms) + " |", "|" + "---|" * (len(ms) + 1)]
    for tag, v in s.items():
        cells = []
        for m in ms:
            d, (lo, hi) = v[m]["diff"], v[m]["ci95"]
            mark = "" if lo <= 0 <= hi else " ✱"
            nd = 5 if max(abs(d), abs(lo), abs(hi)) < 1e-3 else 4
            cells.append(f"{d:+.{nd}f} [{lo:+.{nd}f}; {hi:+.{nd}f}]{mark}")
        lines.append(f"| {labels.get(tag, tag).replace('**', '')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ablation_table(ds):
    s = load(f"{ds}__baseline_summary.json")
    if not s:
        return ""
    ms = METRICS[ds]
    full = s["manual"]["test"]
    names = {"user": "пользователь", "item": "объект", "ui": "user×item (прямые)", "a": "атрибут объекта",
             "ua": "user×атрибут (косвенные)", "s": "сегмент×объект"}
    lines = ["| Убрали группу | шт. | " + " | ".join(f"Δ {m}" for m in ms) + " |", "|" + "---|" * (len(ms) + 2)]
    for g, v in s["ablation"].items():
        lines.append(f"| {names.get(g, g)} | {v['n_dropped']} | " +
                     " | ".join(f"{v['test'][m] - full[m]:+.4f}" for m in ms) + " |")
    return "\n".join(lines)


def feature_cost(ds, names):
    """Amortised point-in-time computation time of a feature set over train+val+test (from the cache log)."""
    p = PROCESSED / ds / "fc" / "cost.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    return sum(c.get(f"{s}::{n}", 0.0) for n in names for s in ("train", "val", "test"))


def cost_table():
    lines = ["| Датасет | Подход | Признаки / графы | Обучение | Инференс test | Железо |",
             "|---|---|---|---|---|---|"]
    for ds in ("yambda", "kion"):
        for tag, label in (("manual", "бустинг, ручные"), ("autofe", "бустинг, AutoFE-набор"), ("grid", "бустинг, сетка")):
            m = load(f"{ds}__{tag}.json")
            if m:
                fc = feature_cost(ds, m["features"])
                lines.append(f"| {ds} | {label} ({m['n_features']} призн.) | {fc / 60:.1f} мин | {m['train_sec']:.0f} с | "
                             f"{m.get('test_predict_sec', 0):.1f} с | CPU, 16 потоков |")
        for tag, label in (("autofe", "поиск AutoFE с нуля"), ("manual_autofe", "поиск добавок к ручным")):
            a = load(f"{ds}__{tag}_search.json")
            if a:
                t = a["timing"]
                total = a["log"][-1]["elapsed_sec"] / 60
                lines.append(f"| {ds} | {label}: {len(a['log'])} поколений, {a['n_evaluated']} кандидатов | "
                             f"{t['features'] / 60:.0f} мин | скрининг {t['screen'] / 60:.0f} мин + переобучения "
                             f"{t['train'] / 60:.0f} мин (всего {total:.0f} мин) | – | CPU |")
        g = load(f"{ds}__gnn.json")
        g1 = load(f"{ds}__gnn_v1.json") if ds == "yambda" else g
        if g:
            build = (g1 or g)["timing"]["graph_build"]
            lines.append(f"| {ds} | GNN ({g['n_params'] / 1e6:.0f}M параметров) | {build / 60:.0f} мин (все отсечки) | "
                         f"{g['timing']['train'] / 60:.1f} мин | {g['timing']['eval'] / 60:.1f} мин (val+test) | "
                         f"RTX 3070 Laptop, пик {g['peak_gpu_gb']:.1f} ГБ |")
    return "\n".join(lines)


def selected_list(ds, tag="autofe", top=None):
    a = load(f"{ds}__{tag}_search.json")
    if not a:
        return ""
    names = a["selected_names"][a.get("n_start", 0):]
    return "\n".join(f"{i + 1}. `{n}`" for i, n in enumerate(names[:top] if top else names))


def fig_headline():
    """Difference from the hand-crafted baseline on test (main metric of each task)."""
    spec = {"yambda": [("auc", "ROC-AUC"), ("gauc", "GAUC")], "kion": [("map@10", "MAP@10"), ("auc", "ROC-AUC")]}
    skip = {"base", "manual", "gnn_emb", "base+lgcn"}  # far below the rest; they stay in the tables
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    for row, ds in enumerate(("yambda", "kion")):
        res, _ = collect(ds)
        if "manual" not in res:
            continue
        for col, (m, mname) in enumerate(spec[ds]):
            ax = axes[row, col]
            labels, vals = [], []
            for tag, label in ROWS:
                if tag in res and tag not in skip:
                    labels.append(label.replace("**", ""))
                    vals.append(res[tag]["test"][m] - res["manual"]["test"][m])
            y = np.arange(len(vals))
            ax.barh(y, vals, color=[BLUE if v >= 0 else RED for v in vals], height=0.66)
            ax.axvline(0, color=INK2, linewidth=1)
            span = max(abs(v) for v in vals)
            ax.set_xlim(min(vals) - span * 0.35, max(max(vals), 0) + span * 0.35)
            for yi, v in zip(y, vals):
                ax.text(v, yi, f" {v:+.4f} " if v >= 0 else f" {v:+.4f} ", va="center",
                        ha="left" if v >= 0 else "right", fontsize=8, color=INK)
            ax.set_yticks(y)
            ax.set_yticklabels(labels if col == 0 else [], fontsize=8)
            ax.invert_yaxis()
            base = res["manual"]["test"][m]
            ax.set_title(f"{ds.upper()}: Δ {mname} к бейзлайну ({base:.4f})", fontsize=10)
            ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "headline.png", dpi=150)
    plt.close(fig)


def fig_autofe():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, ds in zip(axes, ("yambda", "kion")):
        a, m = load(f"{ds}__autofe_search.json"), load(f"{ds}__manual.json")
        if not a:
            continue
        x = [r["elapsed_sec"] / 60 for r in a["log"]]
        y = [r["val_logloss"] for r in a["log"]]
        ax.plot(x, y, color=BLUE, marker="o", markersize=5, label="AutoFE")
        if m:
            ax.axhline(m["val_logloss"], color=ORANGE, linewidth=2, linestyle="--", label="ручные признаки")
        b = load(f"{ds}__manual_autofe_search.json")
        if b:
            ax.plot([r["elapsed_sec"] / 60 for r in b["log"]], [r["val_logloss"] for r in b["log"]],
                    color=AQUA, marker="s", markersize=5, label="ручные + AutoFE")
        ax.set_title(ds.upper())
        ax.set_xlabel("время поиска, мин")
        ax.set_ylabel("val LogLoss")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "autofe_trajectory.png", dpi=150)
    plt.close(fig)


WINDOW_LABELS = {
    "yambda": ["прослушивания трека\nпользователем (user×item)", "прослушивания артиста\nпользователем (user×artist)",
               "популярность трека", "доля дослушанных\nрекомендаций пользователя",
               "доля пропусков артиста\nу пользователя"],
    "kion": ["популярность объекта", "просмотры жанров\nобъекта пользователем (max)", "популярность в\nвозрастной группе",
             "активность пользователя", "доля досмотров объекта"],
}


def fig_windows():
    for ds in ("yambda", "kion"):
        w = load(f"{ds}__window_sweep.json")
        if not w:
            continue
        n = len(w)
        fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0), sharey=False)
        for k, (ax, (name, v)) in enumerate(zip(np.atleast_1d(axes), w.items())):
            name = WINDOW_LABELS[ds][k]
            xs = [x for x in v["windows"] if x != "all"]
            ys = [g * 1000 for g, x in zip(v["in_set"], v["windows"]) if x != "all"]
            ax.plot(xs, ys, color=BLUE, marker="o", markersize=4)
            ax.axhline(v["in_set"][-1] * 1000, color=INK2, linewidth=1, linestyle=":")
            for hw in (1, 7, 30):
                ax.axvline(hw, color=ORANGE, linewidth=1, alpha=0.6)
            i = int(np.argmax(ys))
            ax.annotate(f"max: {xs[i]} дн.", (xs[i], ys[i]), textcoords="offset points", xytext=(0, -12),
                        ha="center", fontsize=8, color=INK)
            ax.set_xscale("log")
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("окно, дней")
        np.atleast_1d(axes)[0].set_ylabel("выигрыш val LogLoss ×10³")
        fig.tight_layout()
        fig.savefig(FIGURES / f"window_sweep_{ds}.png", dpi=150)
        plt.close(fig)


def fig_mix():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ds in zip(axes, ("yambda", "kion")):
        g = load(f"{ds}__gnn.json")
        if not g:
            continue
        names, vals = list(g["mix"]), list(g["mix"].values())
        ax.barh(range(len(vals)), vals, color=BLUE, height=0.7)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{ds.upper()}: веса отношений в графе (softmax)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "gnn_relation_mix.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    parts = []
    for ds in ("yambda", "kion"):
        parts += [f"## {ds}", "", table(ds), "",
                  "Парный бутстрэп по группам (1000 повторов), test; ✱ — интервал не содержит 0:", "", sig_table(ds), "",
                  "Абляция ручных признаков (test, изменение при удалении группы):", "", ablation_table(ds), ""]
    parts += ["## Затраты", "", cost_table(), ""]
    for ds in ("yambda", "kion"):
        parts += [f"## Найденные признаки: {ds}", "", "AutoFE с нуля:", "", selected_list(ds), "",
                  "Добавки к ручным:", "", selected_list(ds, "manual_autofe"), ""]
    (REPORTS / "tables.md").write_text("\n".join(parts), encoding="utf-8")
    fig_headline()
    fig_autofe()
    fig_windows()
    fig_mix()
    print((REPORTS / "tables.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
