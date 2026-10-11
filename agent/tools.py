"""
Analysis tools for the data agent.

Rules every tool follows:
  * No column name is hard-coded: the agent reads the schema first and passes
    column names as ARGUMENTS, so any tabular dataset works.
  * The LLM never computes numbers; every figure comes from pandas/scipy/sklearn.
  * Bad input raises ToolError with a helpful message; the agent loop feeds it
    back to the model so it can correct itself.

Tools (all genuinely different kinds of analysis):
  1. describe_dataset     - schema / quality profile
  2. aggregate_data       - filter + group-by + aggregate + rank (slice & dice)
  3. correlation_analysis - numeric relationships (Pearson/Spearman + p-values)
  4. compare_groups       - hypothesis tests (t-test/Mann-Whitney, ANOVA/Kruskal, chi-square)
  5. detect_outliers      - IQR / z-score / Isolation Forest
  6. time_trend           - resample, rolling mean, growth, trend significance, forecast
  7. plot_chart           - renders a chart to PNG
  + memory tools: save_memory, list_memories, forget_memory
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from scipy import stats  # noqa: E402

from .data import clean, column_kind, table  # noqa: E402
from .memory import MemoryStore, dataset_fingerprint  # noqa: E402


class ToolError(Exception):
    """Raised for expected, recoverable problems (bad column, bad filter...)."""


@dataclass
class Context:
    df: pd.DataFrame
    memory: MemoryStore
    outdir: Path = Path("outputs")
    figures: list = field(default_factory=list)

    def __post_init__(self):
        self.outdir = Path(self.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    @property
    def scope(self) -> str:
        return dataset_fingerprint(self.df.columns)


# ----------------------------------------------------------------- helpers
_FORBIDDEN = re.compile(r"__|import|exec|eval|open\(|os\.|sys\.|subprocess|getattr|globals|lambda|;", re.I)
_AGGS = {"sum", "mean", "median", "min", "max", "count", "nunique", "std", "var", "first", "last"}
_FREQ = {"D": "D", "W": "W", "M": "ME", "Q": "QE", "Y": "YE"}


def _check_cols(ctx: Context, cols):
    cols = [cols] if isinstance(cols, str) else list(cols)
    missing = [c for c in cols if c not in ctx.df.columns]
    if missing:
        raise ToolError(f"Unknown column(s) {missing}. Available columns: {list(ctx.df.columns)}")
    return cols


def _numeric(ctx: Context, df: pd.DataFrame, col: str) -> pd.Series:
    _check_cols(ctx, col)
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        raise ToolError(f"Column '{col}' is not numeric (dtype {df[col].dtype}).")
    return s


def _safe_query(df: pd.DataFrame, expr: str) -> pd.DataFrame:
    if _FORBIDDEN.search(expr) or len(expr) > 300:
        raise ToolError("Filter rejected: only simple pandas query expressions are allowed "
                        "(comparisons, and/or/not, in, str.contains).")
    try:
        return df.query(expr)
    except Exception as e:  # noqa: BLE001
        raise ToolError(
            f"Invalid filter '{expr}': {e}. Use pandas query syntax; wrap column names containing "
            f"spaces in backticks, e.g. `Unit Price` > 5 and region == 'North'."
        )


def _apply_filter(ctx: Context, where: str | None, filter_name: str | None):
    """Apply a saved named filter (from memory) and/or an ad-hoc `where` expression."""
    df, applied = ctx.df, []
    if filter_name:
        f = ctx.memory.get_filter(filter_name, ctx.scope)
        if not f:
            names = [m["key"] for m in ctx.memory.list(ctx.scope) if m["kind"] == "filter"]
            raise ToolError(f"No saved filter named '{filter_name}'. Saved filters: {names}")
        df = _safe_query(df, f["expression"])
        applied.append(f"{f['key']}: {f['expression']}")
    if where:
        df = _safe_query(df, where)
        applied.append(where)
    return df, applied


# --------------------------------------------------------------- the tools
def make_tools(ctx: Context):
    """Build all tools bound to one dataset/memory context."""

    @tool
    def describe_dataset(columns: list[str] | None = None) -> dict:
        """Profile the dataset: row/column counts, per-column type, missing values, unique counts,
        summary statistics (numeric), top values (categorical), date range (datetime), duplicates,
        and 5 sample rows. ALWAYS call this first on a new dataset. Optionally pass `columns`
        to profile only some columns."""
        df = ctx.df
        cols = _check_cols(ctx, columns) if columns else list(df.columns)
        profile = {}
        for c in cols:
            s = df[c]
            kind = column_kind(s)
            info = {"dtype": str(s.dtype), "kind": kind,
                    "missing": int(s.isna().sum()), "missing_pct": round(100 * s.isna().mean(), 2),
                    "unique": int(s.nunique(dropna=True))}
            if kind.startswith("numeric") or kind == "boolean":
                d = pd.to_numeric(s, errors="coerce").describe()
                info.update({k: d.get(k) for k in ("min", "mean", "50%", "max", "std")})
            elif kind == "datetime":
                info.update({"min": s.min(), "max": s.max()})
            else:
                info["top_values"] = s.value_counts().head(5).to_dict()
            profile[c] = info
        return clean({
            "n_rows": len(df), "n_columns": df.shape[1],
            "duplicate_rows": int(df.duplicated().sum()),
            "columns": profile,
            "sample_rows": df.head(5).astype(object).where(df.head(5).notna(), None).to_dict("records"),
        })

    @tool
    def aggregate_data(metrics: list[str], group_by: list[str] | None = None,
                       where: str | None = None, filter_name: str | None = None,
                       sort_by: str | None = None, ascending: bool = False,
                       top_n: int | None = None) -> dict:
        """Filter, group, aggregate and rank rows. Use for totals, averages, counts, 'top 5 X by Y', breakdowns.
        - metrics: list of "column:aggregation" strings, aggregation in sum, mean, median, min, max,
          count, nunique, std. Use "*:count" to count rows. Example: ["revenue:sum", "*:count"].
        - group_by: columns to group by (omit for one overall result).
        - where: optional pandas query string, e.g. "region == 'North' and `Unit Price` > 10".
        - filter_name: name of a SAVED filter from memory (e.g. "active_customers"); combined with `where` by AND.
        - sort_by: result column to sort by (default: first metric). ascending: sort direction.
        - top_n: keep only first N rows after sorting."""
        df, applied = _apply_filter(ctx, where, filter_name)
        if df.empty:
            return {"rows_after_filter": 0, "filters_applied": applied, "result": None,
                    "note": "No rows match the filter."}
        specs = []
        for m in metrics:
            if ":" not in m:
                raise ToolError(f"Metric '{m}' must look like 'column:aggregation' (e.g. 'sales:sum').")
            col, agg = m.rsplit(":", 1)
            agg = agg.strip().lower()
            if agg not in _AGGS:
                raise ToolError(f"Unknown aggregation '{agg}'. Use one of {sorted(_AGGS)}.")
            if col != "*":
                _check_cols(ctx, col)
                if agg in ("sum", "mean", "median", "std", "var"):
                    df = df.assign(**{col: pd.to_numeric(df[col], errors="coerce")})
            specs.append((col.strip(), agg))
        if group_by:
            _check_cols(ctx, group_by)
        if top_n is None and group_by:
            pref = ctx.memory.get_preference("default_top_n")
            if pref and pref.strip().isdigit():
                top_n = int(pref)

        def run(g):
            out = {}
            for col, agg in specs:
                name = f"{col}_{agg}" if col != "*" else "row_count"
                out[name] = len(g) if col == "*" else getattr(g[col], agg)()
            return pd.Series(out)

        if group_by:
            res = df.groupby(group_by, dropna=False).apply(run, include_groups=False).reset_index()
        else:
            res = run(df).to_frame().T
        sort_col = sort_by if sort_by in res.columns else (res.columns[len(group_by or [])])
        res = res.sort_values(sort_col, ascending=ascending)
        if top_n:
            res = res.head(int(top_n))
        return clean({"rows_after_filter": len(df), "filters_applied": applied,
                      "sorted_by": sort_col, "result": table(res)})

    @tool
    def correlation_analysis(target: str | None = None, columns: list[str] | None = None,
                             method: str = "pearson", where: str | None = None,
                             filter_name: str | None = None) -> dict:
        """Measure relationships between NUMERIC columns with p-values.
        - target: if given, rank every other numeric column by correlation with it.
        - columns: restrict to these columns (default: all numeric columns).
        - method: "pearson" (linear) or "spearman" (rank/monotonic).
        - where / filter_name: optional filters as in aggregate_data.
        Without `target`, returns the strongest column pairs."""
        df, applied = _apply_filter(ctx, where, filter_name)
        method = method.lower()
        if method not in ("pearson", "spearman"):
            raise ToolError("method must be 'pearson' or 'spearman'.")
        num = df.select_dtypes(include="number")
        if columns:
            _check_cols(ctx, columns)
            num = df[columns].apply(pd.to_numeric, errors="coerce")
        if target:
            _check_cols(ctx, target)
            num = num.assign(**{target: pd.to_numeric(df[target], errors="coerce")})
        if num.shape[1] < 2:
            raise ToolError(f"Need at least 2 numeric columns; found {list(num.columns)}.")
        fn = stats.pearsonr if method == "pearson" else stats.spearmanr

        def pair(a, b):
            x = num[[a, b]].dropna()
            if len(x) < 3 or x[a].nunique() < 2 or x[b].nunique() < 2:
                return None
            r, p = fn(x[a], x[b])
            return {"col_a": a, "col_b": b, "r": r, "p_value": p, "n": len(x),
                    "strength": ("strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"),
                    "significant_at_0.05": bool(p < 0.05)}

        rows = []
        if target:
            for c in num.columns:
                if c != target and (r := pair(target, c)):
                    rows.append(r)
        else:
            cols = list(num.columns)
            for i, a in enumerate(cols):
                for b in cols[i + 1:]:
                    if r := pair(a, b):
                        rows.append(r)
        rows.sort(key=lambda r: -abs(r["r"]))
        return clean({"method": method, "filters_applied": applied, "n_pairs": len(rows),
                      "results": rows[:15],
                      "caveat": "Correlation does not imply causation."})

    @tool
    def compare_groups(value_column: str, group_column: str, where: str | None = None,
                       filter_name: str | None = None) -> dict:
        """Test whether groups differ, using the right statistical test automatically.
        - value_column NUMERIC: per-group stats, then 2 groups -> Welch t-test + Mann-Whitney U + Cohen's d;
          3+ groups -> one-way ANOVA + Kruskal-Wallis + eta-squared.
        - value_column CATEGORICAL (e.g. survived yes/no): proportions per group + chi-square test + Cramer's V.
        - group_column: the column that defines the groups (<= 20 distinct values).
        Use for 'is A different from B?', 'does X vary by Y?'."""
        df, applied = _apply_filter(ctx, where, filter_name)
        _check_cols(ctx, [value_column, group_column])
        d = df[[value_column, group_column]].dropna()
        if d[group_column].nunique() < 2:
            raise ToolError(f"'{group_column}' has fewer than 2 groups after filtering.")
        if d[group_column].nunique() > 20:
            raise ToolError(f"'{group_column}' has {d[group_column].nunique()} groups; pick a lower-cardinality column.")

        if column_kind(d[value_column]).startswith("numeric") and d[value_column].nunique() > 2:
            v = pd.to_numeric(d[value_column], errors="coerce")
            d = d.assign(**{value_column: v}).dropna()
            summ = d.groupby(group_column)[value_column].agg(["count", "mean", "median", "std"]).reset_index()
            groups = [g[value_column].values for _, g in d.groupby(group_column) if len(g) >= 2]
            if len(groups) < 2:
                raise ToolError("Not enough observations per group (need >= 2).")
            out = {"type": "numeric_by_group", "group_summary": table(summ), "filters_applied": applied}
            if len(groups) == 2:
                t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                u, pu = stats.mannwhitneyu(groups[0], groups[1])
                sp = np.sqrt(((len(groups[0]) - 1) * groups[0].var(ddof=1) + (len(groups[1]) - 1) * groups[1].var(ddof=1))
                             / (len(groups[0]) + len(groups[1]) - 2))
                dcoh = (groups[0].mean() - groups[1].mean()) / sp if sp > 0 else 0
                out["tests"] = {"welch_t": {"statistic": t, "p_value": p},
                                "mann_whitney_u": {"statistic": u, "p_value": pu},
                                "cohens_d": dcoh}
                out["significant_at_0.05"] = bool(p < 0.05)
            else:
                f, p = stats.f_oneway(*groups)
                h, ph = stats.kruskal(*groups)
                allv = np.concatenate(groups)
                ssb = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
                sst = ((allv - allv.mean()) ** 2).sum()
                out["tests"] = {"anova_f": {"statistic": f, "p_value": p},
                                "kruskal_wallis": {"statistic": h, "p_value": ph},
                                "eta_squared": ssb / sst if sst else 0}
                out["significant_at_0.05"] = bool(p < 0.05)
            return clean(out)

        ct = pd.crosstab(d[group_column], d[value_column])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        cramers = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if min(ct.shape) > 1 else 0
        props = ct.div(ct.sum(axis=1), axis=0).round(4)
        return clean({"type": "categorical_by_group", "filters_applied": applied,
                      "counts": table(ct.reset_index()), "row_proportions": table(props.reset_index()),
                      "tests": {"chi_square": {"statistic": chi2, "dof": dof, "p_value": p}, "cramers_v": cramers},
                      "significant_at_0.05": bool(p < 0.05)})

    @tool
    def detect_outliers(columns: list[str], method: str = "iqr", where: str | None = None,
                        filter_name: str | None = None) -> dict:
        """Find unusual values in NUMERIC columns.
        - columns: one or more numeric columns.
        - method: "iqr" (1.5*IQR fences, per column), "zscore" (|z|>3, per column), or
          "isolation_forest" (multivariate anomalies across all given columns jointly).
        Returns counts, bounds and the most extreme example rows."""
        df, applied = _apply_filter(ctx, where, filter_name)
        _check_cols(ctx, columns)
        method = method.lower()
        if method not in ("iqr", "zscore", "isolation_forest"):
            raise ToolError("method must be iqr, zscore or isolation_forest.")
        num = pd.concat({c: _numeric(ctx, df, c) for c in columns}, axis=1)
        out: dict = {"method": method, "filters_applied": applied, "rows_checked": int(len(num))}

        if method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            x = num.dropna()
            if len(x) < 20:
                raise ToolError("Need at least 20 complete rows for isolation_forest.")
            model = IsolationForest(contamination="auto", random_state=0).fit(x)
            score = model.score_samples(x)
            flag = model.predict(x) == -1
            worst = x.assign(anomaly_score=score)[flag].sort_values("anomaly_score").head(10)
            out.update({"n_outliers": int(flag.sum()), "pct_outliers": round(100 * flag.mean(), 2),
                        "most_anomalous_rows": table(worst.reset_index().rename(columns={"index": "row_index"}))})
            return clean(out)

        per = {}
        for c in columns:
            s = num[c].dropna()
            if method == "iqr":
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            else:
                lo, hi = s.mean() - 3 * s.std(), s.mean() + 3 * s.std()
            mask = (s < lo) | (s > hi)
            ex = s[mask].reindex(s[mask].sub(s.median()).abs().sort_values(ascending=False).index).head(8)
            per[c] = {"lower_bound": lo, "upper_bound": hi, "n_outliers": int(mask.sum()),
                      "pct_outliers": round(100 * mask.mean(), 2),
                      "extreme_values": [{"row_index": int(i), "value": v} for i, v in ex.items()]}
        out["per_column"] = per
        return clean(out)

    @tool
    def time_trend(date_column: str, value_column: str, agg: str = "sum", freq: str | None = None,
                   where: str | None = None, filter_name: str | None = None,
                   forecast_periods: int = 3) -> dict:
        """Analyse a metric over time: resample, rolling average, period-over-period growth, peak/trough,
        linear trend with significance, and a simple linear-extrapolation forecast.
        - date_column: a date/datetime column. value_column: numeric column to track (or "*" to count rows).
        - agg: sum, mean, median, min, max, count.
        - freq: D (daily), W (weekly), M (monthly), Q (quarterly), Y (yearly). If omitted, the user's
          remembered preference 'time_granularity' is used, else M.
        - forecast_periods: how many future periods to extrapolate (0 to skip)."""
        df, applied = _apply_filter(ctx, where, filter_name)
        _check_cols(ctx, date_column)
        used_pref = False
        if freq is None:
            pref = ctx.memory.get_preference("time_granularity")
            if pref:
                freq, used_pref = pref.strip()[0].upper(), True
            else:
                freq = "M"
        freq = freq.strip()[0].upper()
        if freq not in _FREQ:
            raise ToolError("freq must be one of D, W, M, Q, Y.")
        dates = pd.to_datetime(df[date_column], errors="coerce")
        if dates.notna().sum() == 0:
            raise ToolError(f"'{date_column}' could not be parsed as dates.")
        if agg not in ("sum", "mean", "median", "min", "max", "count"):
            raise ToolError("agg must be sum, mean, median, min, max or count.")
        if value_column == "*":
            ser = pd.Series(1, index=dates[dates.notna()])
            agg = "sum"
        else:
            vals = _numeric(ctx, df, value_column)
            ser = pd.Series(vals.values, index=dates).dropna()
            ser = ser[ser.index.notna()]
        ser = ser.sort_index().resample(_FREQ[freq]).agg(agg)
        if len(ser) < 3:
            raise ToolError(f"Only {len(ser)} period(s) at freq={freq}; try a finer frequency.")
        roll = ser.rolling(min(3, len(ser))).mean()
        growth = ser.pct_change().replace([np.inf, -np.inf], np.nan) * 100
        x = np.arange(len(ser))
        lr = stats.linregress(x, ser.values)
        out = {
            "freq": freq, "frequency_came_from_saved_preference": used_pref, "aggregation": agg,
            "filters_applied": applied, "n_periods": len(ser),
            "start": ser.index.min(), "end": ser.index.max(),
            "peak": {"period": ser.idxmax(), "value": ser.max()},
            "trough": {"period": ser.idxmin(), "value": ser.min()},
            "overall_change_pct": (100 * (ser.iloc[-1] - ser.iloc[0]) / ser.iloc[0]) if ser.iloc[0] else None,
            "trend": {"slope_per_period": lr.slope, "r_squared": lr.rvalue ** 2, "p_value": lr.pvalue,
                      "direction": "up" if lr.slope > 0 else "down",
                      "significant_at_0.05": bool(lr.pvalue < 0.05)},
            "series": table(pd.DataFrame({"period": ser.index.astype(str), "value": ser.values,
                                          "rolling_3": roll.values, "growth_pct": growth.values}), max_rows=40),
        }
        if forecast_periods and forecast_periods > 0:
            fut = [lr.intercept + lr.slope * (len(ser) + i) for i in range(int(forecast_periods))]
            out["forecast"] = {"method": "linear extrapolation of the fitted trend (naive; ignores seasonality)",
                               "values": fut}
        return clean(out)

    @tool
    def plot_chart(kind: str, x: str | None = None, y: str | None = None,
                   group_by: str | None = None, agg: str = "mean", where: str | None = None,
                   filter_name: str | None = None, title: str | None = None) -> dict:
        """Render a chart to a PNG file and return its path.
        - kind: "bar" (agg of y per x category), "line" (agg of y over x, x may be a date), "scatter" (y vs x),
          "hist" (distribution of x), "box" (distribution of y per x category).
        - x, y: column names (which ones are required depends on kind). group_by: optional series split
          (scatter/line). agg: sum, mean, median, count for bar/line. where / filter_name: optional filters."""
        df, applied = _apply_filter(ctx, where, filter_name)
        kind = kind.lower()
        if kind not in ("bar", "line", "scatter", "hist", "box"):
            raise ToolError("kind must be bar, line, scatter, hist or box.")
        for c in (x, y, group_by):
            if c:
                _check_cols(ctx, c)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if kind == "hist":
            if not x:
                raise ToolError("hist needs x.")
            ax.hist(_numeric(ctx, df, x).dropna(), bins=30)
            ax.set_xlabel(x), ax.set_ylabel("count")
        elif kind == "scatter":
            if not (x and y):
                raise ToolError("scatter needs x and y.")
            if group_by:
                for k, g in df.groupby(group_by):
                    ax.scatter(pd.to_numeric(g[x], errors="coerce"), pd.to_numeric(g[y], errors="coerce"), s=12, alpha=.7, label=str(k))
                ax.legend(title=group_by, fontsize=8)
            else:
                ax.scatter(pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce"), s=12, alpha=.7)
            ax.set_xlabel(x), ax.set_ylabel(y)
        elif kind == "box":
            if not (x and y):
                raise ToolError("box needs x (category) and y (numeric).")
            cats = list(df[x].dropna().unique())[:15]
            ax.boxplot([pd.to_numeric(df[df[x] == c][y], errors="coerce").dropna() for c in cats], tick_labels=[str(c) for c in cats])
            ax.set_xlabel(x), ax.set_ylabel(y)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        else:  # bar / line
            if not x:
                raise ToolError(f"{kind} needs x.")
            d = df.copy()
            if y:
                d[y] = pd.to_numeric(d[y], errors="coerce")
            if kind == "line" and pd.api.types.is_datetime64_any_dtype(d[x]):
                freq = _FREQ.get((ctx.memory.get_preference("time_granularity") or "M").strip()[0].upper(), "ME")
                d = d.set_index(x)
                if group_by:
                    piv = d.groupby([pd.Grouper(freq=freq), group_by])[y].agg(agg).unstack()
                    piv.plot(ax=ax)
                else:
                    (d[y].resample(freq).agg(agg) if y else d.resample(freq).size()).plot(ax=ax)
            else:
                keys = [x] + ([group_by] if group_by else [])
                g = (d.groupby(keys)[y].agg(agg) if y else d.groupby(keys).size()).unstack() if group_by else \
                    (d.groupby(x)[y].agg(agg) if y else d.groupby(x).size())
                g = g.sort_values(ascending=False).head(20) if not group_by else g.head(20)
                g.plot(kind="bar" if kind == "bar" else "line", ax=ax)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.set_ylabel(f"{agg}({y})" if y else "count")
        ax.set_title(title or f"{kind}: {y or ''} {'by' if y else ''} {x or ''}".strip())
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = ctx.outdir / f"chart_{int(time.time() * 1000)}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        ctx.figures.append(str(path))
        return {"saved_to": str(path), "filters_applied": applied,
                "note": "Chart rendered; describe what it shows using numbers from other tools."}

    # ---------------------------------------------------------- memory tools
    @tool
    def save_memory(kind: str, key: str, content: str, expression: str | None = None) -> dict:
        """Remember a durable, reusable fact so future answers improve. Call this when the user
        (a) defines a term/metric/column meaning  -> kind="definition"
        (b) defines a reusable group of rows       -> kind="filter"  (REQUIRES `expression`, a pandas query string)
        (c) states how they want answers           -> kind="preference" (key "time_granularity" with content
            "weekly"/"daily"/"monthly"; key "default_top_n" with a number; or any other style preference)
        (d) corrects something you said or assumed -> kind="correction"
        - key: short snake_case name, e.g. "active_customers". Re-saving the same key REPLACES the old fact.
        - content: ONE short sentence (<300 chars). NEVER store raw data, results, or chat transcripts.
        Filters are validated against the current dataset before saving."""
        kind = kind.lower()
        if kind == "filter":
            if not expression:
                raise ToolError("kind='filter' requires `expression` (pandas query string).")
            sub = _safe_query(ctx.df, expression)  # raises ToolError if invalid
            content = content or expression
            res = ctx.memory.save(kind, key, content, expression, scope=ctx.scope)
            if res.get("saved"):
                res["validated_rows_matching"] = len(sub)
            return clean(res)
        return clean(ctx.memory.save(kind, key, content, expression, scope=ctx.scope))

    @tool
    def list_memories() -> dict:
        """List everything currently remembered (for this dataset and global preferences)."""
        rows = ctx.memory.list(ctx.scope)
        return {"memories": [{k: r[k] for k in ("id", "kind", "key", "content", "expression")} for r in rows]}

    @tool
    def forget_memory(memory_id: int) -> dict:
        """Delete one remembered fact by its id (see list_memories). Only when the user asks to forget it."""
        return {"deleted": ctx.memory.forget(int(memory_id))}

    analysis = [describe_dataset, aggregate_data, correlation_analysis, compare_groups,
                detect_outliers, time_trend, plot_chart]
    return analysis + [save_memory, list_memories, forget_memory]
