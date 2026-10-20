"""Yambda task: response to a recommendation.

An "impression" is a recommendation-driven listen (is_organic = 0): the recommender put the
track into the user's stream. Label = the user listened to more than 50% of it (Listen+),
otherwise the track was skipped. This is the music analogue of impression -> click.

Rows are the first impression of each (user, item) per day. The prediction moment is the
start of that day: features see only events before it, so neither the target listen nor
anything later the same day is visible. Splits are consecutive days (global temporal split):

    train: days 289-294   val: days 295-296   test: days 297-300

Ranking metrics are computed inside (user, day) groups: how well the day's recommendation
stream would have been re-ordered.
"""
import polars as pl

from recfe.datasets import DAY, load_yambda
from recfe.paths import processed

SPLITS = {"train": (289, 295), "val": (295, 297), "test": (297, 301)}


def build_rows() -> pl.DataFrame:
    ds = load_yambda()
    lo = min(a for a, _ in SPLITS.values())
    hi = max(b for _, b in SPLITS.values())
    imp = ds.events.filter((pl.col("ts") >= lo * DAY) & (pl.col("ts") < hi * DAY)
                           & (pl.col("ev") == 0) & (pl.col("org") == 0))
    imp = imp.with_columns((pl.col("ts") // DAY).cast(pl.Int32).alias("day"))
    imp = imp.unique(subset=["user_id", "item_id", "day"], keep="first", maintain_order=True)
    split = pl.lit(None, dtype=pl.String)
    for name, (a, b) in SPLITS.items():
        split = pl.when((pl.col("day") >= a) & (pl.col("day") < b)).then(pl.lit(name)).otherwise(split)
    rows = imp.select(
        "user_id", "item_id", "day",
        (pl.col("day") * DAY).cast(pl.Int32).alias("cutoff"),
        pl.col("ts").alias("event_ts"),
        (pl.col("played") >= 50).cast(pl.Int8).alias("label"),
        (pl.col("user_id").cast(pl.Int64) * 1000 + pl.col("day")).alias("group"),
        split.alias("split"),
    ).sort("event_ts", "user_id", "item_id").with_row_index("row_id").with_columns(pl.col("row_id").cast(pl.Int64))
    # group-level positive count (all positives of a group are among its rows here)
    rows = rows.with_columns(pl.col("label").sum().over("group").cast(pl.Int32).alias("npos"))
    rows.write_parquet(processed("yambda") / "rows.parquet")
    print(rows.group_by("split").agg(pl.len(), pl.col("label").mean().alias("pos_rate"),
                                     pl.col("group").n_unique().alias("groups"),
                                     pl.col("user_id").n_unique().alias("users")).sort("split"))
    return rows


if __name__ == "__main__":
    build_rows()
