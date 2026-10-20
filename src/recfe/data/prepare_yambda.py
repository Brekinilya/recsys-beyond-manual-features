"""Yambda-50M -> unified event table.

Output (data/processed/yambda):
  events.parquet  user_id, item_id, ts (seconds), ev (0 listen, 1 like, 2 dislike, 3 unlike, 4 undislike),
                  org (1 organic / 0 recommendation), played (played_ratio_pct, null for non-listens),
                  artist, album (dense ids, -1 when unknown); sorted by ts
  items.parquet   item_id, orig_item_id, artist, album, length (median track length, seconds)

Note: despite the "5s units" wording in the dataset card, `timestamp` is in seconds
rounded to 5-second bins: with 1 s units the log spans 300.9 days and shows a clear
daily cycle, with 5 s units it would be 1504 "days" with a 5-day periodicity.
"""
import polars as pl

from recfe.paths import RAW, processed

EVENT_CODES = {"listen": 0, "like": 1, "dislike": 2, "unlike": 3, "undislike": 4}


def main():
    raw = RAW / "yambda"
    out = processed("yambda")
    ev = pl.read_parquet(raw / "multi_event.parquet")
    users = ev.select(pl.col("uid").unique().sort()).with_row_index("user_id")
    items = ev.select(pl.col("item_id").unique().sort()).rename({"item_id": "orig_item_id"}).with_row_index("item_id")

    # tracks can have 2 artists and many albums (compilations); keep the smallest id as the primary one
    art = pl.read_parquet(raw / "artist_item_mapping.parquet").group_by("item_id").agg(pl.col("artist_id").min())
    alb = pl.read_parquet(raw / "album_item_mapping.parquet").group_by("item_id").agg(pl.col("album_id").min())
    items = (items.join(art.rename({"item_id": "orig_item_id"}), on="orig_item_id", how="left")
                  .join(alb.rename({"item_id": "orig_item_id"}), on="orig_item_id", how="left"))
    for col, new in [("artist_id", "artist"), ("album_id", "album")]:
        dense = items.select(pl.col(col).drop_nulls().unique().sort()).with_row_index(new)
        items = items.join(dense, on=col, how="left").drop(col)
    length = (ev.filter(pl.col("event_type") == "listen")
                .group_by("item_id").agg(pl.col("track_length_seconds").median().alias("length"))
                .rename({"item_id": "orig_item_id"}))
    items = items.join(length, on="orig_item_id", how="left")
    items = items.with_columns(
        pl.col("item_id").cast(pl.Int32),
        pl.col("artist").cast(pl.Int32).fill_null(-1),
        pl.col("album").cast(pl.Int32).fill_null(-1),
        pl.col("length").cast(pl.Float32),
    ).sort("item_id")

    ev = (ev.join(users, on="uid")
            .join(items.select("orig_item_id", "item_id", "artist", "album").rename({"item_id": "iid"}),
                  left_on="item_id", right_on="orig_item_id")
            .select(
                pl.col("user_id").cast(pl.Int32),
                pl.col("iid").alias("item_id"),
                pl.col("timestamp").cast(pl.Int32).alias("ts"),
                pl.col("event_type").cast(pl.String).replace_strict(EVENT_CODES, return_dtype=pl.UInt8).alias("ev"),
                pl.col("is_organic").cast(pl.UInt8).alias("org"),
                pl.col("played_ratio_pct").cast(pl.UInt16).alias("played"),
                "artist", "album",
            )
            .sort("ts", maintain_order=True))
    ev.write_parquet(out / "events.parquet")
    items.write_parquet(out / "items.parquet")

    day = (pl.col("ts") // 86400).alias("day")
    rec = ev.filter((pl.col("ev") == 0) & (pl.col("org") == 0)).with_columns(day)
    print(ev.shape, items.shape, "days:", ev["ts"].max() / 86400)
    print(rec.group_by("day").agg(pl.len().alias("rec_listens"), pl.col("user_id").n_unique().alias("users"))
             .sort("day").tail(16))


if __name__ == "__main__":
    main()
