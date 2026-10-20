"""KION -> unified event table.

Output (data/processed/kion):
  events.parquet     user_id, item_id, ts (seconds, day resolution), dur (total_dur, s), pct (watched_pct),
                     age, income, sex, kids (user segment codes, -1 unknown); sorted by ts
  items.parquet      item_id, orig_item_id, ctype, year, for_kids, age_rating
  item_attrs.parquet item_id, attr, value  (long table: genre / country / actor / director / studio / ctype)
  users.parquet      user_id, orig_user_id, age, income, sex, kids

Every (user, item) pair appears once in KION: `last_watch_dt` is the date of the last watch and
`total_dur` / `watched_pct` are totals. A pair whose last watch is before a cutoff therefore
carries no information from after the cutoff, so using these totals in features is safe.
"""
import datetime as dt

import polars as pl

from recfe.paths import RAW, processed

START = dt.date(2021, 3, 13)
MULTI = {"genres": "genre", "countries": "country", "actors": "actor", "directors": "director", "studios": "studio"}
MIN_ITEMS = {"genre": 1, "country": 5, "actor": 3, "director": 2, "studio": 1}


def code(df, col, name):
    """Replace a string column by a dense int code (-1 for null)."""
    m = df.select(pl.col(col).drop_nulls().unique().sort()).with_row_index(name + "_code")
    df = df.join(m, on=col, how="left").drop(col).rename({name + "_code": name})
    return df.with_columns(pl.col(name).cast(pl.Int32).fill_null(-1))


def main():
    src = RAW / "kion" / "extracted"
    out = processed("kion")
    inter = pl.read_csv(src / "interactions.csv", try_parse_dates=True)
    users_raw = pl.read_csv(src / "users.csv")
    items_raw = pl.read_csv(src / "items.csv")

    users = (pl.concat([inter.select("user_id"), users_raw.select("user_id")]).unique().sort("user_id")
               .rename({"user_id": "orig_user_id"}).with_row_index("user_id"))
    items = (pl.concat([inter.select("item_id"), items_raw.select("item_id")]).unique().sort("item_id")
               .rename({"item_id": "orig_item_id"}).with_row_index("item_id"))

    u = users.join(users_raw.rename({"user_id": "orig_user_id"}), on="orig_user_id", how="left")
    for col in ["age", "income", "sex"]:
        u = code(u, col, col)
    u = u.with_columns(pl.col("user_id").cast(pl.Int32), pl.col("kids_flg").cast(pl.Int32).fill_null(-1).alias("kids")).drop("kids_flg")
    u.write_parquet(out / "users.parquet")

    it = items.join(items_raw.rename({"item_id": "orig_item_id"}), on="orig_item_id", how="left")
    it = code(it, "content_type", "ctype")
    it = it.with_columns(
        pl.col("item_id").cast(pl.Int32),
        pl.col("release_year").cast(pl.Float32).alias("year"),
        pl.col("for_kids").cast(pl.Float32),
        pl.col("age_rating").cast(pl.Float32),
    )
    attrs = []
    for col, name in MULTI.items():
        long = (it.select("item_id", pl.col(col).str.split(",").alias("v")).explode("v")
                  .with_columns(pl.col("v").str.strip_chars()).filter(pl.col("v").is_not_null() & (pl.col("v") != "")))
        keep = long.group_by("v").agg(pl.len().alias("n")).filter(pl.col("n") >= MIN_ITEMS[name])
        long = long.join(keep.select("v"), on="v", how="semi")
        long = code(long.unique(), "v", "value").with_columns(pl.lit(name).alias("attr"))
        attrs.append(long.select("item_id", "attr", "value"))
    attrs.append(it.select("item_id", pl.lit("ctype").alias("attr"), pl.col("ctype").alias("value")))
    attrs = pl.concat(attrs).with_columns(pl.col("value").cast(pl.Int32))
    attrs.write_parquet(out / "item_attrs.parquet")
    it.select("item_id", "orig_item_id", "ctype", "year", "for_kids", "age_rating").write_parquet(out / "items.parquet")

    ev = (inter.rename({"user_id": "orig_user_id", "item_id": "orig_item_id"})
               .join(u.select("orig_user_id", "user_id", "age", "income", "sex", "kids"), on="orig_user_id")
               .join(it.select("orig_item_id", "item_id"), on="orig_item_id")
               .select(
                   "user_id", "item_id",
                   ((pl.col("last_watch_dt") - pl.lit(START)).dt.total_days() * 86400).cast(pl.Int32).alias("ts"),
                   pl.col("total_dur").cast(pl.Int32).alias("dur"),
                   pl.col("watched_pct").cast(pl.Float32).fill_null(0).alias("pct"),
                   "age", "income", "sex", "kids",
               ).sort("ts", maintain_order=True))
    ev.write_parquet(out / "events.parquet")
    print(ev.shape, u.shape, it.shape, attrs.group_by("attr").agg(pl.len(), pl.col("value").n_unique()))


if __name__ == "__main__":
    main()
