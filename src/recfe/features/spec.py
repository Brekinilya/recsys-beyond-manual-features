"""Feature grammar.

A history feature is an aggregate over the event log restricted to events that happened
before the prediction moment (cutoff):

    Spec(key, event, agg, window, value, hl, red)

key     what the events are grouped by and how the group is matched to a (user, item) row
          user          -- the row's user
          item          -- the row's item
          ui            -- the (user, item) pair            (direct link)
          a:<attr>      -- the item's attribute value       (artist popularity, genre popularity)
          ua:<attr>     -- (user, item's attribute value)   (indirect link: user x artist, user x genre)
          s:<seg>       -- (user's segment, item)           (item popularity inside the user's age group)
event   named event filter, defined per dataset (listen, skip, like, watch, ...)
agg     cnt | sum | mean | last (days since last event) | first (days since first event)
        | dec (exponentially decayed count, half-life `hl` days) | nuniq (distinct counterparts) | days (active days)
window  look-back in days (None = whole history)
value   value column for sum / mean (played ratio, watched pct, ...)
red     reducer over multi-valued attributes (an item has several genres): max | mean | sum | min

Ratio(num, den) divides two specs: conversion rates (skips / listens), shares
(user x artist listens / user listens), trends (7-day count / 30-day count).
"""
from __future__ import annotations

from dataclasses import dataclass, field

COUNT_LIKE = {"cnt", "sum", "dec", "nuniq", "days"}


@dataclass(frozen=True)
class Spec:
    key: str
    event: str
    agg: str
    window: int | None = None
    value: str | None = None
    hl: float | None = None
    red: str | None = None

    @property
    def name(self) -> str:
        agg = self.agg
        if self.value:
            agg += f"({self.value})"
        if self.hl is not None:
            agg += f"(h{self.hl:g})"
        w = "all" if self.window is None else f"w{self.window}"
        s = f"{self.key}|{self.event}|{agg}|{w}"
        if self.red:
            s += f"|{self.red}"
        return s

    @property
    def fill_zero(self) -> bool:
        return self.agg in COUNT_LIKE

    def base(self) -> list[Spec]:
        return [self]


@dataclass(frozen=True)
class Ratio:
    num: Spec
    den: Spec

    @property
    def name(self) -> str:
        return f"R[{self.num.name} / {self.den.name}]"

    def base(self) -> list[Spec]:
        return [self.num, self.den]


Feature = Spec | Ratio


def S(key, event, agg, window=None, value=None, hl=None, red=None) -> Spec:
    return Spec(key, event, agg, window, value, hl, red)


def R(num: Spec, den: Spec) -> Ratio:
    return Ratio(num, den)


@dataclass
class Grammar:
    """Search space of a dataset: which keys, events, aggregations and windows are allowed."""
    keys: list[str]
    events: list[str]
    values: list[str]
    multi_attrs: set[str] = field(default_factory=set)
    max_window: int = 300
    min_window: int = 1
    # pairs of events for which num/den conversion rates make sense, e.g. (skip, listen)
    conversions: list[tuple[str, str]] = field(default_factory=list)
    # how per-value aggregates are folded when an item has several attribute values
    reducers: tuple[str, ...] = ("max", "mean", "sum", "min")
