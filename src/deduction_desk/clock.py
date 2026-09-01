"""Time. All of it in Asia/Kolkata, all of it as ISO-8601 strings with an explicit offset.

Two decisions worth stating.

**Strings, not datetimes, at rest.** SQLite drops `tzinfo` silently on round-trip, so a
naive datetime read back from the database is an hour-and-a-half adrift from what was
written. In a system whose compliance claim rests on "no contact outside 09:30-18:30 IST",
that error lands exactly on the boundary where a violation would hide. ISO strings with a
`+05:30` offset survive the round-trip unchanged, sort correctly, and hash deterministically
for the audit log.

**A simulated clock, never `datetime.now()`.** The generator and the batch runner both take
their time from `SimClock`. A `now()` call anywhere in the generator would make byte-identical
regeneration impossible, and a `now()` call in the runner would make the compliance tests
pass or fail depending on what time of day you ran them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), "IST")

WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


# ----------------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------------
def to_iso(dt: datetime) -> str:
    """Serialise an aware datetime. Rejects naive input rather than assuming a zone."""
    if dt.tzinfo is None:
        raise ValueError("refusing to serialise a naive datetime; attach IST explicitly")
    return dt.astimezone(IST).isoformat(timespec="seconds")


def parse_iso(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def date_str(d: date | datetime) -> str:
    """YYYY-MM-DD, for value dates and due dates where the time of day is meaningless."""
    if isinstance(d, datetime):
        return d.astimezone(IST).date().isoformat()
    return d.isoformat()


def parse_date(text: str) -> date:
    return date.fromisoformat(text[:10])


def at_time(d: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(d, time(hh, mm), tzinfo=IST)


def add_days(text: str, days: int) -> str:
    """Add days to a date string, preserving the date-only shape."""
    return (parse_date(text) + timedelta(days=days)).isoformat()


def days_between(a: str, b: str) -> int:
    """`b - a` in whole days. Negative when b precedes a."""
    return (parse_date(b) - parse_date(a)).days


def weekday_name(d: date | str) -> str:
    if isinstance(d, str):
        d = parse_date(d)
    return WEEKDAY_NAMES[d.weekday()]


# ----------------------------------------------------------------------------------
# Business calendar
# ----------------------------------------------------------------------------------
def is_contact_day(d: date | str, allowed_days: list[str]) -> bool:
    return weekday_name(d) in {x.lower() for x in allowed_days}


def is_within_window(dt: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    """Is this instant inside the permitted contact window?

    Inclusive of both ends. The policy window is expressed as local wall-clock time, so
    the comparison is done on the IST-normalised time-of-day.
    """
    local = dt.astimezone(IST).time()
    sh, sm = (int(x) for x in start_hhmm.split(":"))
    eh, em = (int(x) for x in end_hhmm.split(":"))
    return time(sh, sm) <= local <= time(eh, em)


def next_contact_slot(
    d: date, allowed_days: list[str], start_hhmm: str
) -> datetime:
    """The first permitted contact instant on or after `d`.

    Used by the executor so that an action decided on a Saturday is scheduled for Monday
    morning rather than being dropped or, worse, sent anyway.
    """
    sh, sm = (int(x) for x in start_hhmm.split(":"))
    cursor = d
    for _ in range(14):  # a fortnight is more than enough to clear any weekend
        if is_contact_day(cursor, allowed_days):
            return at_time(cursor, sh, sm)
        cursor += timedelta(days=1)
    raise ValueError(f"no permitted contact day within 14 days of {d} given {allowed_days}")


# ----------------------------------------------------------------------------------
# Simulated clock
# ----------------------------------------------------------------------------------
@dataclass
class SimClock:
    """The batch runner's clock. Ticks a day at a time over a fixed window.

    `--days 45 --tick 1d` produces the recovery curve: without a simulated clock there is
    no time axis, and "money recovered" collapses into a single number with no story about
    how long it took or how many contacts it cost.
    """

    start: date
    days: int
    tick_days: int = 1

    @classmethod
    def from_strings(cls, start: str, days: int, tick_days: int = 1) -> SimClock:
        return cls(start=parse_date(start), days=days, tick_days=tick_days)

    def __iter__(self) -> Iterator[date]:
        for offset in range(0, self.days, self.tick_days):
            yield self.start + timedelta(days=offset)

    @property
    def end(self) -> date:
        return self.start + timedelta(days=self.days - 1)

    def contains(self, d: date | str) -> bool:
        if isinstance(d, str):
            d = parse_date(d)
        return self.start <= d <= self.end

    def day_index(self, d: date | str) -> int:
        if isinstance(d, str):
            d = parse_date(d)
        return (d - self.start).days
