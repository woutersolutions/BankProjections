import calendar
import datetime
from collections.abc import Iterator

from dateutil.relativedelta import relativedelta


class TimeHorizon:
    def __init__(self, dates: list[datetime.date]):
        self.dates = sorted(set(dates))

    @property
    def start_date(self) -> datetime.date:
        return self.dates[0]

    @property
    def end_date(self) -> datetime.date:
        return self.dates[-1]

    def __len__(self) -> int:
        return len(self.dates)

    @staticmethod
    def from_numbers(
        start_date: datetime.date,
        number_of_days: int = 0,
        number_of_weeks: int = 0,
        number_of_months: int = 0,
        number_of_quarters: int = 0,
        number_of_years: int = 0,
        end_of_month: bool | None = None,
    ) -> "TimeHorizon":
        dates = [start_date]

        if end_of_month is None:
            end_of_month = start_date.day == calendar.monthrange(start_date.year, start_date.month)[1]

        # Add days
        for i in range(number_of_days):
            dates.append(start_date + relativedelta(days=i + 1))

        # Add weeks
        for i in range(number_of_weeks):
            dates.append(start_date + relativedelta(weeks=i + 1))

        # Add months
        for i in range(number_of_months):
            projection_date = start_date + relativedelta(months=i + 1)
            if end_of_month:
                projection_date = to_end_of_month(projection_date)
            dates.append(projection_date)

        # Add quarters
        for i in range(number_of_quarters):
            projection_date = start_date + relativedelta(months=3 * i + 1)
            if end_of_month:
                projection_date = to_end_of_month(projection_date)
            dates.append(projection_date)

        # Add years
        for i in range(number_of_years):
            projection_date = start_date + relativedelta(years=i + 1)
            if end_of_month:
                projection_date = to_end_of_month(projection_date)
            dates.append(projection_date)

        return TimeHorizon(dates)

    # Iterator to loop over the dates, returning TimeIncrements
    # By default returns the initial increment start_date to_start_date
    def __iter__(self) -> "Iterator[TimeIncrement]":
        yield TimeIncrement(self.dates[0], self.dates[0])
        for i in range(len(self.dates) - 1):
            yield TimeIncrement(self.dates[i], self.dates[i + 1])


def to_end_of_month(date: datetime.date) -> datetime.date:
    last_day = calendar.monthrange(date.year, date.month)[1]
    return date.replace(day=last_day)


class TimeIncrement:
    def __init__(self, from_date: datetime.date, to_date: datetime.date):
        self.from_date = from_date
        self.to_date = to_date

    @property
    def days(self) -> int:
        return (self.to_date - self.from_date).days

    @property
    def portion_year(self) -> float:
        return self.days / 365.25

    def contains(self, date: datetime.date) -> bool:
        if self.from_date == self.to_date:
            return self.from_date == date
        else:
            return self.from_date < date <= self.to_date

    def overlaps(self, start_date: datetime.date, end_date: datetime.date) -> bool:
        return not (self.to_date < start_date or self.from_date >= end_date)

    def days_overlap(self, start_date: datetime.date, end_date: datetime.date) -> int:
        if not self.overlaps(start_date, end_date):
            return 0
        overlap_start = max(self.from_date + relativedelta(days=1), start_date)
        overlap_end = min(self.to_date, end_date)
        return (overlap_end - overlap_start).days + 1

    def __repr__(self):
        return f"TimeIncrement(from_date={self.from_date}, to_date={self.to_date})"
