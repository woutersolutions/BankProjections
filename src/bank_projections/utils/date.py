import datetime

from dateutil.relativedelta import relativedelta


def end_of_month(date: datetime.date) -> datetime.date:
    """Return the last day of the month of the given date."""
    first_next_month = date.replace(day=1) + relativedelta(months=1)
    return first_next_month - datetime.timedelta(days=1)


def is_end_of_month(date: datetime.date) -> bool:
    return date.day == end_of_month(date).day


def add_months(date: datetime.date, months: int, make_end_of_month: bool = False) -> datetime.date:
    """Add (or subtract) months to a date, optionally snapping to end of month."""
    result = date + relativedelta(months=months)
    return end_of_month(result) if make_end_of_month else result
