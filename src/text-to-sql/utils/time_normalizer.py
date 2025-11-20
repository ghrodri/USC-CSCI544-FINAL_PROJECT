import re
from datetime import date, timedelta
from typing import Optional, Tuple
from dateutil.relativedelta import relativedelta

def normalize_time_phrase(
    phrase: str,
    today: Optional[date] = None,
) -> Optional[Tuple[str, str]]:
    if today is None:
        today = date.today()

    p = phrase.strip().lower()

    if p in {"today"}:
        start = end = today
        return start.isoformat(), end.isoformat()

    if p in {"yesterday"}:
        end = today - timedelta(days=1)
        start = end
        return start.isoformat(), end.isoformat()
    
    if p in {"this week"}:
        weekday = today.weekday() 
        start = today - timedelta(days=weekday)
        end = today
        return start.isoformat(), end.isoformat()
    
    if p in {"last week"}:
        weekday = today.weekday()
        this_monday = today - timedelta(days=weekday)
        last_week_end = this_monday - timedelta(days=1)
        last_week_start = last_week_end - timedelta(days=6)
        return last_week_start.isoformat(), last_week_end.isoformat()

    if p in {"this month"}:
        start = date(today.year, today.month, 1)
        end = today
        return start.isoformat(), end.isoformat()

    if p in {"last month"}:
        first_this_month = date(today.year, today.month, 1)
        last_month_end = first_this_month - timedelta(days=1)
        last_month_start = date(last_month_end.year, last_month_end.month, 1)
        return last_month_start.isoformat(), last_month_end.isoformat()

    if p in {"this year", "year to date", "ytd"}:
        start = date(today.year, 1, 1)
        end = today
        return start.isoformat(), end.isoformat()

    if p in {"last year"}:
        start = date(today.year - 1, 1, 1)
        end = date(today.year - 1, 12, 31)
        return start.isoformat(), end.isoformat()

    if p in {"last quarter"}:
        current_quarter = (today.month - 1) // 3 + 1
        if current_quarter == 1:
            year = today.year - 1
            quarter = 4
        else:
            year = today.year
            quarter = current_quarter - 1

        start_month = 3 * (quarter - 1) + 1
        start = date(year, start_month, 1)
        end_month = start_month + 2
        end = date(year, end_month, 1) + relativedelta(months=1) - timedelta(days=1)
        return start.isoformat(), end.isoformat()
    
    m = re.match(r"last\s+(\d+)\s+days?", p)
    if m:
        n = int(m.group(1))
        end = today
        start = today - timedelta(days=n)
        return start.isoformat(), end.isoformat()
    
    m = re.match(r"last\s+(\d+)\s+months?", p)
    if m:
        n = int(m.group(1))
        end = today
        start = today - relativedelta(months=n)
        return start.isoformat(), end.isoformat()
    m = re.match(r"last\s+(\d+)\s+years?", p)
    if m:
        n = int(m.group(1))
        end = today
        start = today - relativedelta(years=n)
        return start.isoformat(), end.isoformat()
    return None