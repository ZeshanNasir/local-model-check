"""Small summary statistics: median and spread, never a single favourable run."""
import statistics


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def spread_pct(values):
    """(max - min) / median, as a percentage. High values mean the median is not trustworthy."""
    s = summarize(values)
    if s["n"] < 2 or not s["median"]:
        return None
    return 100 * (s["max"] - s["min"]) / s["median"]
