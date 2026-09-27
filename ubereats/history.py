"""Log offers to a CSV and find the hours, days and restaurants that pay best."""

import csv
import os
from collections import defaultdict
from datetime import datetime

from .offer import Offer, Settings, evaluate

FIELDS = ["timestamp", "restaurant", "zone", "payout", "pickup_miles",
          "dropoff_miles", "est_minutes", "restaurant_wait_min", "orders", "accepted", "lat", "lng"]


def log_offer(path, offer: Offer, restaurant="", zone="", accepted=None, when=None, lat=None, lng=None):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            "timestamp": (when or datetime.now()).isoformat(timespec="minutes"),
            "restaurant": restaurant,
            "zone": zone,
            "payout": offer.payout,
            "pickup_miles": offer.pickup_miles,
            "dropoff_miles": offer.dropoff_miles,
            "est_minutes": offer.est_minutes,
            "restaurant_wait_min": offer.restaurant_wait_min,
            "orders": offer.orders,
            "accepted": "" if accepted is None else int(bool(accepted)),
            "lat": "" if lat is None else lat,
            "lng": "" if lng is None else lng,
        })


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            offer = Offer(
                payout=float(r["payout"]),
                pickup_miles=float(r["pickup_miles"]),
                dropoff_miles=float(r["dropoff_miles"]),
                est_minutes=float(r["est_minutes"]),
                restaurant_wait_min=float(r.get("restaurant_wait_min") or 0),
                orders=int(r.get("orders") or 1),
            )
            rows.append((datetime.fromisoformat(r["timestamp"]), r, offer))
    return rows


def _spot(row):
    try:
        return f"{float(row['lat']):.2f},{float(row['lng']):.2f}"
    except (KeyError, TypeError, ValueError):
        return ""


def insights(path, s: Settings = Settings(), min_offers=3):
    """Rank hours, weekdays, zones, restaurants and map spots by average net $/hr and good-offer rate.

    A spot is a ~half-mile square (lat/lng rounded to 2 decimals), the same grid the phone page uses.
    """
    groups = {"hour": defaultdict(list), "weekday": defaultdict(list), "zone": defaultdict(list),
              "restaurant": defaultdict(list), "spot": defaultdict(list)}
    for ts, row, offer in load(path):
        sc = evaluate(offer, s)
        keys = {
            "hour": f"{ts.hour:02d}:00",
            "weekday": ts.strftime("%A"),
            "zone": row.get("zone") or "",
            "restaurant": row.get("restaurant") or "",
            "spot": _spot(row),
        }
        for dim, key in keys.items():
            if key:
                groups[dim][key].append(sc)

    result = {}
    for dim, buckets in groups.items():
        ranked = []
        for key, scores in buckets.items():
            if len(scores) < min_offers:
                continue
            ranked.append({
                "key": key,
                "offers": len(scores),
                "avg_net_per_hour": round(sum(x.net_per_hour for x in scores) / len(scores), 2),
                "good_rate": round(sum(x.accept for x in scores) / len(scores), 2),
            })
        ranked.sort(key=lambda d: (d["avg_net_per_hour"], d["good_rate"]), reverse=True)
        result[dim] = ranked
    return result
