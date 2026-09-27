"""Score Uber Eats delivery offers so you can decide quickly whether to accept."""

from dataclasses import dataclass


@dataclass
class Offer:
    payout: float                 # guaranteed payout shown on the offer ($, incl. tip)
    pickup_miles: float           # your location -> restaurant
    dropoff_miles: float          # restaurant -> customer
    est_minutes: float            # total time Uber estimates for the trip
    restaurant_wait_min: float = 0.0  # expected wait at the restaurant
    orders: int = 1               # number of orders in the offer (stacked = 2+)


@dataclass
class Settings:
    cost_per_mile: float = 0.30        # fuel + wear; use ~0.67 for full IRS rate
    min_net_per_hour: float = 20.0
    min_net_per_mile: float = 1.50
    min_payout: float = 5.00
    max_pickup_miles: float = 4.0
    return_trip_factor: float = 0.5    # share of dropoff miles you drive back unpaid


@dataclass
class Score:
    total_miles: float
    total_minutes: float
    net_payout: float
    net_per_hour: float
    net_per_mile: float
    score: float                       # 0-100, higher is better
    accept: bool
    reasons: list


def evaluate(offer: Offer, s: Settings = Settings()) -> Score:
    paid_miles = offer.pickup_miles + offer.dropoff_miles
    total_miles = paid_miles + offer.dropoff_miles * s.return_trip_factor
    # Assume the dead-head return takes about as long per mile as the delivery leg.
    minutes_per_mile = offer.est_minutes / paid_miles if paid_miles else 3.0
    total_minutes = (offer.est_minutes + offer.restaurant_wait_min
                     + offer.dropoff_miles * s.return_trip_factor * minutes_per_mile)

    net = offer.payout - total_miles * s.cost_per_mile
    per_hour = net / (total_minutes / 60) if total_minutes > 0 else 0.0
    per_mile = net / total_miles if total_miles > 0 else net

    reasons = []
    if offer.payout < s.min_payout:
        reasons.append(f"payout ${offer.payout:.2f} < ${s.min_payout:.2f} minimum")
    if offer.pickup_miles > s.max_pickup_miles:
        reasons.append(f"pickup {offer.pickup_miles:.1f} mi > {s.max_pickup_miles:.1f} mi limit")
    if per_hour < s.min_net_per_hour:
        reasons.append(f"${per_hour:.2f}/hr < ${s.min_net_per_hour:.2f}/hr target")
    if per_mile < s.min_net_per_mile:
        reasons.append(f"${per_mile:.2f}/mi < ${s.min_net_per_mile:.2f}/mi target")

    # Score: half from $/hr, half from $/mi, each capped at 2x its target.
    hour_part = min(per_hour / s.min_net_per_hour, 2.0) / 2 if s.min_net_per_hour else 1.0
    mile_part = min(per_mile / s.min_net_per_mile, 2.0) / 2 if s.min_net_per_mile else 1.0
    score = max(0.0, 50 * hour_part + 50 * mile_part)

    return Score(
        total_miles=round(total_miles, 2),
        total_minutes=round(total_minutes, 1),
        net_payout=round(net, 2),
        net_per_hour=round(per_hour, 2),
        net_per_mile=round(per_mile, 2),
        score=round(score, 1),
        accept=not reasons,
        reasons=reasons,
    )
