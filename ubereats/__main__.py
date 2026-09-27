"""Command line: python -m ubereats {score,insights} ..."""

import argparse

from .history import insights, log_offer
from .offer import Offer, Settings, evaluate


def _settings(a):
    return Settings(cost_per_mile=a.cost_per_mile, min_net_per_hour=a.min_hour,
                    min_net_per_mile=a.min_mile, min_payout=a.min_payout,
                    max_pickup_miles=a.max_pickup)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ubereats", description="Uber Eats offer evaluator")
    p.add_argument("--cost-per-mile", type=float, default=0.30)
    p.add_argument("--min-hour", type=float, default=20.0, help="target net $/hr")
    p.add_argument("--min-mile", type=float, default=1.50, help="target net $/mile")
    p.add_argument("--min-payout", type=float, default=5.00)
    p.add_argument("--max-pickup", type=float, default=4.0, help="max miles to restaurant")
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score", help="evaluate one offer")
    sc.add_argument("payout", type=float)
    sc.add_argument("pickup_miles", type=float)
    sc.add_argument("dropoff_miles", type=float)
    sc.add_argument("est_minutes", type=float)
    sc.add_argument("--wait", type=float, default=0.0, help="expected restaurant wait (min)")
    sc.add_argument("--orders", type=int, default=1)
    sc.add_argument("--log", metavar="CSV", help="append this offer to a CSV log")
    sc.add_argument("--restaurant", default="")
    sc.add_argument("--zone", default="")

    ins = sub.add_parser("insights", help="best hours/days/zones/restaurants from a log")
    ins.add_argument("csv")
    ins.add_argument("--top", type=int, default=5)
    ins.add_argument("--min-offers", type=int, default=3)

    a = p.parse_args(argv)
    s = _settings(a)

    if a.cmd == "score":
        offer = Offer(a.payout, a.pickup_miles, a.dropoff_miles, a.est_minutes,
                      a.wait, a.orders)
        r = evaluate(offer, s)
        print(f"{'ACCEPT' if r.accept else 'DECLINE'}  (score {r.score}/100)")
        print(f"  net ${r.net_payout:.2f} | ${r.net_per_hour:.2f}/hr | ${r.net_per_mile:.2f}/mi"
              f" | {r.total_miles} mi, {r.total_minutes} min incl. return")
        for reason in r.reasons:
            print(f"  - {reason}")
        if a.log:
            log_offer(a.log, offer, a.restaurant, a.zone, accepted=r.accept)
    else:
        data = insights(a.csv, s, a.min_offers)
        for dim, rows in data.items():
            if not rows:
                continue
            print(f"\nBest {dim}s:")
            for row in rows[:a.top]:
                print(f"  {row['key']:<20} ${row['avg_net_per_hour']:>6.2f}/hr  "
                      f"{row['good_rate']:>4.0%} good  ({row['offers']} offers)")


if __name__ == "__main__":
    main()
