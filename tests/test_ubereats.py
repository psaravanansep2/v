import os
import tempfile
import unittest
from datetime import datetime

from ubereats import Offer, Settings, evaluate, insights, log_offer


class EvaluateTest(unittest.TestCase):
    def test_good_offer_is_accepted(self):
        r = evaluate(Offer(payout=12, pickup_miles=1, dropoff_miles=3, est_minutes=18))
        self.assertTrue(r.accept, r.reasons)
        self.assertGreater(r.net_per_hour, 20)

    def test_low_payout_long_trip_is_declined(self):
        r = evaluate(Offer(payout=4, pickup_miles=5, dropoff_miles=8, est_minutes=35))
        self.assertFalse(r.accept)
        self.assertEqual(len(r.reasons), 4)

    def test_restaurant_wait_lowers_hourly(self):
        base = evaluate(Offer(10, 1, 3, 18))
        waited = evaluate(Offer(10, 1, 3, 18, restaurant_wait_min=15))
        self.assertLess(waited.net_per_hour, base.net_per_hour)

    def test_custom_settings(self):
        offer = Offer(8, 1, 2, 15)
        self.assertTrue(evaluate(offer, Settings(min_net_per_hour=15)).accept)
        self.assertFalse(evaluate(offer, Settings(min_net_per_hour=40)).accept)


class InsightsTest(unittest.TestCase):
    def test_ranks_best_hour_first(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "log.csv")
            for day in range(1, 4):
                log_offer(path, Offer(15, 1, 3, 18), "Chipotle", "Downtown",
                          when=datetime(2026, 9, day, 18))
                log_offer(path, Offer(4, 3, 5, 30), "Taco Bell", "Suburbs",
                          when=datetime(2026, 9, day, 14))
            data = insights(path)
        self.assertEqual(data["hour"][0]["key"], "18:00")
        self.assertEqual(data["restaurant"][0]["key"], "Chipotle")
        self.assertEqual(data["zone"][-1]["key"], "Suburbs")


if __name__ == "__main__":
    unittest.main()
