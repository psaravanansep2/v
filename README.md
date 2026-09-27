# Uber Eats offer evaluator

Decide in seconds whether an Uber Eats delivery offer is worth taking, and learn
from your own history which hours, days, zones and restaurants pay best.

Pure Python 3.9+, no dependencies. It does not connect to or automate the Uber app
(that violates Uber's terms and risks deactivation) — you enter the numbers the
offer card shows.

## How the algorithm works

For each offer it estimates the **real** cost of the trip:

- **Miles** = pickup + dropoff + a share of the dropoff you drive back unpaid (`return_trip_factor`, default 50%)
- **Time** = Uber's estimate + expected restaurant wait + the return drive
- **Net** = payout − miles × `cost_per_mile` (fuel + wear)

It then checks net $/hour, net $/mile, minimum payout and maximum pickup distance
against your targets, and gives a 0–100 score (half $/hr, half $/mi, capped at 2× target).
Any failed check → **DECLINE**, with the reasons listed.

## Usage

```bash
# score PAYOUT PICKUP_MILES DROPOFF_MILES EST_MINUTES
python -m ubereats score 9.50 1.2 3.4 20 --wait 5
# DECLINE  (score 37.8/100)
#   net $7.61 | $14.10/hr | $1.21/mi | 6.3 mi, 32.4 min incl. return

# Your own targets (global options go before the command)
python -m ubereats --min-hour 25 --min-mile 2 --cost-per-mile 0.40 score 12 1 3 18

# Log offers as you get them...
python -m ubereats score 11 0.8 2.5 16 --restaurant "Chipotle" --zone Downtown --log offers.csv

# ...then find your best hours / days / zones / restaurants
python -m ubereats insights offers.csv
```

## Tests

```bash
python -m unittest discover -s tests
```
