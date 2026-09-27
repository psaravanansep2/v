# Uber Eats offer evaluator

Decide in seconds whether an Uber Eats delivery offer is worth taking, and learn
from your own history which hours, days, zones and restaurants pay best.

Pure Python 3.9+, no dependencies. It does not connect to or automate the Uber app
(that violates Uber's terms and risks deactivation) — you enter the numbers the
offer card shows.

## On your phone: Dash Spotter (`docs/index.html`)

A one-page web app for use while you drive, with the same scoring as the Python tool.

- **Offer:** type in payout, minutes and miles, and ACCEPT or DECLINE updates as you type.
  Tap *I took it* or *I passed* to log the offer with your GPS position and the time.
- **Go online:** tracks your location while the page is open. It adds up miles driven and
  time spent in each ~half-mile square, and keeps the screen awake. Phones pause location for
  pages in the background, so keep it on screen (a dashboard mount works well).
- **Spots:** for the current hour and day, or any time you pick, it ranks the squares where good
  offers came in most often per hour you spent there. Tap *Go* to open directions in Google Maps.
  Offers from similar hours and days count partly, and squares with under ~2 hours of data are
  pulled toward your average, so one lucky offer doesn't put a spot on top.
- **History:** today's earnings, online time and net $/hr. *Export CSV* saves a file that
  `python -m ubereats insights` can read (it adds a "Best spots" list).

All data stays in the phone's browser. Nothing is uploaded.

### Getting it on your phone

Location only works over HTTPS, so host the page with GitHub Pages:

1. Merge this branch into `main` (or pick this branch in step 2).
2. On GitHub: **Settings → Pages → Build and deployment → Deploy from a branch**, choose
   `main` and the `/docs` folder, then save. (Pages on a private repo needs a paid GitHub plan.
   Otherwise make the repo public. The page holds no personal data, since your logs stay on your phone.)
3. Open `https://<your-username>.github.io/<repo>/` on your phone, allow location, and use
   **Add to Home Screen** so it opens like an app.

To try it before your first shift, use **Settings → Load 3 weeks of demo data** (remove it after).

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
