"""One-time fetch: each current driver's race wins BEFORE 2018.

collect_data.py only has results from 2018 on, so its CareerWins / WinsLast3Years
undercount anyone who was winning earlier (Hamilton showed 44 instead of ~106).
Pre-2018 history never changes, so fetch it once from Jolpica (the Ergast mirror)
and store it in data/pre2018_wins.json:

    {"HAM": {"total": 62, "by_year": {"2016": 10, "2017": 9}}, ...}

total    = every win before 2018
by_year  = wins in 2016 and 2017 only -- the only pre-2018 years that fall inside
           the 3-year window used for WinsLast3Years in the 2018 and 2019 seasons.

Drivers are keyed by the same abbreviation collect_data.py uses, and matched through
the Ergast driverId that FastF1 already stores in the raw snapshots.

Run:  python fetch_pre2018_wins.py
"""
import json
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
API = 'https://api.jolpi.ca/ergast/f1'
OUT = ROOT / 'data' / 'pre2018_wins.json'
WINDOW_YEARS = (2016, 2017)


def get(url, tries=4):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'formula-forecast'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))   # rate-limited or flaky: back off and retry


def main():
    raw = pd.concat([pd.read_csv(p) for p in sorted((ROOT / 'data' / 'raw').glob('season_*.csv'))])
    driver_ids = raw.groupby('Abbreviation')['DriverId'].first().to_dict()    # {'HAM': 'hamilton', ...}
    by_id = {v: k for k, v in driver_ids.items()}

    total = Counter()                      # driverId -> wins before 2018
    per_year = defaultdict(Counter)        # driverId -> {year: wins} for the window years
    for season in range(1950, 2018):
        j = get(f'{API}/{season}/results/1.json?limit=100')                    # position 1 only
        races = j['MRData']['RaceTable']['Races']
        rows = 0
        for race in races:
            for res in race['Results']:          # >1 row when a win was a shared drive
                winner = res['Driver']['driverId']
                total[winner] += 1
                rows += 1
                if season in WINDOW_YEARS:
                    per_year[winner][season] += 1
        if rows != int(j['MRData']['total']):    # guard against a truncated response
            raise SystemExit(f'{season}: expected {j["MRData"]["total"]} winning results, got {rows}')
        time.sleep(0.4)

    out = {abbr: {'total': int(total.get(did, 0)),
                  'by_year': {str(y): int(per_year[did].get(y, 0)) for y in WINDOW_YEARS}}
           for abbr, did in sorted(driver_ids.items())}
    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'wrote {OUT} ({len(out)} drivers, {sum(total.values())} pre-2018 wins in total)')

    # Cross-check: Ergast career total should equal (our 2018+ wins) + (pre-2018 wins).
    print('\ncross-check against each driver\'s Jolpica career total:')
    bad = 0
    ours = raw[raw['Position'] == 1].groupby('Abbreviation').size()
    for abbr, did in sorted(driver_ids.items()):
        if out[abbr]['total'] == 0 and ours.get(abbr, 0) == 0:
            continue
        career = int(get(f'{API}/drivers/{did}/results/1.json?limit=1')['MRData']['total'])
        mine = out[abbr]['total'] + int(ours.get(abbr, 0))
        flag = '' if career == mine else f'   <-- MISMATCH (api {career})'
        bad += career != mine
        print(f'  {abbr}: pre-2018 {out[abbr]["total"]:>3} + since {int(ours.get(abbr, 0)):>3} = {mine:>3}{flag}')
        time.sleep(0.4)
    print(f'\n{bad} mismatches')


if __name__ == '__main__':
    main()
