# Airport Opportunity Score methodology

The Airport Opportunity Score is a **preliminary screening proxy**. It helps decide which
airports deserve further review. It does not measure passenger-only capacity, terminal capacity,
unmet demand, profitability or investment returns, and it is not an investment recommendation.

All calculations run in Python (`src/analytics.py`, `src/scoring.py`). The LLM selects tools and
explains their results; it never calculates or changes a score.

## The 11 weighted metrics

`P`, `D` and `S` are annual passengers, departures and seats from the BTS airport summary
(`r495-tyji`). Each counts domestic and outbound international traffic at the origin airport.

| # | Metric | Weight | Formula | Cargo-inclusive |
|---:|---|---:|---|---|
| 1 | Passenger CAGR 3Y | 15 | `(P2025 / P2022)^(1/3) - 1` | No |
| 2 | Passenger CAGR 10Y | 10 | `(P2025 / P2015)^(1/10) - 1` | No |
| 3 | Absolute Passenger Growth 3Y | 10 | `P2025 - P2022` | No |
| 4 | Total Departure CAGR 3Y | 5 | `(D2025 / D2022)^(1/3) - 1` | **Yes** |
| 5 | Total Departure CAGR 10Y | 5 | `(D2025 / D2015)^(1/10) - 1` | **Yes** |
| 6 | Passenger vs Total Departure Growth Gap 3Y | 15 | Metric 1 minus metric 4 | **Yes** |
| 7 | Total Load Factor 2025 (passengers / seats) | 15 | `P2025 / S2025` | No |
| 8 | Total Load Factor Change 3Y (passengers / seats) | 10 | `P2025 / S2025 - P2022 / S2022` | No |
| 9 | Passengers per Departure Growth 3Y | 5 | `((P2025 / D2025) / (P2022 / D2022))^(1/3) - 1` | **Yes** |
| 10 | Total Departure Growth Acceleration | 5 | Metric 4 minus metric 5 | **Yes** |
| 11 | Passenger Growth Acceleration | 5 | Metric 1 minus metric 2 | No |
| | **Total** | **100** | | **35** |

Higher values are treated as favorable for every metric. This is a provisional screening
assumption: for example, faster growth or a higher load factor can also signal constraints.

Absolute Passenger Growth 10Y is also available in rankings, profiles and comparisons, but it has
no weight.

## Normalization and score

For each metric, calculated from unrounded values across the fixed national peer group:

```text
L = 5th percentile, U = 95th percentile (linear interpolation)
N = 100 x (min(max(value, L), U) - L) / (U - L)
Score = sum(weight x N) / 100
```

Scores are rounded only for display. Ranks use unrounded scores, and ties break by airport code.
If `U` equals `L` for a metric, the calculation stops with an error.

## Peer group

The peer group is every airport with at least 100,000 FAA CY2025 enplanements and all 11 metrics.
It is selected from the data, and the scoring code names no airport. Limits are calculated once, before any
filter, so a geographic filter only changes which airports are shown. It never changes a score or
national rank. The 100,000 threshold is provisional.

In the current snapshot, 229 of 233 eligible airports are scored.

## Display filters

State and passenger filters only change which airports are shown. They do not change the 11 metrics,
weights, peer group, normalization limits, scores or national ranks. A filtered list can show a separate
"Position in this list".

The optional passenger filter uses **2025 departing passengers** from T-100, which is about half of total
two-way traffic. Any non-negative whole number is accepted, with "at least" and "over" kept distinct. Users can
ask for any size in their question. No filter applies by default, including for terminal-expansion questions.

Scores are points out of 100. They are not probabilities or chances of success, and they are never rescaled
so that the top airport reaches 100.

## Missing data

| Month in the source | Treatment |
|---|---|
| Departures, passengers and seats are all above zero | Observed |
| Reported row with zero passengers | A genuine zero |
| Blank departures, passengers or seats | Missing |
| No row | Missing, because this API cannot confirm zero activity |

A year is usable only with 12 reported months, no blank totals, and positive annual departures,
passengers and seats. An airport missing any scored metric gets no score, and the reason is
reported. Nothing is estimated, and the remaining weights are never adjusted.

In the current snapshot, four eligible airports have no score: HDN (2015), JAC (2022), VRB (2015 and
2022) and XWA (2015). JAC and HDN had documented runway closures in those years, but missing rows
alone do not prove zero activity, so they stay unscored.

## Warnings

Warnings are shown separately from the score and never change a value or rank.

- **Small base:** an airport with fewer than 25,000 passengers in 2015 or 2022 is flagged in
  rankings and profiles, not excluded. ILG had 4,988 passengers in 2022, and PAE had 92 in 2015, so their
  growth rates say little about the market. ILG currently ranks first, which should be read with this warning.
- **Route-file warnings:** these flag low passenger-aircraft or scheduled-service shares. They use
  January to July 2026 data for US carriers only, so they show recent mix, not historical change.

## Limitations

- **Cargo:** 35% of the score uses cargo-inclusive departure metrics. The API's departures include
  freighters and cannot be filtered. In the 2026 route file, 89 of 232 eligible airports are under 95%
  passenger aircraft, and 24 are under 80%. Growth is distorted when an airport's cargo share changes over
  time, which this API cannot show.
- **Load factor:** Total Load Factor is passengers divided by seats. It is not distance-weighted, and
  it is not a passenger-only measure, because a few seats are reported on cargo-class flights.
- **Size bias:** absolute passenger growth favors larger airports. Its rank correlation with FAA
  enplanements is 0.79. Twelve airports reach the upper limit and earn the same 10 points, however far above
  it they are. Read it alongside CAGR.
- **2022 recovery base:** 2022 was still a recovery year, so 3-year growth may reflect rebound rather than
  new demand.
- **Passenger counts:** T-100 airport passengers count departing passengers only and include people connecting
  at the airport, so hub growth may reflect airline network decisions rather than local demand.
- **Metric overlap:** weights are preserved, but metrics overlap. The Growth Gap and Passengers per Departure
  Growth rank airports almost identically (rank correlation about 1.00), so about 20 points effectively
  measure one signal. Passenger CAGR 3Y also feeds the Growth Gap and Passenger Growth Acceleration.
- **Sensitivity:** results depend on the weights, the threshold and the data snapshot. Scores are
  relative to the current peer group and cannot be compared across different groups or thresholds.
