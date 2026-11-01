# Data sources and limitations

Figures below come from the snapshot built by `python -m src.pipeline`.
`data/processed/manifest.json` records retrieval time, row counts and file hashes.

## Sources

| Source | Access | Coverage | Use |
|---|---|---|---|
| T-100 airport summary, BTS `r495-tyji` | [Public API](https://data.bts.gov/resource/r495-tyji.json) | Jan 2014–Jul 2026, 134,619 airport-month rows | Annual passengers, departures and seats |
| T-100 national summary, BTS `bu82-4pwz` | [Public API](https://data.bts.gov/resource/bu82-4pwz.json) | Jan 2014–Jul 2026, 151 national monthly rows, no airport field | Reconciliation only |
| T-100 Segment, US Carriers Only | Manual TranStats download: `data/raw/T_T100_SEGMENT_US_CARRIER_ONLY 3.csv` (not in Git) | Jan–Jul 2026, 302,774 route rows, 91 MB | Route-level long-haul calculation |
| Distance group lookup | Manual TranStats download: `data/raw/L_DISTANCE_GROUP_500.csv` (in Git) | 25 bands of 500 miles | Reference; the calculation uses exact `DISTANCE` |
| FAA CY2025 commercial service enplanements | [FAA XLSX](https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/arp-cy2025-commercial-service-enplanements.xlsx) | 516 airports, CY2024 and CY2025 | State, hub class, cross-check |
| BTS On-Time Performance | Monthly TranStats ZIPs (not in Git) | Jan–Dec 2025 | Operational congestion only |

## Git policy

Raw files stay local, except the small distance lookup. The route file must be
downloaded by hand from TranStats and placed in `data/raw/`. The compact files in
`data/processed/` let the app run without any raw data.

## How flights and passengers are counted

**Inbound and outbound.** National series count international flights in both
directions; airport totals count them only at their origin:

- `national total = domestic + outbound + inbound`
- `airport total = domestic + outbound`

Both identities hold exactly for all 13 years, and the airport identity holds on
every airport-month row. Airport totals therefore count each departure once.
Inbound fields must never be added to them.

**Connecting passengers.** T-100 Segment counts passengers per flight segment. A
connecting passenger appears once at each airport they depart from, so:

- Passengers are never summed across airports as a measure of demand.
- Origin and destination counts for an airport are never added together.
- An airport's total still includes people connecting there. T-100 cannot separate
  connecting from local passengers; that needs origin-destination data (DB1B),
  which is not in the MVP. Hub growth may reflect airline network decisions rather
  than local demand.

**Reconciliation across API datasets.** For 2015, 2022 and 2025, airport departures, passengers and
seats each equal national domestic plus outbound exactly, and national totals equal domestic plus
outbound plus inbound exactly. Airport and national totals must not be mixed in a ratio, because
national totals include inbound traffic. The API publishes field names but no column definitions, so this
evidence comes from these checks.

**Cross-check.** For the 144 hub airports matched to the FAA file, 2025 T-100
passengers are 0.995–1.033 times FAA enplanements, with a median of 1.004. For 2024 the range is
0.989–1.030, also with a median of 1.004.

## Limitations

**Route file**

- It covers 7 months of 2026, so results may not represent a full year and must not
  be compared with calendar-year figures.
- It has US carriers only. Against the all-carrier summary for the same months, it
  covers 86.3% of ANC departures and 99.4% of ANC passengers. The carriers and
  distances of the missing departures are unknown, so the all-flight share is
  reported with bounds.
- 994 rows (0.33%) have `DEPARTURES_PERFORMED = 0`, and all of them report zero seats,
  passengers, freight and mail. 211 show scheduled departures (795 in total) and 783 show
  none; Spirit accounts for 584 rows, and 693 rows fall in June or July. The cause is not
  documented in the data. These rows add nothing to departure-weighted results. At ANC
  there are 11 such rows, all under 1,000 miles.
- The gap to the all-carrier summary at ANC is 4,502 domestic and 2,363 international
  departures, so it is not only international flights.
- 2,582 flown rows nationally have `DISTANCE = 0`, all with the same origin and
  destination. They stay in the departure total and count as not long-haul. At ANC
  they are 183 of 43,066 departures, so excluding them would move the all-flight share
  from 23.0% to 23.1%.
- The route file uses `PBI` for the airport the summary and FAA file call `DJT`. The tools map it, so
  route-file warnings and long-haul results use `DJT`.
- BTS lookup tables confirm `AIRCRAFT_CONFIG` 1 is passenger configuration, 2 freight,
  3 combined passenger and freight, and 4 seaplane. `CLASS` F is scheduled passenger/cargo
  service, G scheduled all-cargo, and L non-scheduled civilian service.

**Airport summary**

- `departures` includes freighters and cannot be filtered by aircraft type. In the
  2026 route file, passenger aircraft account for only 34.3% of US-carrier departures
  at MEM and 54.5% at ANC, so passengers per departure and total departure growth are
  distorted at cargo-heavy airports. See [METHODOLOGY.md](METHODOLOGY.md).
- 2022, the 3-year base, was still a recovery year: national airport-level departures
  were 8.97M, against 10.23M in 2019 and 9.95M in 2025.

The summary was fetched from the API, and no raw copy or hash was kept. It can't be reconciled with
an original file, including the 2015 and 2022 inputs to the score. Its internal totals reconcile
exactly with the national series, but that comes from the same provider, so it isn't independent
proof. FAA enplanements give an independent check only for 2024 and 2025, and the route file only for
January–July 2026. The route, FAA and on-time Parquet files have each been reconciled exactly with
their raw files.

**FAA file**

- 35 of 516 FAA codes do not appear in the summary. `FAA_TO_T100_CODES` in `src/config.py` maps 34
  codes, including IWA and GPI, which are small hubs with full history and scores. One of the 34 is
  A61, which the summary uses for a different airport. RVR and AFE stay unmatched, have no history in
  this snapshot, and fall below the eligibility threshold. The mapping comes from OurAirports, an
  unofficial source; each mapped pair was checked for overlapping airport names, not against an
  official list.
- T-100 passengers and FAA enplanements differ unusually at three eligible airports: KTN (1.57 times
  FAA in 2025, 1.55 in 2024), GRK (0.81 and 0.79) and JNU (1.20 in 2025). **The cause has not been
  established.** Scores use T-100, and eligibility uses FAA, so GRK qualifies with 91,854 T-100
  passengers in 2025.

**On-time data**

- It covers reporting carriers on domestic flights only, so international departures
  and most cargo are absent.
- Departures in the sample are scheduled flights, including cancelled ones, while T-100 counts
  flights performed. The coverage share can therefore slightly overstate coverage: LAX is 91.1%,
  or 90.5% counting flown flights only, and five eligible airports exceed 100%.
- Delay minutes per arrival divide by scheduled arrivals, including cancelled flights: LAX NAS delay
  is 1.96 minutes, or 1.97 per completed arrival. Delay rates and average taxi-out use only flights
  with a reported value.
- Delay-cause minutes are reported on arriving flights and may originate elsewhere in
  the network, so they are grouped by destination and kept apart from departure measures.
- The source also omits details such as slot limits, curfews and runway layouts.

**Not available in any source used**

- Gate counts, terminal size or terminal capacity.
- Observed unmet demand, airport revenue or investment returns.
