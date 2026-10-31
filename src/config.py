"""Paths, source URLs and fixed period definitions for the data layer."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Public sources. Each was fetched successfully before being listed here.
T100_AIRPORT_SUMMARY_URL = "https://data.bts.gov/resource/r495-tyji.json"
T100_NATIONAL_SUMMARY_URL = "https://data.bts.gov/resource/bu82-4pwz.json"
FAA_ENPLANEMENTS_URL = (
    "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/"
    "passenger/arp-cy2025-commercial-service-enplanements.xlsx"
)
ONTIME_URL_TEMPLATE = (
    "https://www.transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

T100_ROUTE_GLOB = "T_T100_SEGMENT*.csv"

# Comparable periods. 2026 is partial in every source, so it must never be
# compared with these full calendar years.
LATEST_COMPLETE_YEAR = 2025
BASE_YEAR_3Y = 2022
BASE_YEAR_10Y = 2015
HISTORY_YEARS = (BASE_YEAR_10Y, BASE_YEAR_3Y, LATEST_COMPLETE_YEAR)
ONTIME_YEAR = LATEST_COMPLETE_YEAR

NEW_ENGLAND_STATES = ("CT", "MA", "ME", "NH", "RI", "VT")

# Aircraft configuration codes used in the route file.
CONFIG_PASSENGER = 1
CONFIG_FREIGHT = 2
CLASS_SCHEDULED = "F"

# Long-haul has no single regulatory definition. This default is an assumption,
# so results are always reported alongside the sensitivity thresholds.
DEFAULT_LONG_HAUL_MILES = 3000
LONG_HAUL_SENSITIVITY_MILES = (1500, 2400, 3000, 4000)

# Provisional demo settings. None of these is a final methodology decision.
DEFAULT_MIN_ENPLANEMENTS = 100_000
SMALL_BASE_WARNING_PASSENGERS = 25_000
SCHEDULED_PASSENGER_WARNING_PCT = 95.0
PASSENGER_AIRCRAFT_WARNING_PCT = 80.0

# FAA location IDs that differ from the T-100 codes. Mapped using OurAirports,
# an unofficial source. RVR and AFE have no mapping and stay unmatched.
FAA_TO_T100_CODES = {
    "IWA": "AZA", "GPI": "FCA", "JQF": "USA", "GSN": "SPN", "UNV": "SCE", "HXD": "HHH",
    "CRQ": "CLD", "SAW": "MQT", "JKA": "GUF", "TNI": "TIQ", "AKW": "KLW", "FHR": "FRD",
    "ENM": "EMK", "GRO": "ROP", "SKX": "TSM", "ORS": "ESD", "AQH": "KWN", "W33": "FBS",
    "IIK": "KPN", "Z09": "KUK", "0AK": "PQS", "A61": "WTL", "CFK": "CYF", "2A9": "KOT",
    "AWI": "AIN", "Z13": "KKI", "BVK": "BKC", "D76": "ORV", "DUY": "KKH", "16A": "NUP",
    "MDM": "MLL", "HLA": "HSL", "4A2": "ATT", "GGV": "KWK",
}
