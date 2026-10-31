"""Data ingestion: fetch, validate and snapshot public aviation data.

Run with: python -m src.pipeline [--skip-ontime]

This module loads and checks data only. It does not compute investment metrics
or scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from . import config

HEADERS = {"User-Agent": "airport-investment-intelligence/0.1"}
TIMEOUT_SECONDS = 120
SOCRATA_PAGE_SIZE = 50_000

# Socrata truncates long column names, so the numbered names below are mapped
# using the labels published in the dataset metadata.
SUMMARY_COLUMNS = {
    "origin_airport_code": "airport",
    "origin_airport_name": "airport_name",
    "reporting_month": "reporting_month",
    "total_departures": "departures",
    "total_passengers": "passengers",
    "total_seats": "seats",
    "total_freight_lbs": "freight_lbs",
    "domestic_departures": "domestic_departures",
    "domestic_passengers": "domestic_passengers",
    "domestic_seats": "domestic_seats",
    "outbound_international": "outbound_intl_departures",
    "outbound_international_1": "outbound_intl_passengers",
    "outbound_international_seats": "outbound_intl_seats",
    "inbound_international": "inbound_intl_departures",
    "inbound_international_1": "inbound_intl_passengers",
}
SUMMARY_NUMERIC = [
    c for c in SUMMARY_COLUMNS.values() if c not in ("airport", "airport_name", "reporting_month")
]

NATIONAL_COLUMNS = {
    "date": "reporting_month",
    "departures": "departures",
    "passengers": "passengers",
    "domestic_departures": "domestic_departures",
    "outbound_international": "outbound_intl_departures",
    "inbound_international": "inbound_intl_departures",
}

ROUTE_COLUMNS = {
    "YEAR": "int16",
    "MONTH": "int8",
    "ORIGIN": "string",
    "ORIGIN_COUNTRY": "string",
    "DEST": "string",
    "DEST_COUNTRY": "string",
    "DISTANCE": "float64",
    "DEPARTURES_SCHEDULED": "float64",
    "DEPARTURES_PERFORMED": "float64",
    "SEATS": "float64",
    "PASSENGERS": "float64",
    "FREIGHT": "float64",
    "UNIQUE_CARRIER": "string",
    "UNIQUE_CARRIER_NAME": "string",
    "AIRCRAFT_CONFIG": "int16",
    "CLASS": "string",
    "DATA_SOURCE": "string",
}

ONTIME_COLUMNS = [
    "Year", "Month", "Origin", "Dest", "Cancelled", "DepDel15", "TaxiOut", "ArrDel15",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]
DELAY_CAUSES = {
    "CarrierDelay": "carrier_delay_min",
    "WeatherDelay": "weather_delay_min",
    "NASDelay": "nas_delay_min",
    "SecurityDelay": "security_delay_min",
    "LateAircraftDelay": "late_aircraft_delay_min",
}


class DataValidationError(ValueError):
    """Raised when a source fails a check the analysis relies on."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_socrata(url: str, select_columns: list[str], order: str) -> list[dict]:
    """Fetch every row of a Socrata dataset, one page at a time."""
    rows: list[dict] = []
    offset = 0
    while True:
        response = requests.get(
            url,
            params={
                "$select": ",".join(select_columns),
                "$order": order,
                "$limit": SOCRATA_PAGE_SIZE,
                "$offset": offset,
            },
            headers=HEADERS,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        page = response.json()
        rows.extend(page)
        if len(page) < SOCRATA_PAGE_SIZE:
            return rows
        offset += SOCRATA_PAGE_SIZE


def normalize_airport_summary(rows: list[dict]) -> pd.DataFrame:
    """Convert raw summary rows into one typed row per airport and month."""
    frame = pd.DataFrame(rows).reindex(columns=list(SUMMARY_COLUMNS)).rename(columns=SUMMARY_COLUMNS)
    frame["reporting_month"] = pd.to_datetime(frame["reporting_month"])
    frame["year"] = frame["reporting_month"].dt.year.astype("int16")
    frame["month"] = frame["reporting_month"].dt.month.astype("int8")
    for column in SUMMARY_NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    # Socrata omits a field when its value is absent. A blank domestic or
    # international component means no such traffic; blank totals stay missing.
    components = [c for c in SUMMARY_NUMERIC if c.startswith(("domestic_", "outbound_", "inbound_"))]
    frame[components] = frame[components].fillna(0)

    validate_airport_summary(frame)
    return frame


def validate_airport_summary(frame: pd.DataFrame) -> None:
    duplicates = frame.duplicated(["airport", "reporting_month"]).sum()
    if duplicates:
        raise DataValidationError(f"{duplicates} duplicate airport-month rows in T-100 summary")
    measures = ["departures", "passengers", "seats"]
    if (frame[measures].fillna(0) < 0).any().any():
        raise DataValidationError("Negative values in T-100 summary")


def reconcile_counting(airport_summary: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    """Check how international flights are counted at each level, by year.

    National totals include domestic, outbound and inbound international flights.
    Airport totals should include domestic and outbound only, so that each flight
    is counted once, at its origin. Inbound fields must never be added to them.
    """
    airport = airport_summary.groupby("year").agg(
        airport_departures=("departures", "sum"),
        airport_passengers=("passengers", "sum"),
        months=("reporting_month", "nunique"),
    )
    national = national.assign(year=national["reporting_month"].dt.year)
    nat = national.groupby("year")[
        ["departures", "domestic_departures", "outbound_intl_departures", "inbound_intl_departures"]
    ].sum().add_prefix("national_")

    out = airport.join(nat, how="inner")
    out["airport_minus_dom_plus_outbound"] = (
        out["airport_departures"]
        - out["national_domestic_departures"]
        - out["national_outbound_intl_departures"]
    )
    out["national_minus_all_components"] = (
        out["national_departures"]
        - out["national_domestic_departures"]
        - out["national_outbound_intl_departures"]
        - out["national_inbound_intl_departures"]
    )
    return out.reset_index()


def airport_row_counting_errors(airport_summary: pd.DataFrame) -> int:
    """Count airport-months whose total differs from domestic plus outbound."""
    expected = airport_summary["domestic_departures"] + airport_summary["outbound_intl_departures"]
    return int(((airport_summary["departures"] - expected).abs() > 0.5).sum())


def history_coverage(airport_summary: pd.DataFrame, years=config.HISTORY_YEARS) -> pd.DataFrame:
    """Report, per airport, whether each required year is complete and usable.

    A year is complete only with 12 reported months and positive annual
    departures, passengers and seats. Nothing is imputed.
    """
    subset = airport_summary[airport_summary["year"].isin(years)]
    annual = subset.groupby(["airport", "year"]).agg(
        months=("reporting_month", "nunique"),
        departures=("departures", "sum"),
        passengers=("passengers", "sum"),
        seats=("seats", "sum"),
    )
    annual["complete"] = (
        (annual["months"] == 12)
        & (annual["departures"] > 0)
        & (annual["passengers"] > 0)
        & (annual["seats"] > 0)
    )
    wide = annual["complete"].unstack("year").reindex(columns=list(years)).astype("boolean").fillna(False)
    wide.columns = [f"complete_{year}" for year in wide.columns]
    wide["history_complete"] = wide.all(axis=1)
    return wide.reset_index()


def load_route_file(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load the route-level T-100 file and describe what it actually covers."""
    full = pd.read_csv(path, low_memory=False)
    missing = sorted(set(ROUTE_COLUMNS) - set(full.columns))
    if missing:
        raise DataValidationError(f"Route file is missing columns: {missing}")

    exact_duplicates = int(full.duplicated().sum())
    routes = full[list(ROUTE_COLUMNS)].astype(ROUTE_COLUMNS)
    del full

    numeric = ["DISTANCE", "DEPARTURES_PERFORMED", "SEATS", "PASSENGERS"]
    if (routes[numeric] < 0).any().any():
        raise DataValidationError("Negative values in route file")
    if routes[numeric].isna().any().any():
        raise DataValidationError("Missing values in required route-file measures")

    periods = routes[["YEAR", "MONTH"]].drop_duplicates().sort_values(["YEAR", "MONTH"])
    profile = {
        "rows": int(len(routes)),
        "exact_duplicate_rows": exact_duplicates,
        "first_period": f"{periods.iloc[0].YEAR}-{periods.iloc[0].MONTH:02d}",
        "last_period": f"{periods.iloc[-1].YEAR}-{periods.iloc[-1].MONTH:02d}",
        "months": int(len(periods)),
        "origin_airports": int(routes["ORIGIN"].nunique()),
        "carriers": int(routes["UNIQUE_CARRIER"].nunique()),
        "data_sources": sorted(routes["DATA_SOURCE"].unique().tolist()),
        "rows_with_zero_departures": int((routes["DEPARTURES_PERFORMED"] == 0).sum()),
        "performed_rows_with_zero_distance": int(
            ((routes["DEPARTURES_PERFORMED"] > 0) & (routes["DISTANCE"] == 0)).sum()
        ),
    }
    return routes, profile


def parse_faa_enplanements(path: Path) -> pd.DataFrame:
    """Parse the FAA commercial-service table, dropping any non-airport rows."""
    raw = pd.read_excel(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    required = ["RO", "ST", "Locid", "City", "Airport Name", "S/L", "Hub",
                "CY 25 Enplanements", "CY 24 Enplanements"]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise DataValidationError(f"FAA file is missing columns: {missing}")
    frame = raw[required].rename(columns={
        "RO": "faa_region", "ST": "state", "Locid": "airport", "City": "city",
        "Airport Name": "airport_name", "S/L": "service_level", "Hub": "hub",
        "CY 25 Enplanements": "enplanements_2025", "CY 24 Enplanements": "enplanements_2024",
    })
    frame = frame[frame["airport"].notna() & frame["state"].notna()].copy()
    frame["airport"] = frame["airport"].astype(str).str.strip()
    for column in ("enplanements_2025", "enplanements_2024"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["airport"].duplicated().any():
        raise DataValidationError("Duplicate airport codes in FAA enplanement file")
    return frame.reset_index(drop=True)


def aggregate_ontime_month(zip_path: Path) -> pd.DataFrame:
    """Aggregate one on-time ZIP to airport level, reading only needed columns.

    Departure measures group by origin. Delay-cause minutes are reported on
    arriving flights, so they group by destination and are kept apart.
    """
    with zipfile.ZipFile(zip_path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(name) as handle:
            flights = pd.read_csv(handle, usecols=ONTIME_COLUMNS, low_memory=False)
    return aggregate_ontime_flights(flights)


def aggregate_ontime_flights(flights: pd.DataFrame) -> pd.DataFrame:
    departures = flights.groupby(["Year", "Month", "Origin"]).agg(
        departures=("Origin", "size"),
        cancelled=("Cancelled", "sum"),
        dep_del15=("DepDel15", "sum"),
        dep_del15_reported=("DepDel15", "count"),
        taxi_out_min=("TaxiOut", "sum"),
        taxi_out_reported=("TaxiOut", "count"),
    )
    departures.index = departures.index.set_names(["year", "month", "airport"])

    named = {source: (source, "sum") for source in DELAY_CAUSES}
    arrivals = flights.groupby(["Year", "Month", "Dest"]).agg(
        arrivals=("Dest", "size"),
        arr_del15=("ArrDel15", "sum"),
        arr_del15_reported=("ArrDel15", "count"),
        **named,
    ).rename(columns=DELAY_CAUSES)
    arrivals.index = arrivals.index.set_names(["year", "month", "airport"])

    return departures.join(arrivals, how="outer").fillna(0).reset_index()


def download(url: str, destination: Path) -> Path:
    """Download a file once; an existing file is reused."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True, timeout=TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                handle.write(block)
    temporary.rename(destination)
    return destination


def build_ontime(year: int, manifest: dict) -> pd.DataFrame:
    frames = []
    for month in range(1, 13):
        url = config.ONTIME_URL_TEMPLATE.format(year=year, month=month)
        path = download(url, config.RAW_DIR / "ontime" / f"ontime_{year}_{month:02d}.zip")
        frames.append(aggregate_ontime_month(path))
        print(f"  on-time {year}-{month:02d} aggregated", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    manifest["sources"]["ontime"] = {
        "url_template": config.ONTIME_URL_TEMPLATE,
        "period": f"{year}-01 to {year}-12",
        "months": 12,
        "airports": int(frame["airport"].nunique()),
    }
    return frame


def run(skip_ontime: bool = False) -> dict:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest: dict = {"retrieved_at": retrieved_at, "sources": {}, "validation": {}}

    print("Fetching T-100 airport summary", flush=True)
    airport_rows = fetch_socrata(
        config.T100_AIRPORT_SUMMARY_URL, list(SUMMARY_COLUMNS), "origin_airport_id,reporting_month"
    )
    summary = normalize_airport_summary(airport_rows)

    print("Fetching T-100 national summary", flush=True)
    national_rows = fetch_socrata(config.T100_NATIONAL_SUMMARY_URL, list(NATIONAL_COLUMNS), "date")
    national = pd.DataFrame(national_rows).rename(columns=NATIONAL_COLUMNS)
    national["reporting_month"] = pd.to_datetime(national["reporting_month"])
    for column in NATIONAL_COLUMNS.values():
        if column != "reporting_month":
            national[column] = pd.to_numeric(national[column])

    print("Loading T-100 route file", flush=True)
    route_files = sorted(config.RAW_DIR.glob(config.T100_ROUTE_GLOB))
    if len(route_files) != 1:
        raise DataValidationError(f"Expected one route file, found {[p.name for p in route_files]}")
    routes, route_profile = load_route_file(route_files[0])

    print("Loading FAA enplanements", flush=True)
    faa_path = download(config.FAA_ENPLANEMENTS_URL, config.RAW_DIR / "faa_cy2025_enplanements.xlsx")
    airports = parse_faa_enplanements(faa_path)

    reconciliation = reconcile_counting(summary, national)
    coverage = history_coverage(summary)

    summary.to_parquet(config.PROCESSED_DIR / "t100_airport_month.parquet", index=False)
    national.to_parquet(config.PROCESSED_DIR / "t100_national_month.parquet", index=False)
    routes.to_parquet(config.PROCESSED_DIR / "t100_routes.parquet", index=False)
    airports.to_parquet(config.PROCESSED_DIR / "faa_airports.parquet", index=False)
    reconciliation.to_parquet(config.PROCESSED_DIR / "counting_reconciliation.parquet", index=False)
    coverage.to_parquet(config.PROCESSED_DIR / "history_coverage.parquet", index=False)

    manifest["sources"].update({
        "t100_airport_summary": {
            "url": config.T100_AIRPORT_SUMMARY_URL,
            "rows": int(len(summary)),
            "first_month": str(summary["reporting_month"].min().date()),
            "last_month": str(summary["reporting_month"].max().date()),
            "airports": int(summary["airport"].nunique()),
        },
        "t100_national_summary": {
            "url": config.T100_NATIONAL_SUMMARY_URL,
            "rows": int(len(national)),
            "last_month": str(national["reporting_month"].max().date()),
        },
        "t100_route_file": {
            "file": route_files[0].name,
            "size_bytes": route_files[0].stat().st_size,
            "sha256": sha256_file(route_files[0]),
            **route_profile,
        },
        "faa_enplanements": {
            "url": config.FAA_ENPLANEMENTS_URL,
            "rows": int(len(airports)),
            "sha256": sha256_file(faa_path),
        },
    })
    manifest["validation"]["airport_months_where_total_differs_from_dom_plus_outbound"] = (
        airport_row_counting_errors(summary)
    )

    if not skip_ontime:
        print("Building on-time aggregates", flush=True)
        ontime = build_ontime(config.ONTIME_YEAR, manifest)
        ontime.to_parquet(config.PROCESSED_DIR / "ontime_airport_month.parquet", index=False)

    (config.PROCESSED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Done", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build processed data snapshots.")
    parser.add_argument("--skip-ontime", action="store_true", help="Skip the on-time downloads.")
    run(skip_ontime=parser.parse_args().skip_ontime)
