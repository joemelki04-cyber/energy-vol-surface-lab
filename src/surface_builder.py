from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iv_solver import implied_vol_black76


def load_inputs(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    futures = pd.read_csv(data_dir / "futures_prices.csv")
    quotes = pd.read_csv(data_dir / "option_quotes.csv")
    rates = pd.read_csv(data_dir / "rates.csv")
    return futures, quotes, rates


def prepare_inputs(
    futures: pd.DataFrame,
    quotes: pd.DataFrame,
    rates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    futures = futures.copy()
    quotes = quotes.copy()
    rates = rates.copy()

    futures["valuation_date"] = pd.to_datetime(futures["valuation_date"]).dt.normalize()
    quotes["valuation_date"] = pd.to_datetime(quotes["valuation_date"]).dt.normalize()
    quotes["expiry_date"] = pd.to_datetime(quotes["expiry_date"]).dt.normalize()
    rates["valuation_date"] = pd.to_datetime(rates["valuation_date"]).dt.normalize()

    quotes["option_type"] = quotes["option_type"].astype(str).str.upper().str.strip()
    if not quotes["option_type"].isin(["C", "P"]).all():
        bad = quotes.loc[~quotes["option_type"].isin(["C", "P"])]
        raise ValueError(f"Invalid option types found:\n{bad.to_string(index=False)}")

    return futures, quotes, rates


def interpolate_rate(tenor_days: float, curve: pd.DataFrame) -> float:
    curve = curve.sort_values("tenor_days")
    x = curve["tenor_days"].to_numpy(dtype=float)
    y = curve["rate"].to_numpy(dtype=float)

    if tenor_days <= x.min():
        return float(y[0])
    if tenor_days >= x.max():
        return float(y[-1])

    return float(np.interp(tenor_days, x, y))


def build_surface(
    futures: pd.DataFrame,
    quotes: pd.DataFrame,
    rates: pd.DataFrame,
    valuation_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valuation_date = pd.Timestamp(valuation_date).normalize()

    futures_slice = futures.loc[futures["valuation_date"] == valuation_date].copy()
    quotes_slice = quotes.loc[quotes["valuation_date"] == valuation_date].copy()
    rates_slice = rates.loc[rates["valuation_date"] == valuation_date].copy()

    if futures_slice.empty:
        raise ValueError(f"No futures prices found for {valuation_date.date()}")
    if quotes_slice.empty:
        raise ValueError(f"No option quotes found for {valuation_date.date()}")
    if rates_slice.empty:
        raise ValueError(f"No rate data found for {valuation_date.date()}")

    df = quotes_slice.merge(
        futures_slice,
        on=["valuation_date", "contract_month"],
        how="left",
        validate="many_to_one",
    )

    missing_futures = df.loc[df["futures_price"].isna(), ["contract_month"]].drop_duplicates()
    if not missing_futures.empty:
        raise ValueError(
            "Missing futures prices for contract months:\n"
            + missing_futures.to_string(index=False)
        )

    df["days_to_expiry"] = (df["expiry_date"] - df["valuation_date"]).dt.days
    if (df["days_to_expiry"] <= 0).any():
        bad = df.loc[df["days_to_expiry"] <= 0, ["expiry_date", "valuation_date"]]
        raise ValueError(f"Found expired or same-day options:\n{bad.to_string(index=False)}")

    df["T"] = df["days_to_expiry"] / 365.0
    df["rate"] = df["days_to_expiry"].apply(lambda d: interpolate_rate(d, rates_slice))
    df["log_moneyness"] = np.log(df["strike"] / df["futures_price"])
    df["moneyness_ratio"] = df["strike"] / df["futures_price"]

    ivs = []
    errors = []

    for row in df.itertuples(index=False):
        try:
            iv = implied_vol_black76(
                premium=float(row.premium),
                F=float(row.futures_price),
                K=float(row.strike),
                T=float(row.T),
                r=float(row.rate),
                option_type=str(row.option_type),
            )
            ivs.append(iv)
            errors.append("")
        except Exception as exc:
            ivs.append(np.nan)
            errors.append(str(exc))

    df["implied_vol"] = ivs
    df["solver_error"] = errors

    df["abs_log_moneyness"] = df["log_moneyness"].abs()

    atm_idx = (
        df.groupby(["valuation_date", "expiry_date", "contract_month"])["abs_log_moneyness"]
        .idxmin()
        .tolist()
    )
    df["is_atm_nearest"] = False
    df.loc[atm_idx, "is_atm_nearest"] = True

    atm_term = (
        df.loc[df["is_atm_nearest"]]
        .groupby(["valuation_date", "expiry_date", "contract_month"], as_index=False)
        .agg(
            futures_price=("futures_price", "first"),
            atm_strike=("strike", "first"),
            atm_implied_vol=("implied_vol", "mean"),
            days_to_expiry=("days_to_expiry", "first"),
            rate=("rate", "first"),
        )
        .sort_values("expiry_date")
    )

    ordered_cols = [
        "valuation_date",
        "expiry_date",
        "contract_month",
        "option_type",
        "strike",
        "premium",
        "futures_price",
        "days_to_expiry",
        "T",
        "rate",
        "moneyness_ratio",
        "log_moneyness",
        "implied_vol",
        "is_atm_nearest",
        "solver_error",
    ]

    surface = df[ordered_cols].sort_values(
        ["expiry_date", "contract_month", "option_type", "strike"]
    )

    return surface, atm_term


def write_outputs(
    surface: pd.DataFrame,
    atm_term: pd.DataFrame,
    output_dir: str | Path,
    valuation_date: pd.Timestamp,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    surface.to_csv(output_dir / f"iv_table_{valuation_date.date()}.csv", index=False)
    atm_term.to_csv(output_dir / f"atm_term_structure_{valuation_date.date()}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build implied vol table for energy futures options")
    parser.add_argument("--date", required=True, help="Valuation date, e.g. 2026-03-15")
    parser.add_argument("--data-dir", default="data/raw", help="Input CSV folder")
    parser.add_argument("--output-dir", default="reports", help="Output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    futures, quotes, rates = load_inputs(args.data_dir)
    futures, quotes, rates = prepare_inputs(futures, quotes, rates)

    valuation_date = pd.Timestamp(args.date).normalize()
    surface, atm_term = build_surface(
        futures=futures,
        quotes=quotes,
        rates=rates,
        valuation_date=valuation_date,
    )

    write_outputs(
        surface=surface,
        atm_term=atm_term,
        output_dir=args.output_dir,
        valuation_date=valuation_date,
    )

    solved = surface["implied_vol"].notna().sum()
    total = len(surface)

    print("\nEnergy Vol Surface Lab completed.")
    print(f"Valuation date:      {valuation_date.date()}")
    print(f"Quotes loaded:       {total}")
    print(f"IVs solved:          {solved}")
    print(f"ATM expiries built:  {len(atm_term)}")
    print("\nATM Term Structure")
    print(atm_term.to_string(index=False))


if __name__ == "__main__":
    main()
