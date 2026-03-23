from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from surface_builder import (
    load_inputs,
    prepare_inputs,
    build_surface,
    write_outputs,
)


def _build_smile_figures(surface: pd.DataFrame) -> list[tuple[str, go.Figure]]:
    figs: list[tuple[str, go.Figure]] = []

    valid = surface.loc[surface["implied_vol"].notna()].copy()
    if valid.empty:
        return figs

    for expiry in sorted(valid["expiry_date"].dropna().unique()):
        expiry_df = valid.loc[valid["expiry_date"] == expiry].copy()
        expiry_label = pd.Timestamp(expiry).date()

        fig = go.Figure()

        for option_type in ["C", "P"]:
            sub = expiry_df.loc[expiry_df["option_type"] == option_type].sort_values("strike")
            if sub.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=sub["strike"],
                    y=sub["implied_vol"],
                    mode="lines+markers",
                    name=f"{option_type} - {sub['contract_month'].iloc[0]}",
                    text=[
                        f"Strike: {row.strike}<br>"
                        f"Premium: {row.premium:.4f}<br>"
                        f"Futures: {row.futures_price:.4f}<br>"
                        f"IV: {row.implied_vol:.4%}"
                        for row in sub.itertuples(index=False)
                    ],
                    hoverinfo="text",
                )
            )

        fig.update_layout(
            title=f"Implied Vol Smile - Expiry {expiry_label}",
            template="plotly_white",
            xaxis_title="Strike",
            yaxis_title="Implied Vol",
        )
        figs.append((str(expiry_label), fig))

    return figs


def _build_term_structure_figure(atm_term: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if not atm_term.empty:
        atm_term = atm_term.sort_values("expiry_date").copy()

        fig.add_trace(
            go.Scatter(
                x=atm_term["expiry_date"],
                y=atm_term["atm_implied_vol"],
                mode="lines+markers",
                text=[
                    f"Expiry: {pd.Timestamp(row.expiry_date).date()}<br>"
                    f"Contract: {row.contract_month}<br>"
                    f"ATM Strike: {row.atm_strike}<br>"
                    f"Futures: {row.futures_price:.4f}<br>"
                    f"ATM IV: {row.atm_implied_vol:.4%}"
                    for row in atm_term.itertuples(index=False)
                ],
                hoverinfo="text",
                name="ATM IV",
            )
        )

    fig.update_layout(
        title="ATM Implied Vol Term Structure",
        template="plotly_white",
        xaxis_title="Expiry Date",
        yaxis_title="ATM Implied Vol",
        showlegend=False,
    )
    return fig


def _build_surface_scatter(surface: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    valid = surface.loc[surface["implied_vol"].notna()].copy()
    if not valid.empty:
        valid = valid.sort_values(["expiry_date", "strike"])

        fig.add_trace(
            go.Scatter(
                x=valid["strike"],
                y=valid["implied_vol"],
                mode="markers",
                text=[
                    f"Expiry: {pd.Timestamp(row.expiry_date).date()}<br>"
                    f"Contract: {row.contract_month}<br>"
                    f"Type: {row.option_type}<br>"
                    f"Strike: {row.strike}<br>"
                    f"Futures: {row.futures_price:.4f}<br>"
                    f"IV: {row.implied_vol:.4%}"
                    for row in valid.itertuples(index=False)
                ],
                hoverinfo="text",
                name="Surface Points",
            )
        )

    fig.update_layout(
        title="All Surface Points",
        template="plotly_white",
        xaxis_title="Strike",
        yaxis_title="Implied Vol",
        showlegend=False,
    )
    return fig


def _build_market_snapshot(
    surface: pd.DataFrame,
    atm_term: pd.DataFrame,
    valuation_date: pd.Timestamp,
) -> pd.DataFrame:
    solved = int(surface["implied_vol"].notna().sum())
    total_quotes = int(len(surface))
    expiries = int(surface["expiry_date"].nunique())
    contracts = int(surface["contract_month"].nunique())
    avg_atm_iv = float(atm_term["atm_implied_vol"].mean()) if not atm_term.empty else float("nan")

    futures_summary = (
        surface[["contract_month", "futures_price"]]
        .drop_duplicates()
        .sort_values("contract_month")
    )

    snapshot_rows = [
        {"metric": "valuation_date", "value": str(valuation_date.date())},
        {"metric": "quotes_loaded", "value": total_quotes},
        {"metric": "ivs_solved", "value": solved},
        {"metric": "unique_expiries", "value": expiries},
        {"metric": "unique_contract_months", "value": contracts},
        {
            "metric": "average_atm_implied_vol",
            "value": f"{avg_atm_iv:.4%}" if pd.notna(avg_atm_iv) else "N/A",
        },
    ]

    for row in futures_summary.itertuples(index=False):
        snapshot_rows.append(
            {
                "metric": f"futures_price_{row.contract_month}",
                "value": f"{row.futures_price:.4f}",
            }
        )

    return pd.DataFrame(snapshot_rows)


def _html_table(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    df = df.copy()

    if float_cols:
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].map(lambda x: f"{x:,.6f}" if pd.notna(x) else "")

    return df.to_html(index=False, escape=False)


def write_html_report(
    surface: pd.DataFrame,
    atm_term: pd.DataFrame,
    output_dir: str | Path,
    valuation_date: pd.Timestamp,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / f"vol_report_{valuation_date.date()}.html"

    snapshot = _build_market_snapshot(surface, atm_term, valuation_date)

    smile_figs = _build_smile_figures(surface)
    term_fig = _build_term_structure_figure(atm_term)
    surface_fig = _build_surface_scatter(surface)

    top_rows = (
        surface.loc[surface["implied_vol"].notna()]
        .sort_values(["expiry_date", "option_type", "strike"])
        .head(20)
        .copy()
    )

    failed_rows = surface.loc[surface["solver_error"].astype(str).str.len() > 0].copy()
    if failed_rows.empty:
        failed_rows = pd.DataFrame([{"message": "No solver errors."}])

    smile_sections = ""
    for expiry_label, fig in smile_figs:
        smile_sections += f"""
        <h3>Expiry {expiry_label}</h3>
        {fig.to_html(full_html=False, include_plotlyjs=False)}
        """

    html = f"""
    <html>
      <head>
        <title>Energy Vol Surface Report - {valuation_date.date()}</title>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>
          body {{
            font-family: Arial, sans-serif;
            margin: 32px;
            line-height: 1.4;
          }}
          h1, h2, h3 {{
            margin-top: 28px;
          }}
          table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 12px;
            margin-bottom: 20px;
          }}
          th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
          }}
          th {{
            background-color: #f5f5f5;
          }}
          .meta {{
            margin-bottom: 20px;
          }}
          .chart {{
            margin-bottom: 30px;
          }}
        </style>
      </head>
      <body>
        <h1>Energy Vol Surface Lab Report</h1>

        <div class="meta">
          <p><strong>Valuation date:</strong> {valuation_date.date()}</p>
          <p><strong>Report scope:</strong> Black-76 implied vol surface, smiles, and ATM term structure</p>
        </div>

        <h2>Market Snapshot</h2>
        {snapshot.to_html(index=False, escape=False)}

        <h2>ATM Term Structure</h2>
        <div class="chart">
          {term_fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>

        <h2>All Surface Points</h2>
        <div class="chart">
          {surface_fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>

        <h2>Smile by Expiry</h2>
        {smile_sections if smile_sections else "<p>No valid smile data available.</p>"}

        <h2>ATM Term Table</h2>
        {_html_table(
            atm_term,
            float_cols=["futures_price", "atm_strike", "atm_implied_vol", "rate"]
        )}

        <h2>Sample Surface Rows</h2>
        {_html_table(
            top_rows,
            float_cols=["strike", "premium", "futures_price", "T", "rate", "moneyness_ratio", "log_moneyness", "implied_vol"]
        )}

        <h2>Solver Errors</h2>
        {_html_table(failed_rows)}
      </body>
    </html>
    """

    html_path.write_text(html, encoding="utf-8")
    return html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HTML report for energy vol surface lab")
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

    html_path = write_html_report(
        surface=surface,
        atm_term=atm_term,
        output_dir=args.output_dir,
        valuation_date=valuation_date,
    )

    print("\nEnergy Vol Surface report completed.")
    print(f"Valuation date: {valuation_date.date()}")
    print(f"HTML report:    {html_path}")


if __name__ == "__main__":
    main()
