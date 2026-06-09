from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from whoop_analytics.config import Settings
from whoop_analytics.api.auth import TokenManager
from whoop_analytics.api.client import WhoopClient
from whoop_analytics.pipeline.ingest import IngestPipeline
from whoop_analytics.pipeline.transform import build_daily_dataset
from whoop_analytics.pipeline.features import add_lag_features, add_rolling_features
from whoop_analytics.analysis.discovery import CausalDiscovery
from whoop_analytics.analysis.estimation import EffectEstimator, EffectResult
from whoop_analytics.report.generator import AnalysisReport, ReportGenerator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Whoop Causal Analytics")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Fetch data from Whoop API")
    ingest_parser.add_argument("--days", type=int, default=180, help="Days of history to fetch")

    analyze_parser = subparsers.add_parser("analyze", help="Run causal analysis")
    analyze_parser.add_argument("--target", default="brain_fog", help="Target variable")
    analyze_parser.add_argument("--max-lag", type=int, default=3, help="Maximum lag in days")
    analyze_parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")

    report_parser = subparsers.add_parser("report", help="Generate report from latest analysis")

    full_parser = subparsers.add_parser("run", help="Full pipeline: ingest → analyze → report")
    full_parser.add_argument("--days", type=int, default=180)
    full_parser.add_argument("--target", default="brain_fog")
    full_parser.add_argument("--max-lag", type=int, default=3)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    settings = Settings.from_env()

    if args.command == "ingest":
        return _cmd_ingest(settings, args.days)
    elif args.command == "analyze":
        return _cmd_analyze(settings, args.target, args.max_lag, args.alpha)
    elif args.command == "report":
        return _cmd_report(settings)
    elif args.command == "run":
        return _cmd_run(settings, args.days, args.target, args.max_lag)

    return 0


def _cmd_ingest(settings: Settings, days: int) -> int:
    token_manager = TokenManager(
        client_id=settings.client_id,
        client_secret=settings.client_secret,
        redirect_uri=settings.redirect_uri,
        token_file=settings.data_dir / ".tokens.json",
    )
    client = WhoopClient(token_manager=token_manager)
    pipeline = IngestPipeline(client=client, data_dir=settings.data_dir)

    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=days)).isoformat()

    print(f"Ingesting {days} days of data ({start_date} to {end_date})...")
    pipeline.ingest(start_date=start_date, end_date=end_date)
    print("Done.")
    return 0


def _cmd_analyze(settings: Settings, target: str, max_lag: int, alpha: float) -> int:
    print("Building daily dataset...")
    df = build_daily_dataset(settings.data_dir)

    if df.empty:
        print("No data found. Run 'ingest' first.", file=sys.stderr)
        return 1

    feature_cols = [c for c in df.columns if c != target]
    df = add_lag_features(df, columns=feature_cols, lags=[1, 2])
    df = add_rolling_features(df, columns=feature_cols, windows=[3, 7])
    df = df.dropna()

    print(f"Running causal discovery (target={target}, max_lag={max_lag})...")
    discovery = CausalDiscovery(max_lag=max_lag, significance_level=alpha)
    result = discovery.run(df, target=target)

    print(f"Found {len(result.links)} causal links.")
    for link in result.links:
        print(f"  {link.source} → {link.target} (lag={link.lag}, strength={link.strength:.3f})")

    return 0


def _cmd_report(settings: Settings) -> int:
    print("Generating report...")
    print("Report generation requires running 'run' command for full pipeline.")
    return 0


def _cmd_run(settings: Settings, days: int, target: str, max_lag: int) -> int:
    ret = _cmd_ingest(settings, days)
    if ret != 0:
        return ret

    print("\nBuilding daily dataset...")
    df = build_daily_dataset(settings.data_dir)

    if df.empty:
        print("No data after ingestion.", file=sys.stderr)
        return 1

    feature_cols = [c for c in df.columns if c != target]
    df = add_lag_features(df, columns=feature_cols, lags=[1, 2])
    df = add_rolling_features(df, columns=feature_cols, windows=[3, 7])
    df = df.dropna()

    if len(df) < 30:
        print(f"Only {len(df)} observations after feature engineering. Need at least 30.", file=sys.stderr)
        return 1

    print(f"\nRunning causal discovery (n={len(df)}, target={target}, max_lag={max_lag})...")
    discovery = CausalDiscovery(max_lag=max_lag, significance_level=0.05)
    disc_result = discovery.run(df, target=target)

    print(f"Found {len(disc_result.links)} causal links.")

    effects = []
    if disc_result.links:
        print("\nEstimating causal effects...")
        estimator = EffectEstimator()
        for link in disc_result.links:
            other_sources = [l.source for l in disc_result.links if l.source != link.source]
            effect = estimator.estimate(df=df, link=link, common_causes=other_sources)
            effects.append(effect)

    report = AnalysisReport(
        generated_date=date.today(),
        data_start=df.index[0].date(),
        data_end=df.index[-1].date(),
        n_observations=len(df),
        causal_links=disc_result.links,
        effects=effects,
    )

    generator = ReportGenerator()
    report_path = settings.data_dir / "reports" / f"report-{date.today().isoformat()}.md"
    generator.save(report, report_path)
    print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
