from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

from whoop_analytics.analysis.discovery import DiscoveryResult, CausalLink
from whoop_analytics.analysis.estimation import EffectResult


def generate_html_report(
    daily_df: pd.DataFrame,
    discovery_result: DiscoveryResult | None,
    effects: list[EffectResult],
    output_path: Path,
) -> None:
    sections = [
        _header_section(daily_df),
        _hrv_section(daily_df),
        _sleep_section(daily_df),
        _brain_fog_section(daily_df),
    ]

    if discovery_result and discovery_result.links:
        sections.append(_causal_graph_section(discovery_result))
        sections.append(_links_table_section(discovery_result))

    if effects:
        sections.append(_effects_section(effects))

    html = _wrap_html("\n".join(sections))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def _wrap_html(body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Whoop Causal Analytics</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px 40px; background: #fafafa; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}
  h2 {{ color: #16213e; margin-top: 40px; }}
  .chart {{ margin: 20px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #2196F3; color: white; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .effect-card {{ background: white; border-radius: 8px; padding: 16px; margin: 12px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .robust {{ border-left: 4px solid #4CAF50; }}
  .uncertain {{ border-left: 4px solid #F44336; }}
  .meta {{ color: #666; font-size: 0.9em; }}
</style>
</head>
<body>
{body}
<footer style="margin-top:40px;padding-top:20px;border-top:1px solid #ddd;color:#999;font-size:0.8em;">
Generated {date.today().isoformat()} by whoop-analytics
</footer>
</body>
</html>"""


def _header_section(df: pd.DataFrame) -> str:
    if df.empty:
        return "<h1>Whoop Causal Analytics</h1><p>No data available.</p>"
    n = len(df)
    start = df.index[0].strftime("%Y-%m-%d")
    end = df.index[-1].strftime("%Y-%m-%d")
    return f"""<h1>Whoop Causal Analytics</h1>
<p class="meta">{n} observations &middot; {start} to {end}</p>"""


def _hrv_section(df: pd.DataFrame) -> str:
    if df.empty or "hrv_rmssd" not in df.columns:
        return ""

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df.index, y=df["hrv_rmssd"], name="HRV (rMSSD)", line=dict(color="#2196F3")),
        secondary_y=False,
    )
    if "resting_hr" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["resting_hr"], name="Resting HR", line=dict(color="#FF5722")),
            secondary_y=True,
        )
    fig.update_layout(title="HRV & Resting Heart Rate", height=350, template="plotly_white")
    fig.update_yaxes(title_text="HRV (ms)", secondary_y=False)
    fig.update_yaxes(title_text="HR (bpm)", secondary_y=True)

    return f'<h2>Heart Rate Variability</h2>\n<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'


def _sleep_section(df: pd.DataFrame) -> str:
    stage_cols = [c for c in ["sws_minutes", "rem_minutes", "light_minutes"] if c in df.columns]
    if not stage_cols:
        return ""

    fig = go.Figure()
    colors = {"sws_minutes": "#3F51B5", "rem_minutes": "#9C27B0", "light_minutes": "#00BCD4"}
    labels = {"sws_minutes": "Deep (SWS)", "rem_minutes": "REM", "light_minutes": "Light"}

    for col in stage_cols:
        fig.add_trace(go.Bar(x=df.index, y=df[col], name=labels[col], marker_color=colors[col]))

    fig.update_layout(title="Sleep Stages", barmode="stack", height=350, template="plotly_white")
    fig.update_yaxes(title_text="Minutes")

    return f'<h2>Sleep Architecture</h2>\n<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'


def _brain_fog_section(df: pd.DataFrame) -> str:
    if "brain_fog" not in df.columns:
        return ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["brain_fog"], name="Brain Fog",
        mode="lines+markers", line=dict(color="#F44336"), marker=dict(size=5),
    ))
    fig.update_layout(title="Brain Fog Score", height=300, template="plotly_white")
    fig.update_yaxes(title_text="Score (1-5)")

    return f'<h2>Brain Fog</h2>\n<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'


def _causal_graph_section(result: DiscoveryResult) -> str:
    G = nx.DiGraph()
    for link in result.links:
        G.add_edge(link.source, link.target, weight=abs(link.strength), lag=link.lag)

    pos = nx.spring_layout(G, seed=42, k=2)

    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(1, data["weight"] * 5), color="#888"),
            hoverinfo="text", text=f"{u} → {v} (lag={data['lag']})",
            showlegend=False,
        ))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text", text=list(G.nodes()), textposition="top center",
        marker=dict(size=20, color="#2196F3", line=dict(width=2, color="#1565C0")),
        hoverinfo="text", showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Discovered Causal Structure",
        height=450, template="plotly_white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    return f'<h2>Causal Graph</h2>\n<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'


def _links_table_section(result: DiscoveryResult) -> str:
    rows = ""
    for link in result.links:
        rows += f"<tr><td>{link.source}</td><td>{link.target}</td><td>{link.lag}</td><td>{link.strength:.3f}</td><td>{link.p_value:.4f}</td></tr>\n"

    return f"""<h2>Causal Links</h2>
<table>
<tr><th>Source</th><th>Target</th><th>Lag (days)</th><th>Strength</th><th>p-value</th></tr>
{rows}</table>"""


def _effects_section(effects: list[EffectResult]) -> str:
    sources = [e.source for e in effects]
    ates = [e.ate for e in effects]
    colors = ["#4CAF50" if e.refutation_passed else "#F44336" for e in effects]

    fig = go.Figure(go.Bar(
        x=ates, y=sources, orientation="h", marker_color=colors,
        text=[f"{a:.3f}" for a in ates], textposition="outside",
    ))
    fig.update_layout(
        title="Average Treatment Effects on Brain Fog",
        xaxis_title="ATE (negative = reduces fog)",
        height=max(250, len(effects) * 50 + 100),
        template="plotly_white",
    )

    chart = f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'

    cards = ""
    for e in effects:
        cls = "robust" if e.refutation_passed else "uncertain"
        status = "Robust" if e.refutation_passed else "Failed refutation"
        p_str = f"{e.refutation_p_value:.3f}" if e.refutation_p_value is not None else "N/A"
        cards += f"""<div class="effect-card {cls}">
<strong>{e.source} &rarr; {e.target}</strong><br>
ATE: {e.ate:.4f} &middot; Refutation: {status} (p={p_str}) &middot; Method: {e.method}
</div>\n"""

    return f"<h2>Effect Estimates</h2>\n{chart}\n{cards}"
