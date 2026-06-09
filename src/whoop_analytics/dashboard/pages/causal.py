from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import networkx as nx

from whoop_analytics.analysis.discovery import DiscoveryResult


def render(discovery_result: DiscoveryResult | None) -> None:
    st.header("Causal Graph")

    if discovery_result is None:
        st.warning("No analysis results. Run analysis from the sidebar first.")
        return

    if not discovery_result.links:
        st.info("No significant causal links found at the current significance level.")
        return

    _render_graph(discovery_result)
    _render_links_table(discovery_result)


def _render_graph(result: DiscoveryResult) -> None:
    G = nx.DiGraph()

    for link in result.links:
        G.add_edge(
            link.source,
            link.target,
            weight=abs(link.strength),
            lag=link.lag,
            label=f"lag={link.lag}, str={link.strength:.2f}",
        )

    pos = nx.spring_layout(G, seed=42, k=2)

    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(1, abs(edge[2]["weight"]) * 5), color="#888"),
            hoverinfo="text",
            text=edge[2]["label"],
            showlegend=False,
        ))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = list(G.nodes())

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(size=20, color="#2196F3", line=dict(width=2, color="#1565C0")),
        hoverinfo="text",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Discovered Causal Structure",
        showlegend=False,
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_links_table(result: DiscoveryResult) -> None:
    st.subheader("Causal Links")
    rows = []
    for link in result.links:
        rows.append({
            "Source": link.source,
            "Target": link.target,
            "Lag (days)": link.lag,
            "Strength": f"{link.strength:.3f}",
            "p-value": f"{link.p_value:.4f}",
        })
    st.dataframe(rows, use_container_width=True)
