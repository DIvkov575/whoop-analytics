from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from whoop_analytics.analysis.estimation import EffectResult


def render(effects: list[EffectResult]) -> None:
    st.header("Effect Estimates")

    if not effects:
        st.warning("No effect estimates available. Run analysis first.")
        return

    _render_summary_chart(effects)
    _render_details(effects)


def _render_summary_chart(effects: list[EffectResult]) -> None:
    sources = [e.source for e in effects]
    ates = [e.ate for e in effects]
    colors = ["#4CAF50" if e.refutation_passed else "#F44336" for e in effects]

    fig = go.Figure(go.Bar(
        x=ates,
        y=sources,
        orientation="h",
        marker_color=colors,
        text=[f"{a:.3f}" for a in ates],
        textposition="outside",
    ))
    fig.update_layout(
        title="Average Treatment Effects on Brain Fog",
        xaxis_title="ATE (negative = reduces fog)",
        height=max(250, len(effects) * 50 + 100),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Green = passed refutation test (robust). Red = failed refutation (uncertain).")


def _render_details(effects: list[EffectResult]) -> None:
    st.subheader("Detailed Results")

    for effect in effects:
        status_icon = "pass" if effect.refutation_passed else "fail"
        with st.expander(f"[{status_icon}] {effect.source} -> {effect.target}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("ATE", f"{effect.ate:.4f}")
            col2.metric("Refutation p-value", f"{effect.refutation_p_value:.3f}" if effect.refutation_p_value else "N/A")
            col3.metric("Method", effect.method.split(".")[-1])

            if effect.confidence_interval:
                st.write(f"95% CI: [{effect.confidence_interval[0]:.3f}, {effect.confidence_interval[1]:.3f}]")

            if effect.refutation_passed:
                st.success("This effect passed robustness checks.")
            else:
                st.error("This effect did NOT pass refutation. Treat with caution.")
