"""Shared Plotly chart style (D-017, D-018), used by the notebooks and the Streamlit app.

The palette is the validated reference palette (dataviz skill, light mode):
series 1 blue, series 2 orange; blue<->red diverging pair; recessive grey chrome.
Validated: blue/orange and blue/red pass lightness, chroma, CVD (>=21 dE) and contrast checks.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"        # series 1 / primary model / headline bars
ORANGE = "#eb6834"      # series 2 / challenger
RED = "#e34948"         # diverging negative pole
BAR_BLUE = "#4F97FB"    # headline bar colour from the reference images
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def style(fig: go.Figure, title: str | None = None, height: int = 400,
          showlegend: bool | None = None) -> go.Figure:
    """Apply the house style: light surface, hairline dashed grid, muted axes, no clutter."""
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)) if title else None,
        height=height,
        font=dict(family=FONT, size=12, color=INK_2),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        margin=dict(l=60, r=30, t=60 if title else 20, b=50),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
    )
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    axis = dict(gridcolor=GRID, griddash="dash", linecolor=AXIS, zeroline=False,
                tickfont=dict(color=MUTED), title_font=dict(color=INK_2))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


AQUA = "#1baf7a"        # series 3 (segment colours)
YELLOW = "#eda100"      # series 4
SEGMENT_COLORS = (BLUE, ORANGE, AQUA, YELLOW)   # fixed order = config.SEGMENTS (colour follows the segment)
WHITE = "#ffffff"
PROTOTYPE_NOTE = "Prototype results based on stated, illustrative assumptions; not bank performance."


def reference_bars(x, y, money: bool = False, height: int = 360, text: bool = True,
                   hover_label: str = "") -> go.Figure:
    """Headline bar chart in the Screen 1 / Screen 3 reference style (D-017, D-038): one blue,
    rounded tops, dashed light-grey horizontal grid, no legend, plain category labels, white
    background, auto-scaled y-axis (counts, or £K with a £ prefix)."""
    vals = list(y)
    fmt = (lambda v: f"£{v / 1e3:,.1f}K") if money else (lambda v: f"{v:,.0f}")
    fig = go.Figure(go.Bar(
        x=list(x), y=vals, marker=dict(color=BAR_BLUE, cornerradius=8, line=dict(width=0)),
        text=[fmt(v) for v in vals] if text else None, textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_2, size=12),
        hovertemplate="%{x}<br>" + ("£%{y:,.0f}" if money else "%{y:,.0f}") + (f" {hover_label}" if hover_label else "")
                      + "<extra></extra>"))
    fig.update_layout(height=height, showlegend=False, bargap=0.38, plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                      font=dict(family=FONT, size=13, color=INK_2), margin=dict(l=56, r=16, t=28, b=40),
                      hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)))
    fig.update_xaxes(showgrid=False, showline=False, ticks="", tickfont=dict(color=INK, size=13))
    fig.update_yaxes(gridcolor=GRID, griddash="dash", showline=False, zeroline=False, ticks="",
                     tickfont=dict(color=MUTED), rangemode="tozero", tickformat=",")
    if vals:
        set_value_axis(fig, max(vals) * 1.15, money)
    return fig


def _nice_step(top: float, n: int = 4) -> float:
    raw = top / n
    mag = 10 ** int(np.floor(np.log10(raw))) if raw > 0 else 1
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def set_value_axis(fig: go.Figure, top: float, money: bool = False) -> go.Figure:
    """Auto-scaled y-axis with about 4 round steps; money ticks read £0, £25K, £50K (reference style)."""
    step = _nice_step(top)
    ticks = list(np.arange(0, top + step * 0.001, step))
    if money:
        text = ["£0" if v == 0 else (f"£{v / 1e6:g}M" if v >= 1e6 else f"£{v / 1e3:g}K" if v >= 1e3 else f"£{v:g}")
                for v in ticks]
    else:
        text = [f"{v:,.0f}" for v in ticks]
    fig.update_yaxes(range=[0, top], tickvals=ticks, ticktext=text)
    return fig


def reference_line(fig: go.Figure, x0, y0, x1, y1, label: str | None = None,
                   label_at: str = "end") -> go.Figure:
    """Dashed muted reference line (diagonal, base rate, optimum).
    label_at="start" puts the label at (x0, y0) when the data crowds the line's end."""
    fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                  line=dict(color=MUTED, width=1.5, dash="dash"))
    if label:
        at_start = label_at == "start"
        fig.add_annotation(x=x0 if at_start else x1, y=y0 if at_start else y1, text=label,
                           showarrow=False, xanchor="left" if at_start else "right",
                           yanchor="bottom", font=dict(color=MUTED, size=11))
    return fig
