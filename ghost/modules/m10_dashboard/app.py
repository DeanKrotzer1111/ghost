"""Ghost Trade Visualization Dashboard — Interactive Plotly Dash app."""
import json
from datetime import datetime, timezone
from dataclasses import asdict
from typing import List, Optional, Dict

import dash
from dash import dcc, html, callback_context, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ghost.modules.m00_backtest.engine import BacktestTrade, BacktestResult, BacktestConfig

# ---------------------------------------------------------------------------
# Color palette (dark theme)
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "border": "#30363d",
    "text": "#c9d1d9",
    "text_muted": "#8b949e",
    "green": "#3fb950",
    "red": "#f85149",
    "blue": "#58a6ff",
    "orange": "#d29922",
    "purple": "#bc8cff",
    "cyan": "#39d2c0",
    "candle_up": "#3fb950",
    "candle_down": "#f85149",
    "grid": "#21262d",
}

CHART_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="JetBrains Mono, Fira Code, monospace", color=COLORS["text"], size=11),
    margin=dict(l=60, r=20, t=40, b=30),
    xaxis=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False, side="right"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)


# ---------------------------------------------------------------------------
# Helper: timestamp -> readable string
# ---------------------------------------------------------------------------
def _ts_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Build the Dash app
# ---------------------------------------------------------------------------
def create_app(
    backtest_results: Dict[str, BacktestResult],
    bar_data: Dict[str, list],  # instrument -> list of dicts {timestamp, open, high, low, close, volume}
    equity_curves: Dict[str, List[float]],
    initial_balance: float = 50000.0,
) -> dash.Dash:
    """Create and return a configured Dash application.

    Args:
        backtest_results: dict mapping instrument name to BacktestResult.
        bar_data: dict mapping instrument name to list of OHLCV dicts.
        equity_curves: dict mapping instrument name to equity list.
        initial_balance: starting account balance.
    """

    instruments = sorted(backtest_results.keys())
    if not instruments:
        instruments = ["MNQ"]

    # Serialize trades for client-side storage
    trades_store = {}
    for inst, result in backtest_results.items():
        trades_store[inst] = [asdict(t) for t in result.trades]

    # -----------------------------------------------------------------------
    # App & Layout
    # -----------------------------------------------------------------------
    app = dash.Dash(
        __name__,
        title="Ghost Trading Dashboard",
        update_title=None,
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div(
        style={
            "backgroundColor": COLORS["bg"],
            "color": COLORS["text"],
            "fontFamily": "JetBrains Mono, Fira Code, Consolas, monospace",
            "minHeight": "100vh",
            "padding": "0",
        },
        children=[
            # Hidden stores
            dcc.Store(id="trades-store", data=trades_store),
            dcc.Store(id="bars-store", data=bar_data),
            dcc.Store(id="equity-store", data=equity_curves),
            dcc.Store(id="playback-index", data=0),
            dcc.Store(id="playback-state", data="stopped"),  # playing, paused, stopped
            dcc.Store(id="selected-trade-idx", data=None),
            dcc.Interval(id="playback-interval", interval=500, n_intervals=0, disabled=True),

            # --- Top bar ---
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "padding": "12px 24px",
                    "borderBottom": f"1px solid {COLORS['border']}",
                    "backgroundColor": COLORS["panel"],
                },
                children=[
                    html.Div([
                        html.Span("GHOST", style={"fontSize": "20px", "fontWeight": "700", "color": COLORS["cyan"]}),
                        html.Span(" TRADE VISUALIZER", style={"fontSize": "20px", "fontWeight": "300", "color": COLORS["text_muted"]}),
                    ]),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "16px"},
                        children=[
                            html.Label("Instrument", style={"color": COLORS["text_muted"], "fontSize": "12px"}),
                            dcc.Dropdown(
                                id="instrument-selector",
                                options=[{"label": i, "value": i} for i in instruments],
                                value=instruments[0],
                                clearable=False,
                                style={
                                    "width": "140px",
                                    "backgroundColor": COLORS["panel"],
                                    "color": COLORS["text"],
                                    "border": f"1px solid {COLORS['border']}",
                                    "fontSize": "13px",
                                },
                            ),
                            # Playback controls
                            html.Button(
                                "\u25b6", id="btn-play",
                                style=_btn_style(COLORS["green"]),
                                title="Play",
                            ),
                            html.Button(
                                "\u23f8", id="btn-pause",
                                style=_btn_style(COLORS["orange"]),
                                title="Pause",
                            ),
                            html.Button(
                                "\u23f9", id="btn-stop",
                                style=_btn_style(COLORS["red"]),
                                title="Stop",
                            ),
                            html.Label("Speed", style={"color": COLORS["text_muted"], "fontSize": "12px"}),
                            dcc.Dropdown(
                                id="speed-selector",
                                options=[
                                    {"label": "1x", "value": 500},
                                    {"label": "2x", "value": 250},
                                    {"label": "5x", "value": 100},
                                    {"label": "10x", "value": 50},
                                ],
                                value=500,
                                clearable=False,
                                style={
                                    "width": "80px",
                                    "backgroundColor": COLORS["panel"],
                                    "color": COLORS["text"],
                                    "border": f"1px solid {COLORS['border']}",
                                    "fontSize": "13px",
                                },
                            ),
                            html.Div(id="playback-status", style={"color": COLORS["text_muted"], "fontSize": "12px", "minWidth": "120px"}),
                        ],
                    ),
                ],
            ),

            # --- Main content ---
            html.Div(
                style={"display": "flex", "height": "calc(100vh - 60px)"},
                children=[
                    # Left: Charts
                    html.Div(
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            # Candlestick chart
                            dcc.Graph(
                                id="main-chart",
                                config={"displayModeBar": True, "scrollZoom": True},
                                style={"flex": "3", "minHeight": "0"},
                            ),
                            # Equity curve
                            dcc.Graph(
                                id="equity-chart",
                                config={"displayModeBar": False},
                                style={"flex": "1", "minHeight": "0", "borderTop": f"1px solid {COLORS['border']}"},
                            ),
                        ],
                    ),
                    # Right: Side panel
                    html.Div(
                        id="side-panel",
                        style={
                            "width": "360px",
                            "borderLeft": f"1px solid {COLORS['border']}",
                            "backgroundColor": COLORS["panel"],
                            "overflowY": "auto",
                            "padding": "16px",
                        },
                        children=[
                            html.Div(id="stats-panel"),
                            html.Hr(style={"borderColor": COLORS["border"]}),
                            html.Div(id="trade-list-panel"),
                        ],
                    ),
                ],
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    @app.callback(
        [
            Output("playback-state", "data"),
            Output("playback-interval", "disabled"),
            Output("playback-index", "data", allow_duplicate=True),
        ],
        [
            Input("btn-play", "n_clicks"),
            Input("btn-pause", "n_clicks"),
            Input("btn-stop", "n_clicks"),
        ],
        [State("playback-state", "data"), State("playback-index", "data")],
        prevent_initial_call=True,
    )
    def handle_playback_buttons(play_clicks, pause_clicks, stop_clicks, state, index):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if btn_id == "btn-play":
            return "playing", False, index
        elif btn_id == "btn-pause":
            return "paused", True, index
        elif btn_id == "btn-stop":
            return "stopped", True, 0
        return no_update, no_update, no_update

    @app.callback(
        Output("playback-interval", "interval"),
        Input("speed-selector", "value"),
    )
    def update_speed(speed):
        return speed or 500

    @app.callback(
        Output("playback-index", "data"),
        Input("playback-interval", "n_intervals"),
        [
            State("playback-index", "data"),
            State("playback-state", "data"),
            State("instrument-selector", "value"),
            State("bars-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def advance_playback(n, index, state, instrument, bars_data):
        if state != "playing" or not instrument or instrument not in bars_data:
            return no_update
        max_idx = len(bars_data[instrument]) - 1
        new_idx = min(index + 1, max_idx)
        if new_idx >= max_idx:
            return max_idx
        return new_idx

    @app.callback(
        Output("playback-status", "children"),
        [Input("playback-index", "data"), Input("playback-state", "data")],
        State("instrument-selector", "value"),
    )
    def update_playback_status(index, state, instrument):
        if not instrument or instrument not in bar_data:
            return ""
        total = len(bar_data.get(instrument, []))
        icon = {"playing": "\u25b6", "paused": "\u23f8", "stopped": "\u23f9"}.get(state, "")
        return f"{icon} Bar {index}/{total}"

    @app.callback(
        [Output("main-chart", "figure"), Output("equity-chart", "figure"), Output("stats-panel", "children"), Output("trade-list-panel", "children")],
        [Input("instrument-selector", "value"), Input("playback-index", "data"), Input("playback-state", "data")],
        [State("trades-store", "data"), State("bars-store", "data"), State("equity-store", "data")],
    )
    def update_charts(instrument, playback_idx, playback_state, trades_data, bars_data, eq_data):
        if not instrument:
            empty_fig = go.Figure()
            empty_fig.update_layout(**CHART_LAYOUT)
            return empty_fig, empty_fig, "", ""

        # Get bars
        bars = bars_data.get(instrument, [])
        trades_list = trades_data.get(instrument, [])
        eq_curve = eq_data.get(instrument, [])

        # Determine visible range
        if playback_state == "stopped" or playback_idx == 0:
            visible_bars = bars
            visible_idx = len(bars)
        else:
            visible_idx = min(playback_idx, len(bars))
            visible_bars = bars[:visible_idx]

        if not visible_bars:
            empty_fig = go.Figure()
            empty_fig.update_layout(**CHART_LAYOUT)
            return empty_fig, empty_fig, "", ""

        # Filter trades visible at current playback position
        if playback_state == "stopped" and playback_idx == 0:
            visible_trades = trades_list
        else:
            first_ts = visible_bars[0]["timestamp"] if visible_bars else 0
            last_ts = visible_bars[-1]["timestamp"] if visible_bars else 0
            visible_trades = [t for t in trades_list if t["entry_time"] <= last_ts]

        # ---- Build main chart ----
        timestamps = [datetime.fromtimestamp(b["timestamp"], tz=timezone.utc) for b in visible_bars]
        opens = [b["open"] for b in visible_bars]
        highs = [b["high"] for b in visible_bars]
        lows = [b["low"] for b in visible_bars]
        closes = [b["close"] for b in visible_bars]
        volumes = [b["volume"] for b in visible_bars]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.82, 0.18],
            vertical_spacing=0.02,
        )

        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=timestamps,
                open=opens, high=highs, low=lows, close=closes,
                increasing=dict(line=dict(color=COLORS["candle_up"]), fillcolor=COLORS["candle_up"]),
                decreasing=dict(line=dict(color=COLORS["candle_down"]), fillcolor=COLORS["candle_down"]),
                name="Price",
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Volume bars
        vol_colors = [COLORS["candle_up"] if c >= o else COLORS["candle_down"] for o, c in zip(opens, closes)]
        fig.add_trace(
            go.Bar(
                x=timestamps, y=volumes,
                marker_color=vol_colors,
                opacity=0.35,
                name="Volume",
                showlegend=False,
            ),
            row=2, col=1,
        )

        # Trade markers and level lines
        for t in visible_trades:
            entry_dt = datetime.fromtimestamp(t["entry_time"], tz=timezone.utc)
            exit_dt = datetime.fromtimestamp(t["exit_time"], tz=timezone.utc)
            is_bull = t["direction"] == "BULLISH"

            # Entry marker
            fig.add_trace(
                go.Scatter(
                    x=[entry_dt], y=[t["entry_price"]],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up" if is_bull else "triangle-down",
                        size=14,
                        color=COLORS["green"] if is_bull else COLORS["red"],
                        line=dict(width=1, color="white"),
                    ),
                    name=f"Entry #{t['trade_id']}",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Entry #{t['trade_id']}</b><br>"
                        f"{'LONG' if is_bull else 'SHORT'} @ {t['entry_price']:.2f}<br>"
                        f"TQS: {t['tqs_score']:.0f} | Conf: {t['confluence_score']:.2f}"
                        "<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

            # Exit marker
            exit_color = COLORS["green"] if t["pnl"] > 0 else COLORS["red"]
            # Only show exit if it's within visible range
            if playback_state == "stopped" or t["exit_time"] <= (visible_bars[-1]["timestamp"] if visible_bars else 0):
                fig.add_trace(
                    go.Scatter(
                        x=[exit_dt], y=[t["exit_price"]],
                        mode="markers",
                        marker=dict(symbol="x", size=12, color=exit_color, line=dict(width=2, color=exit_color)),
                        name=f"Exit #{t['trade_id']}",
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Exit #{t['trade_id']}</b><br>"
                            f"{t['outcome']} @ {t['exit_price']:.2f}<br>"
                            f"P&L: ${t['pnl']:.2f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=1, col=1,
                )

            # SL / TP lines
            fig.add_shape(
                type="line", x0=entry_dt, x1=exit_dt,
                y0=t["stop"], y1=t["stop"],
                line=dict(color=COLORS["red"], width=1, dash="dash"),
                row=1, col=1,
            )
            for tp_key, tp_label in [("tp1", "TP1"), ("tp2", "TP2"), ("tp3", "TP3")]:
                fig.add_shape(
                    type="line", x0=entry_dt, x1=exit_dt,
                    y0=t[tp_key], y1=t[tp_key],
                    line=dict(color=COLORS["green"], width=1, dash="dash"),
                    row=1, col=1,
                )

        fig.update_layout(
            **CHART_LAYOUT,
            title=dict(text=f"{instrument} — Price Chart", font=dict(size=14)),
            xaxis_rangeslider_visible=False,
            xaxis2=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False),
            yaxis2=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False, side="right"),
            height=None,
        )

        # ---- Equity chart ----
        eq_fig = go.Figure()
        if eq_curve:
            visible_eq = eq_curve[:visible_idx] if (playback_state != "stopped" or playback_idx != 0) else eq_curve
            eq_fig.add_trace(go.Scatter(
                y=visible_eq,
                mode="lines",
                line=dict(color=COLORS["cyan"], width=2),
                fill="tozeroy",
                fillcolor="rgba(57,210,192,0.08)",
                name="Equity",
            ))
        eq_fig.update_layout(
            **CHART_LAYOUT,
            title=dict(text="Equity Curve", font=dict(size=12)),
            height=None,
            yaxis=dict(gridcolor=COLORS["grid"], showgrid=True, zeroline=False, side="right",
                       tickformat="$,.0f"),
        )

        # ---- Stats panel ----
        result = backtest_results.get(instrument)
        stats_children = _build_stats_panel(result, initial_balance) if result else []

        # ---- Trade list ----
        trade_list_children = _build_trade_list(visible_trades)

        return fig, eq_fig, stats_children, trade_list_children

    return app


# ---------------------------------------------------------------------------
# UI Builders
# ---------------------------------------------------------------------------

def _btn_style(color: str) -> dict:
    return {
        "backgroundColor": "transparent",
        "border": f"1px solid {color}",
        "color": color,
        "borderRadius": "6px",
        "padding": "4px 12px",
        "cursor": "pointer",
        "fontSize": "16px",
        "fontWeight": "700",
    }


def _stat_row(label: str, value: str, color: str = COLORS["text"]) -> html.Div:
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between", "padding": "4px 0"},
        children=[
            html.Span(label, style={"color": COLORS["text_muted"], "fontSize": "12px"}),
            html.Span(value, style={"color": color, "fontSize": "13px", "fontWeight": "600"}),
        ],
    )


def _build_stats_panel(result: BacktestResult, initial_balance: float) -> list:
    if not result:
        return [html.Div("No results", style={"color": COLORS["text_muted"]})]

    pnl_color = COLORS["green"] if result.total_pnl >= 0 else COLORS["red"]

    return [
        html.Div("PERFORMANCE", style={
            "fontSize": "11px", "fontWeight": "700", "color": COLORS["cyan"],
            "letterSpacing": "2px", "marginBottom": "12px",
        }),
        _stat_row("Total P&L", f"${result.total_pnl:,.2f}", pnl_color),
        _stat_row("Win Rate", f"{result.win_rate * 100:.1f}%",
                  COLORS["green"] if result.win_rate >= 0.5 else COLORS["red"]),
        _stat_row("Total Trades", str(result.total_trades)),
        _stat_row("Wins / Losses", f"{result.winning_trades} / {result.losing_trades}"),
        _stat_row("Avg Win", f"${result.avg_win:,.2f}", COLORS["green"]),
        _stat_row("Avg Loss", f"${result.avg_loss:,.2f}", COLORS["red"]),
        _stat_row("Expectancy", f"${result.expectancy:,.2f}",
                  COLORS["green"] if result.expectancy > 0 else COLORS["red"]),
        _stat_row("Profit Factor", f"{result.profit_factor:.2f}",
                  COLORS["green"] if result.profit_factor > 1 else COLORS["red"]),
        _stat_row("Max Drawdown", f"${result.max_drawdown:,.2f} ({result.max_drawdown_pct * 100:.1f}%)", COLORS["red"]),
        _stat_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}",
                  COLORS["green"] if result.sharpe_ratio > 1 else COLORS["text"]),
        _stat_row("Final Balance", f"${result.final_balance:,.2f}"),
        html.Div(style={"height": "8px"}),
        _stat_row("Signals Generated", str(result.signals_generated)),
        _stat_row("Signals Rejected", str(result.signals_rejected)),
        _stat_row("Shadow Signals", str(result.shadow_signals)),
    ]


def _build_trade_list(trades: list) -> list:
    if not trades:
        return [html.Div("No trades", style={"color": COLORS["text_muted"], "fontSize": "12px"})]

    children = [
        html.Div("TRADE LOG", style={
            "fontSize": "11px", "fontWeight": "700", "color": COLORS["cyan"],
            "letterSpacing": "2px", "marginBottom": "8px",
        }),
    ]

    for t in reversed(trades):  # Most recent first
        is_win = t["pnl"] > 0
        border_color = COLORS["green"] if is_win else COLORS["red"]
        direction_label = "LONG" if t["direction"] == "BULLISH" else "SHORT"
        grade_color = {
            "A+": COLORS["green"], "A": COLORS["green"],
            "B+": COLORS["cyan"], "B": COLORS["cyan"],
            "C": COLORS["orange"], "D": COLORS["red"], "F": COLORS["red"],
        }.get(t.get("tqs_grade", ""), COLORS["text_muted"])

        children.append(
            html.Div(
                style={
                    "borderLeft": f"3px solid {border_color}",
                    "backgroundColor": COLORS["bg"],
                    "borderRadius": "4px",
                    "padding": "8px 10px",
                    "marginBottom": "6px",
                    "fontSize": "11px",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"},
                        children=[
                            html.Span(f"#{t['trade_id']} {direction_label}", style={"fontWeight": "700"}),
                            html.Span(
                                f"${t['pnl']:+,.2f}",
                                style={"fontWeight": "700", "color": COLORS["green"] if is_win else COLORS["red"]},
                            ),
                        ],
                    ),
                    html.Div(f"Entry: {t['entry_price']:.2f}  |  Exit: {t['exit_price']:.2f}", style={"color": COLORS["text_muted"]}),
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "marginTop": "2px"},
                        children=[
                            html.Span(t["outcome"], style={"color": COLORS["green"] if is_win else COLORS["red"]}),
                            html.Span(f"TQS: {t.get('tqs_score', 0):.0f}", style={"color": grade_color}),
                            html.Span(f"Conf: {t.get('confluence_score', 0):.2f}", style={"color": COLORS["text_muted"]}),
                        ],
                    ),
                    html.Div(
                        style={"color": COLORS["text_muted"], "marginTop": "2px"},
                        children=[
                            html.Span(f"SL: {t['stop']:.2f}"),
                            html.Span(" | ", style={"color": COLORS["border"]}),
                            html.Span(f"TP1: {t['tp1']:.2f}  TP2: {t['tp2']:.2f}  TP3: {t['tp3']:.2f}"),
                        ],
                    ),
                    html.Div(
                        f"{_ts_str(t['entry_time'])} \u2192 {_ts_str(t['exit_time'])}  ({t['bars_held']} bars)",
                        style={"color": COLORS["text_muted"], "marginTop": "2px"},
                    ),
                ],
            )
        )

    return children
