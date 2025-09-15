import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import asyncio
import threading
from typing import Dict, List, Any
import logging

# Import our analysis components
import sys
sys.path.append('..')
from main import StockAnalysisApp
from utils.stock_universe import get_universe_info

logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__, title="Stock Analysis Dashboard")

# Global variables for data storage
current_analysis = {}
analysis_history = []
last_update = None

# Initialize the stock analysis app
stock_app = StockAnalysisApp()

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🚀 Sophisticated Stock Analysis Dashboard",
                className="header-title"),
        html.P("Powered by Google Agent Development Kit",
               className="header-subtitle"),
        html.Div(id="last-update", className="last-update")
    ], className="header"),

    # Control panel
    html.Div([
        html.Div([
            html.Label("Stock Universe:"),
            dcc.Dropdown(
                id="universe-dropdown",
                options=[
                    {"label": f"{name} ({info['count']} stocks)", "value": name}
                    for name, info in get_universe_info().items()
                ],
                value="SP500_SAMPLE",
                className="dropdown"
            ),
        ], className="control-item"),

        html.Div([
            html.Label("Max Picks:"),
            dcc.Slider(
                id="max-picks-slider",
                min=5,
                max=20,
                step=1,
                value=10,
                marks={i: str(i) for i in range(5, 21, 5)},
                className="slider"
            ),
        ], className="control-item"),

        html.Div([
            html.Button("🔄 Run Analysis", id="run-analysis-btn",
                       className="run-button"),
            html.Button("📊 Auto Refresh", id="auto-refresh-btn",
                       className="auto-button"),
        ], className="button-group"),

    ], className="control-panel"),

    # Analysis status
    html.Div(id="analysis-status", className="status-panel"),

    # Main content tabs
    dcc.Tabs(id="main-tabs", value="overview", children=[

        # Overview Tab
        dcc.Tab(label="📈 Overview", value="overview", children=[
            html.Div([
                # KPI Cards
                html.Div(id="kpi-cards", className="kpi-container"),

                # Market overview
                html.Div([
                    html.H3("Market Overview"),
                    html.Div(id="market-overview-text"),
                    dcc.Graph(id="market-indices-chart")
                ], className="market-overview"),

                # Top picks summary
                html.Div([
                    html.H3("Top Stock Picks"),
                    html.Div(id="top-picks-summary")
                ], className="top-picks")
            ])
        ]),

        # Detailed Analysis Tab
        dcc.Tab(label="🔍 Detailed Analysis", value="analysis", children=[
            html.Div([
                html.Div([
                    html.H3("Stock Analysis Details"),
                    dash_table.DataTable(
                        id="analysis-table",
                        columns=[
                            {"name": "Symbol", "id": "symbol"},
                            {"name": "Price", "id": "current_price", "type": "numeric", "format": {"specifier": ".2f"}},
                            {"name": "Overall Score", "id": "overall_score", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Technical", "id": "technical_score", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Fundamental", "id": "fundamental_score", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Sentiment", "id": "sentiment_score", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Signal", "id": "technical_signal"},
                            {"name": "Risk", "id": "risk_level"},
                        ],
                        sort_action="native",
                        sort_by=[{"column_id": "overall_score", "direction": "desc"}],
                        style_cell={"textAlign": "left"},
                        style_data_conditional=[
                            {
                                "if": {"filter_query": "{overall_score} >= 80"},
                                "backgroundColor": "#d4edda",
                            },
                            {
                                "if": {"filter_query": "{overall_score} >= 70"},
                                "backgroundColor": "#fff3cd",
                            },
                            {
                                "if": {"filter_query": "{overall_score} < 60"},
                                "backgroundColor": "#f8d7da",
                            }
                        ]
                    )
                ], className="analysis-table-container"),

                html.Div([
                    html.H3("Stock Details"),
                    html.Div(id="stock-detail-panel")
                ], className="stock-details")
            ])
        ]),

        # Trading Strategies Tab
        dcc.Tab(label="⚡ Trading Strategies", value="strategies", children=[
            html.Div([
                html.H3("Entry & Exit Strategies"),
                html.Div(id="strategies-content")
            ])
        ]),

        # Performance Tab
        dcc.Tab(label="📊 Performance", value="performance", children=[
            html.Div([
                html.H3("Analysis Performance"),
                dcc.Graph(id="performance-chart"),
                html.Div(id="performance-metrics")
            ])
        ])
    ]),

    # Footer
    html.Div([
        html.P("Real-time stock analysis using AI-powered agents"),
        html.P("⚠️ For educational purposes only. Not financial advice.")
    ], className="footer"),

    # Interval component for auto-refresh
    dcc.Interval(
        id="interval-component",
        interval=60*1000,  # Update every minute
        n_intervals=0,
        disabled=True
    ),

    # Store components for data
    dcc.Store(id="analysis-data-store"),
    dcc.Store(id="auto-refresh-store", data=False)
])

# Callbacks

@app.callback(
    [Output("analysis-status", "children"),
     Output("analysis-data-store", "data")],
    [Input("run-analysis-btn", "n_clicks"),
     Input("interval-component", "n_intervals")],
    [State("universe-dropdown", "value"),
     State("max-picks-slider", "value"),
     State("auto-refresh-store", "data")]
)
def run_analysis(n_clicks, n_intervals, universe, max_picks, auto_refresh):
    """Run stock analysis and update data store"""
    global current_analysis, last_update

    # Check if we should run analysis
    should_run = False
    if n_clicks and n_clicks > 0:
        should_run = True
    elif auto_refresh and n_intervals > 0:
        should_run = True

    if not should_run:
        if current_analysis:
            return f"Last analysis: {last_update}", current_analysis
        return "Click 'Run Analysis' to start", {}

    try:
        # Show loading status
        status = html.Div([
            html.Div(className="loading-spinner"),
            html.P("Running analysis... This may take several minutes.")
        ], className="loading-status")

        # Run analysis in background thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            stock_app.run_daily_analysis(universe, max_picks)
        )

        current_analysis = result
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        success_status = html.Div([
            html.P(f"✅ Analysis completed successfully at {last_update}"),
            html.P(f"Analyzed {result.get('total_analyzed', 0)} stocks, "
                   f"generated {result.get('top_picks_count', 0)} picks")
        ], className="success-status")

        return success_status, result

    except Exception as e:
        error_status = html.Div([
            html.P(f"❌ Analysis failed: {str(e)}"),
            html.P("Please check logs for details")
        ], className="error-status")

        return error_status, {}

@app.callback(
    Output("auto-refresh-store", "data"),
    Output("interval-component", "disabled"),
    Input("auto-refresh-btn", "n_clicks"),
    State("auto-refresh-store", "data")
)
def toggle_auto_refresh(n_clicks, current_state):
    """Toggle auto-refresh functionality"""
    if n_clicks:
        new_state = not current_state
        return new_state, not new_state
    return current_state, not current_state

@app.callback(
    Output("kpi-cards", "children"),
    Input("analysis-data-store", "data")
)
def update_kpi_cards(data):
    """Update KPI cards with analysis results"""
    if not data:
        return html.Div("No data available")

    analysis_summary = data.get("analysis_summary", {})
    daily_picks = data.get("daily_picks", {})

    cards = [
        html.Div([
            html.H4(str(data.get("top_picks_count", 0))),
            html.P("Top Picks")
        ], className="kpi-card"),

        html.Div([
            html.H4(str(analysis_summary.get("total_analyzed", 0))),
            html.P("Stocks Analyzed")
        ], className="kpi-card"),

        html.Div([
            html.H4(f"{analysis_summary.get('average_score', 0):.1f}"),
            html.P("Average Score")
        ], className="kpi-card"),

        html.Div([
            html.H4(data.get("market_sentiment", "Unknown")),
            html.P("Market Sentiment")
        ], className="kpi-card")
    ]

    return cards

@app.callback(
    Output("analysis-table", "data"),
    Input("analysis-data-store", "data")
)
def update_analysis_table(data):
    """Update the detailed analysis table"""
    if not data or "daily_picks" not in data:
        return []

    picks = data["daily_picks"].get("picks", [])
    table_data = []

    for pick in picks:
        table_data.append({
            "symbol": pick["symbol"],
            "current_price": pick["current_price"],
            "overall_score": pick["overall_score"],
            "technical_score": pick["technical_score"],
            "fundamental_score": pick["fundamental_score"],
            "sentiment_score": pick["sentiment_score"],
            "technical_signal": pick["technical_signal"],
            "risk_level": pick["risk_level"]
        })

    return table_data

@app.callback(
    Output("market-overview-text", "children"),
    Input("analysis-data-store", "data")
)
def update_market_overview(data):
    """Update market overview text"""
    if not data:
        return "No market data available"

    daily_picks = data.get("daily_picks", {})
    market_overview = daily_picks.get("market_overview", "No market overview available")

    return html.P(market_overview)

@app.callback(
    Output("top-picks-summary", "children"),
    Input("analysis-data-store", "data")
)
def update_top_picks_summary(data):
    """Update top picks summary"""
    if not data:
        return "No picks available"

    picks = data.get("daily_picks", {}).get("picks", [])[:5]  # Top 5

    if not picks:
        return "No picks generated"

    summary_items = []
    for i, pick in enumerate(picks, 1):
        summary_items.append(
            html.Div([
                html.Div([
                    html.H4(f"{i}. {pick['symbol']}"),
                    html.P(f"${pick['current_price']:.2f}"),
                    html.P(f"Score: {pick['overall_score']:.1f}/100", className="score")
                ], className="pick-header"),
                html.Div([
                    html.P(f"Signal: {pick['technical_signal']}"),
                    html.P(f"Risk: {pick['risk_level']}"),
                    html.P(f"Entry: ${pick['entry_strategy']['entry_price']:.2f}")
                ], className="pick-details")
            ], className="pick-card")
        )

    return summary_items

@app.callback(
    Output("strategies-content", "children"),
    Input("analysis-data-store", "data")
)
def update_strategies_content(data):
    """Update trading strategies content"""
    if not data:
        return "No strategy data available"

    picks = data.get("daily_picks", {}).get("picks", [])

    if not picks:
        return "No strategies generated"

    strategies = []
    for pick in picks:
        entry_strategy = pick.get("entry_strategy", {})
        exit_strategy = pick.get("exit_strategy", {})

        strategies.append(
            html.Div([
                html.H4(f"{pick['symbol']} Trading Strategy"),
                html.Div([
                    html.Div([
                        html.H5("Entry Strategy"),
                        html.P(f"Entry Price: ${entry_strategy.get('entry_price', 0):.2f}"),
                        html.P(f"Stop Loss: ${entry_strategy.get('stop_loss', 0):.2f}"),
                        html.P(f"Take Profit: ${entry_strategy.get('take_profit', 0):.2f}"),
                        html.P(f"Risk/Reward: {entry_strategy.get('risk_reward_ratio', 0):.2f}"),
                        html.P(f"Confidence: {entry_strategy.get('confidence_score', 0):.1f}%")
                    ], className="strategy-column"),

                    html.Div([
                        html.H5("Exit Strategy"),
                        html.P(f"Exit Price: ${exit_strategy.get('exit_price', 0):.2f}"),
                        html.P(f"Timeframe: {exit_strategy.get('timeframe', 'N/A')}"),
                        html.P(entry_strategy.get('reasoning', 'No reasoning provided'))
                    ], className="strategy-column")
                ], className="strategy-row")
            ], className="strategy-card")
        )

    return strategies

# CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }

            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                text-align: center;
            }

            .header-title {
                margin: 0;
                font-size: 2.5rem;
                font-weight: 600;
            }

            .header-subtitle {
                margin: 0.5rem 0 0 0;
                font-size: 1.2rem;
                opacity: 0.9;
            }

            .control-panel {
                background: white;
                padding: 1.5rem;
                margin: 1rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: flex;
                gap: 2rem;
                align-items: center;
                flex-wrap: wrap;
            }

            .control-item {
                min-width: 200px;
            }

            .button-group {
                display: flex;
                gap: 1rem;
            }

            .run-button, .auto-button {
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 6px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }

            .run-button {
                background: #28a745;
                color: white;
            }

            .run-button:hover {
                background: #218838;
            }

            .auto-button {
                background: #007bff;
                color: white;
            }

            .auto-button:hover {
                background: #0056b3;
            }

            .kpi-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 1rem;
            }

            .kpi-card {
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }

            .kpi-card h4 {
                margin: 0;
                font-size: 2rem;
                color: #333;
            }

            .kpi-card p {
                margin: 0.5rem 0 0 0;
                color: #666;
                font-weight: 500;
            }

            .pick-card {
                background: white;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-between;
            }

            .strategy-card {
                background: white;
                border-radius: 8px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .strategy-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
            }

            .loading-status {
                text-align: center;
                padding: 2rem;
            }

            .loading-spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #007bff;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 1rem auto;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .success-status {
                background: #d4edda;
                color: #155724;
                padding: 1rem;
                border-radius: 6px;
                margin: 1rem;
            }

            .error-status {
                background: #f8d7da;
                color: #721c24;
                padding: 1rem;
                border-radius: 6px;
                margin: 1rem;
            }

            .footer {
                background: #333;
                color: white;
                text-align: center;
                padding: 2rem;
                margin-top: 2rem;
            }

            .footer p {
                margin: 0.5rem 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)