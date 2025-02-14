import math

from langchain_core.messages import HumanMessage

from langgraph.graph import AgentState, show_agent_reasoning

import json
import pandas as pd
import numpy as np

from tools.api import get_prices, prices_to_df
from utils.progress import progress


### Technical Analysis Agent ###
def technical_analyst_agent(state: AgentState):
    """
    Sophisticated technical analysis system that combines multiple trading strategies for multiple tickers:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    for ticker in tickers:
        progress.update_status("technical_analyst_agent", ticker, "Analyzing price data")
        
        # Get the price data for the ticker
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        
        if not prices:
            progress.update_status("technical_analyst_agent", ticker, "No price data available")
            continue
        
        # convert prices to a DataFrame
        prices_df = prices_to_df(prices)
        
        progress.update_status("technical_analyst_agent", ticker, "Calculating trend signals")
        trend_signals = calculate_trend_signals(prices_df)
        
        progress.update_status("technical_analyst_agent", ticker, "Calculating mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating momentum")
        momentum_signals = calculate_momentum_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Analyzing volatility")
        volatility_signals = calculate_volatility_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Statistical analysis")
        stat_arb_signals = calculate_stat_arb_signals(prices_df)