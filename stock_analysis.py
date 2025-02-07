from typing import Union, Dict, Set, List, TypedDict, Annotated
import pandas as pd
from langchain_core.tools import tool
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volume import volume_weighted_average_price
from langgraph.graph import StateGraph, START, END
import datetime as dt
import dotenv

@tool
def get_stock_prices(ticker: str) -> Union[Dict, str]:

    try:
        data = yf.download(
          ticker,
          start=dt.datetime.now() - dt.timedelta(weeks=24*3),
          end=dt.datetime.now(),
          interval="wk",
        )
        df = data.copy()
        data.reset_index(inplace=True)
        data.Date = data.Date.astype(str)
        
        indicators = {}
        
        rsi_series = RSIIndicator(df['Close'], window=14).rsi().iloc[-12:]
        indicators["RSI"] = {date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in rsi_series.dropna().to_dict().items()}
        
        sto_series = StochasticOscillator(
            df['High'], df['Low'], df['Close'], window=14).stoch().iloc[-12:]
        indicators["Stochastic_Oscillator"] = {
                    date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in sto_series.dropna().to_dict().items()}

        macd = MACD(df['Close'])
        macd_series = macd.macd().iloc[-12:]
        indicators["MACD"] = {date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in macd_series.to_dict().items()}
        
        macd_signal_series = macd.macd_signal().iloc[-12:]
        indicators["MACD_Signal"] = {date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in macd_signal_series.to_dict().items()}
        
        vwap_series = volume_weighted_average_price(
            high=df['High'], low=df['Low'], close=df['Close'], 
            volume=df['Volume'],
        ).iloc[-12:]
        indicators["vwap"] = {date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in vwap_series.to_dict().items()}
        
        return {'stock_price': data.to_dict(orient='recores'),
                'indicators': indicators}
        
    except Exception as e:
        return f"Error fetching price data: {str(e)}"

class State(TypedDict):
        messages: Annotated[list, add_messages]
        stock: str
    

    


        



    
graph = graph_builder.compile()
events = graph.stream({'messages':[('user', 'Should I buy this stock?')],
        'stock': 'TSLA'}, stream_mode='values')
for event in events:
        if 'messages' in event:
            event['messages'][-1].pretty_print()