import json
import logging
import yfinance as yf
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any
from datetime import datetime, timedelta
import pandas as pd

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define the schema for the ticker query input
class TickerQuery(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol to look up (e.g., AAPL, GOOGL, MSFT)")

# Define the YFinance tool class
class YFinanceTools(BaseTool):
    name: str = "Get Stock Financial Data"
    description: str = "Useful to get financial data about a stock ticker including current price, historical performance, and key metrics"
    args_schema: type[BaseModel] = TickerQuery

    # Main method to run the financial data fetch
    def _run(self, ticker: str) -> str:
        try:
            logger.info(f"Starting financial data fetch for ticker: {ticker}")
            
            # Create a yfinance ticker object
            stock = yf.Ticker(ticker.upper())
            
            # Get stock info
            info = stock.info
            logger.debug(f"Retrieved stock info for {ticker}")
            
            # Get historical data for the last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            hist_data = stock.history(start=start_date, end=end_date)
            
            if hist_data.empty:
                logger.warning(f"No historical data found for ticker: {ticker}")
                return f"No historical data found for ticker: {ticker}"
            
            # Calculate key metrics
            current_price = hist_data['Close'].iloc[-1] if not hist_data.empty else None
            
            # Calculate 7-day change
            if len(hist_data) >= 7:
                price_7d_ago = hist_data['Close'].iloc[-7]
                change_7d = ((current_price - price_7d_ago) / price_7d_ago) * 100
            else:
                change_7d = None
            
            # Calculate volatility (standard deviation of returns)
            returns = hist_data['Close'].pct_change().dropna()
            volatility = returns.std() * 100  # Convert to percentage
            
            # Determine volatility category
            if volatility < 2:
                volatility_category = "Low"
            elif volatility < 5:
                volatility_category = "Moderate"
            else:
                volatility_category = "High"
            
            # Get volume data
            avg_volume = hist_data['Volume'].mean()
            current_volume = hist_data['Volume'].iloc[-1]
            
            # Extract key company information
            company_name = info.get('longName', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            pe_ratio = info.get('trailingPE', 'N/A')
            dividend_yield = info.get('dividendYield', 'N/A')
            
            # Format dividend yield as percentage if available
            if dividend_yield != 'N/A' and dividend_yield is not None:
                dividend_yield = f"{dividend_yield * 100:.2f}%"
            
            # Format market cap
            if market_cap != 'N/A' and market_cap is not None:
                if market_cap >= 1e12:
                    market_cap = f"${market_cap / 1e12:.2f}T"
                elif market_cap >= 1e9:
                    market_cap = f"${market_cap / 1e9:.2f}B"
                elif market_cap >= 1e6:
                    market_cap = f"${market_cap / 1e6:.2f}M"
            
            # Create formatted result
            formatted_result = {
                "ticker": ticker.upper(),
                "company": company_name,
                "current_price": f"${current_price:.2f}" if current_price else "N/A",
                "change_7d": f"{change_7d:.2f}%" if change_7d is not None else "N/A",
                "volatility": f"{volatility:.2f}% ({volatility_category})" if volatility else "N/A",
                "market_cap": market_cap,
                "pe_ratio": f"{pe_ratio:.2f}" if pe_ratio and pe_ratio != 'N/A' else "N/A",
                "dividend_yield": dividend_yield,
                "avg_volume": f"{avg_volume:,.0f}" if avg_volume else "N/A",
                "current_volume": f"{current_volume:,.0f}" if current_volume else "N/A",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Format the output as a readable string
            result_string = f"""
Financial Data for {ticker.upper()}:
Company: {formatted_result['company']}
Current Price: {formatted_result['current_price']}
7-Day Change: {formatted_result['change_7d']}
Volatility: {formatted_result['volatility']}
Market Cap: {formatted_result['market_cap']}
P/E Ratio: {formatted_result['pe_ratio']}
Dividend Yield: {formatted_result['dividend_yield']}
Average Volume (30d): {formatted_result['avg_volume']}
Current Volume: {formatted_result['current_volume']}
Last Updated: {formatted_result['last_updated']}
"""
            
            logger.info(f"Successfully retrieved financial data for {ticker}")
            return result_string.strip()
            
        except Exception as e:
            logger.exception(f"Error fetching financial data for {ticker}: {str(e)}")
            return f"Error fetching financial data for {ticker}: {str(e)}"

    # Helper method to get multiple tickers at once
    def get_multiple_tickers(self, tickers: list) -> str:
        """
        Get financial data for multiple tickers at once
        """
        try:
            results = []
            for ticker in tickers:
                result = self._run(ticker)
                results.append(result)
            
            return "\n" + "="*50 + "\n".join(results)
            
        except Exception as e:
            logger.exception(f"Error fetching multiple ticker data: {str(e)}")
            return f"Error fetching multiple ticker data: {str(e)}"