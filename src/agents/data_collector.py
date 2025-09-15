import yfinance as yf
import requests
import feedparser
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from .base_agent import DataCollectionAgent
from ..models.stock_data import StockPrice, NewsItem, FundamentalData
import logging

logger = logging.getLogger(__name__)

class StockDataCollector(DataCollectionAgent):
    """Collects real-time and historical stock price data"""

    def __init__(self):
        super().__init__(
            name="StockDataCollector",
            description="Collects comprehensive stock price and volume data",
            model="gemini-2.0-flash"
        )

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        symbols = data.get("symbols", [])
        period = data.get("period", "1y")
        return await self.collect_data(symbols, period=period)

    async def collect_data(self, symbols: List[str], period: str = "1y", **kwargs) -> Dict[str, Any]:
        """Collect stock data for given symbols"""
        self.log_analysis_start()

        stock_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)

                # Get historical data
                hist_data = ticker.history(period=period)

                # Get current info
                info = ticker.info

                # Convert to our StockPrice model
                prices = []
                for index, row in hist_data.iterrows():
                    price = StockPrice(
                        symbol=symbol,
                        timestamp=index,
                        open=row['Open'],
                        high=row['High'],
                        low=row['Low'],
                        close=row['Close'],
                        volume=int(row['Volume']),
                        adjusted_close=row.get('Adj Close')
                    )
                    prices.append(price)

                # Get fundamental data
                fundamental = FundamentalData(
                    symbol=symbol,
                    market_cap=info.get('marketCap'),
                    pe_ratio=info.get('forwardPE'),
                    peg_ratio=info.get('pegRatio'),
                    price_to_book=info.get('priceToBook'),
                    debt_to_equity=info.get('debtToEquity'),
                    return_on_equity=info.get('returnOnEquity'),
                    return_on_assets=info.get('returnOnAssets'),
                    profit_margin=info.get('profitMargins'),
                    revenue_growth=info.get('revenueGrowth'),
                    earnings_growth=info.get('earningsGrowth'),
                    dividend_yield=info.get('dividendYield'),
                    beta=info.get('beta')
                )

                stock_data[symbol] = {
                    "prices": [price.dict() for price in prices],
                    "fundamental": fundamental.dict(),
                    "current_price": hist_data['Close'].iloc[-1] if not hist_data.empty else None,
                    "volume_avg": hist_data['Volume'].rolling(window=20).mean().iloc[-1] if not hist_data.empty else None
                }

            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {str(e)}")
                stock_data[symbol] = {"error": str(e)}

        self.log_analysis_complete(result_summary=f"Collected data for {len(stock_data)} symbols")
        return stock_data

class NewsCollector(DataCollectionAgent):
    """Collects and processes financial news"""

    def __init__(self, news_api_key: Optional[str] = None):
        super().__init__(
            name="NewsCollector",
            description="Collects financial news and market sentiment",
            model="gemini-2.0-flash"
        )
        self.news_api_key = news_api_key

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        symbols = data.get("symbols", [])
        days_back = data.get("days_back", 7)
        return await self.collect_data(symbols, days_back=days_back)

    async def collect_data(self, symbols: List[str], days_back: int = 7, **kwargs) -> Dict[str, Any]:
        """Collect news data for given symbols"""
        self.log_analysis_start()

        news_data = {}

        # RSS feeds for financial news
        rss_feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.marketwatch.com/rss/topstories",
            "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
        ]

        all_news = []

        # Collect from RSS feeds
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    if self._is_recent(entry.get('published_parsed'), days_back):
                        news_item = NewsItem(
                            title=entry.title,
                            content=entry.get('summary', ''),
                            source=feed.feed.get('title', 'Unknown'),
                            url=entry.link,
                            published_at=datetime(*entry.published_parsed[:6]),
                            symbols=self._extract_symbols(entry.title + ' ' + entry.get('summary', ''), symbols)
                        )
                        all_news.append(news_item)
            except Exception as e:
                logger.error(f"Error collecting from RSS feed {feed_url}: {str(e)}")

        # Group news by symbol
        for symbol in symbols:
            symbol_news = [news for news in all_news if symbol in news.symbols]
            news_data[symbol] = [news.dict() for news in symbol_news]

        news_data["general_market"] = [news.dict() for news in all_news if not news.symbols]

        self.log_analysis_complete(result_summary=f"Collected {len(all_news)} news items")
        return news_data

    def _is_recent(self, published_parsed, days_back: int) -> bool:
        """Check if news item is within the specified time range"""
        if not published_parsed:
            return False

        published_date = datetime(*published_parsed[:6])
        cutoff_date = datetime.now() - timedelta(days=days_back)
        return published_date >= cutoff_date

    def _extract_symbols(self, text: str, target_symbols: List[str]) -> List[str]:
        """Extract relevant stock symbols from news text"""
        text_upper = text.upper()
        found_symbols = []

        for symbol in target_symbols:
            if symbol.upper() in text_upper:
                found_symbols.append(symbol)

        return found_symbols

class MarketDataCollector(DataCollectionAgent):
    """Collects broader market and sector data"""

    def __init__(self):
        super().__init__(
            name="MarketDataCollector",
            description="Collects market indices, sector performance, and economic indicators",
            model="gemini-2.0-flash"
        )

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return await self.collect_data([])

    async def collect_data(self, symbols: List[str], **kwargs) -> Dict[str, Any]:
        """Collect market-wide data"""
        self.log_analysis_start()

        market_data = {}

        # Major indices
        indices = {
            "SPY": "S&P 500",
            "QQQ": "NASDAQ",
            "DIA": "Dow Jones",
            "IWM": "Russell 2000",
            "VIX": "Volatility Index"
        }

        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")

                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change_pct = ((current_price - prev_close) / prev_close) * 100

                    market_data[symbol] = {
                        "name": name,
                        "current_price": current_price,
                        "change_percent": change_pct,
                        "volume": hist['Volume'].iloc[-1],
                        "volatility": hist['Close'].pct_change().std() * 100
                    }
            except Exception as e:
                logger.error(f"Error collecting market data for {symbol}: {str(e)}")
                market_data[symbol] = {"error": str(e)}

        # Economic indicators (simplified - would integrate with FRED API in production)
        market_data["economic_indicators"] = {
            "market_fear_greed": self._calculate_fear_greed_index(market_data),
            "market_trend": self._determine_market_trend(market_data)
        }

        self.log_analysis_complete(result_summary="Collected market overview data")
        return market_data

    def _calculate_fear_greed_index(self, market_data: Dict[str, Any]) -> str:
        """Simplified fear & greed calculation"""
        vix_data = market_data.get("VIX", {})
        if "current_price" in vix_data:
            vix_level = vix_data["current_price"]
            if vix_level > 30:
                return "Fear"
            elif vix_level < 15:
                return "Greed"
            else:
                return "Neutral"
        return "Unknown"

    def _determine_market_trend(self, market_data: Dict[str, Any]) -> str:
        """Determine overall market trend"""
        spy_data = market_data.get("SPY", {})
        if "change_percent" in spy_data:
            change = spy_data["change_percent"]
            if change > 1:
                return "Strong Uptrend"
            elif change > 0:
                return "Uptrend"
            elif change > -1:
                return "Downtrend"
            else:
                return "Strong Downtrend"
        return "Sideways"