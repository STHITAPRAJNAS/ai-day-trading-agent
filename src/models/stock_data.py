from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MarketSentiment(str, Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"

class TechnicalSignal(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class StockPrice(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

class TechnicalIndicators(BaseModel):
    symbol: str
    timestamp: datetime
    rsi: Optional[float] = None
    macd: Optional[Dict[str, float]] = None
    bollinger_bands: Optional[Dict[str, float]] = None
    moving_averages: Optional[Dict[str, float]] = None
    stochastic: Optional[Dict[str, float]] = None
    adx: Optional[float] = None
    williams_r: Optional[float] = None
    cci: Optional[float] = None

class FundamentalData(BaseModel):
    symbol: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    debt_to_equity: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None

class NewsItem(BaseModel):
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    symbols: List[str]
    sentiment_score: Optional[float] = None
    impact_score: Optional[float] = None

class IndustryTrend(BaseModel):
    industry: str
    sector: str
    trend_direction: str
    strength: float
    timeframe: str
    key_factors: List[str]
    related_symbols: List[str]

class TradingStrategy(BaseModel):
    strategy_type: str
    entry_price: float
    exit_price: Optional[float] = None
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    confidence_score: float
    timeframe: str
    reasoning: str

class StockAnalysis(BaseModel):
    symbol: str
    analysis_timestamp: datetime
    current_price: float

    technical_score: float = Field(..., ge=0, le=100)
    fundamental_score: float = Field(..., ge=0, le=100)
    sentiment_score: float = Field(..., ge=0, le=100)
    overall_score: float = Field(..., ge=0, le=100)

    technical_signal: TechnicalSignal
    market_sentiment: MarketSentiment

    entry_strategy: TradingStrategy
    exit_strategy: TradingStrategy

    risk_level: str
    price_target_1w: Optional[float] = None
    price_target_1m: Optional[float] = None
    price_target_3m: Optional[float] = None

    key_insights: List[str]
    risks: List[str]
    catalysts: List[str]

class DailyStockPicks(BaseModel):
    date: datetime
    picks: List[StockAnalysis]
    market_overview: str
    key_themes: List[str]
    risk_factors: List[str]
    generated_at: datetime