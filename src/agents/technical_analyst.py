import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_agent import AnalysisAgent
from ..models.stock_data import TechnicalIndicators, TechnicalSignal
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyst(AnalysisAgent):
    """Advanced technical analysis agent with sophisticated indicators"""

    def __init__(self):
        super().__init__(
            name="TechnicalAnalyst",
            description="Performs comprehensive technical analysis using multiple indicators",
            model="gemini-2.0-flash"
        )

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technical indicators for given stock data"""
        if not self.validate_input(data, ["symbol", "prices"]):
            return {"error": "Invalid input data"}

        symbol = data["symbol"]
        prices = data["prices"]

        self.log_analysis_start(symbol)

        # Convert to DataFrame
        df = self._prepare_dataframe(prices)

        if df.empty:
            return {"error": "No price data available"}

        # Calculate all technical indicators
        indicators = self._calculate_all_indicators(df)

        # Generate technical signal
        signal = await self.generate_signal({"indicators": indicators, "prices": df})

        # Calculate support and resistance levels
        support_resistance = self._calculate_support_resistance(df)

        # Trend analysis
        trend_analysis = self._analyze_trend(df, indicators)

        result = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "indicators": indicators,
            "signal": signal,
            "support_resistance": support_resistance,
            "trend_analysis": trend_analysis,
            "technical_score": self._calculate_technical_score(indicators, signal)
        }

        self.log_analysis_complete(symbol, f"Signal: {signal['overall_signal']}")
        return result

    async def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive trading signal"""
        indicators = data["indicators"]
        prices = data["prices"]

        signals = {}

        # RSI Signal
        rsi = indicators.get("rsi")
        if rsi is not None:
            if rsi < 30:
                signals["rsi"] = "oversold_buy"
            elif rsi > 70:
                signals["rsi"] = "overbought_sell"
            else:
                signals["rsi"] = "neutral"

        # MACD Signal
        macd_data = indicators.get("macd", {})
        if macd_data:
            macd_line = macd_data.get("macd")
            signal_line = macd_data.get("signal")
            if macd_line and signal_line:
                if macd_line > signal_line:
                    signals["macd"] = "bullish"
                else:
                    signals["macd"] = "bearish"

        # Moving Average Signal
        ma_data = indicators.get("moving_averages", {})
        if ma_data:
            sma_20 = ma_data.get("sma_20")
            sma_50 = ma_data.get("sma_50")
            current_price = prices['close'].iloc[-1]

            if sma_20 and sma_50 and current_price:
                if current_price > sma_20 > sma_50:
                    signals["ma"] = "strong_bullish"
                elif current_price > sma_20:
                    signals["ma"] = "bullish"
                elif current_price < sma_20 < sma_50:
                    signals["ma"] = "strong_bearish"
                else:
                    signals["ma"] = "bearish"

        # Bollinger Bands Signal
        bb_data = indicators.get("bollinger_bands", {})
        if bb_data:
            current_price = prices['close'].iloc[-1]
            lower_band = bb_data.get("lower")
            upper_band = bb_data.get("upper")

            if lower_band and upper_band and current_price:
                if current_price <= lower_band:
                    signals["bollinger"] = "oversold_buy"
                elif current_price >= upper_band:
                    signals["bollinger"] = "overbought_sell"
                else:
                    signals["bollinger"] = "neutral"

        # ADX Trend Strength
        adx = indicators.get("adx")
        if adx:
            if adx > 50:
                signals["trend_strength"] = "very_strong"
            elif adx > 25:
                signals["trend_strength"] = "strong"
            else:
                signals["trend_strength"] = "weak"

        # Overall signal calculation
        overall_signal = self._calculate_overall_signal(signals)

        return {
            "individual_signals": signals,
            "overall_signal": overall_signal,
            "confidence": self._calculate_signal_confidence(signals)
        }

    def _prepare_dataframe(self, prices: List[Dict]) -> pd.DataFrame:
        """Convert price data to pandas DataFrame"""
        df = pd.DataFrame(prices)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df.sort_index()

    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive set of technical indicators"""
        indicators = {}

        # RSI
        indicators["rsi"] = self._calculate_rsi(df['close'])

        # MACD
        indicators["macd"] = self._calculate_macd(df['close'])

        # Bollinger Bands
        indicators["bollinger_bands"] = self._calculate_bollinger_bands(df['close'])

        # Moving Averages
        indicators["moving_averages"] = self._calculate_moving_averages(df['close'])

        # Stochastic
        indicators["stochastic"] = self._calculate_stochastic(df['high'], df['low'], df['close'])

        # ADX
        indicators["adx"] = self._calculate_adx(df['high'], df['low'], df['close'])

        # Williams %R
        indicators["williams_r"] = self._calculate_williams_r(df['high'], df['low'], df['close'])

        # CCI
        indicators["cci"] = self._calculate_cci(df['high'], df['low'], df['close'])

        # Volume indicators
        indicators["volume"] = self._calculate_volume_indicators(df)

        return indicators

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else None
        except:
            return None

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line

            return {
                "macd": macd_line.iloc[-1],
                "signal": signal_line.iloc[-1],
                "histogram": histogram.iloc[-1]
            }
        except:
            return {}

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()

            return {
                "upper": (sma + (std * std_dev)).iloc[-1],
                "middle": sma.iloc[-1],
                "lower": (sma - (std * std_dev)).iloc[-1]
            }
        except:
            return {}

    def _calculate_moving_averages(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate various moving averages"""
        try:
            return {
                "sma_10": prices.rolling(window=10).mean().iloc[-1],
                "sma_20": prices.rolling(window=20).mean().iloc[-1],
                "sma_50": prices.rolling(window=50).mean().iloc[-1],
                "sma_200": prices.rolling(window=200).mean().iloc[-1],
                "ema_12": prices.ewm(span=12).mean().iloc[-1],
                "ema_26": prices.ewm(span=26).mean().iloc[-1]
            }
        except:
            return {}

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """Calculate Stochastic Oscillator"""
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()

            return {
                "k": k_percent.iloc[-1],
                "d": d_percent.iloc[-1]
            }
        except:
            return {}

    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Average Directional Index"""
        try:
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0

            tr1 = pd.DataFrame(high - low)
            tr2 = pd.DataFrame(abs(high - close.shift(1)))
            tr3 = pd.DataFrame(abs(low - close.shift(1)))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean())
            minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean()))

            dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
            adx = dx.ewm(alpha=1/period).mean()

            return adx.iloc[-1] if not adx.empty else None
        except:
            return None

    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Williams %R"""
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return williams_r.iloc[-1] if not williams_r.empty else None
        except:
            return None

    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> Optional[float]:
        """Calculate Commodity Channel Index"""
        try:
            tp = (high + low + close) / 3
            sma = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad())
            cci = (tp - sma) / (0.015 * mad)
            return cci.iloc[-1] if not cci.empty else None
        except:
            return None

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based indicators"""
        try:
            volume = df['volume']
            close = df['close']

            return {
                "volume_sma_20": volume.rolling(window=20).mean().iloc[-1],
                "volume_ratio": volume.iloc[-1] / volume.rolling(window=20).mean().iloc[-1],
                "price_volume_trend": ((close.diff() / close.shift(1)) * volume).cumsum().iloc[-1]
            }
        except:
            return {}

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate support and resistance levels"""
        try:
            high = df['high']
            low = df['low']

            # Pivot points
            recent_highs = high.rolling(window=20).max()
            recent_lows = low.rolling(window=20).min()

            resistance_levels = [recent_highs.iloc[-1], recent_highs.iloc[-5], recent_highs.iloc[-10]]
            support_levels = [recent_lows.iloc[-1], recent_lows.iloc[-5], recent_lows.iloc[-10]]

            return {
                "resistance": sorted(set(resistance_levels), reverse=True),
                "support": sorted(set(support_levels))
            }
        except:
            return {"resistance": [], "support": []}

    def _analyze_trend(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive trend analysis"""
        try:
            close = df['close']
            ma_data = indicators.get("moving_averages", {})

            # Trend direction
            if ma_data.get("sma_20", 0) > ma_data.get("sma_50", 0):
                short_term_trend = "up"
            else:
                short_term_trend = "down"

            if ma_data.get("sma_50", 0) > ma_data.get("sma_200", 0):
                long_term_trend = "up"
            else:
                long_term_trend = "down"

            # Trend strength
            adx = indicators.get("adx", 0)
            if adx > 50:
                trend_strength = "very_strong"
            elif adx > 25:
                trend_strength = "strong"
            elif adx > 20:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"

            return {
                "short_term": short_term_trend,
                "long_term": long_term_trend,
                "strength": trend_strength,
                "adx_value": adx
            }
        except:
            return {"short_term": "unknown", "long_term": "unknown", "strength": "unknown"}

    def _calculate_overall_signal(self, signals: Dict[str, str]) -> str:
        """Calculate overall trading signal from individual signals"""
        bullish_signals = ["oversold_buy", "bullish", "strong_bullish"]
        bearish_signals = ["overbought_sell", "bearish", "strong_bearish"]

        bullish_count = sum(1 for signal in signals.values() if signal in bullish_signals)
        bearish_count = sum(1 for signal in signals.values() if signal in bearish_signals)

        if bullish_count > bearish_count + 1:
            return TechnicalSignal.BUY
        elif bullish_count > bearish_count:
            return TechnicalSignal.HOLD
        elif bearish_count > bullish_count + 1:
            return TechnicalSignal.SELL
        elif bearish_count > bullish_count:
            return TechnicalSignal.HOLD
        else:
            return TechnicalSignal.HOLD

    def _calculate_signal_confidence(self, signals: Dict[str, str]) -> float:
        """Calculate confidence level for the signal"""
        total_signals = len(signals)
        if total_signals == 0:
            return 0.0

        # Count aligned signals
        bullish_signals = ["oversold_buy", "bullish", "strong_bullish"]
        bearish_signals = ["overbought_sell", "bearish", "strong_bearish"]

        bullish_count = sum(1 for signal in signals.values() if signal in bullish_signals)
        bearish_count = sum(1 for signal in signals.values() if signal in bearish_signals)

        max_aligned = max(bullish_count, bearish_count)
        confidence = (max_aligned / total_signals) * 100

        return min(confidence, 100.0)

    def _calculate_technical_score(self, indicators: Dict[str, Any], signal: Dict[str, Any]) -> float:
        """Calculate overall technical analysis score (0-100)"""
        score = 50.0  # Base score

        # Adjust based on signal strength
        confidence = signal.get("confidence", 50)
        overall_signal = signal.get("overall_signal", TechnicalSignal.HOLD)

        if overall_signal == TechnicalSignal.BUY:
            score += (confidence * 0.5)
        elif overall_signal == TechnicalSignal.SELL:
            score -= (confidence * 0.5)

        # Adjust based on trend strength
        adx = indicators.get("adx", 25)
        if adx > 50:
            score += 10
        elif adx > 25:
            score += 5

        # RSI adjustment
        rsi = indicators.get("rsi")
        if rsi:
            if 30 <= rsi <= 70:
                score += 5  # Neutral RSI is good
            elif rsi < 20 or rsi > 80:
                score -= 10  # Extreme levels

        return max(0, min(100, score))