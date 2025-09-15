import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
from .base_agent import AnalysisAgent
from ..models.stock_data import TechnicalSignal
import logging

logger = logging.getLogger(__name__)

class DayTradingAnalyst(AnalysisAgent):
    """Specialized agent for day trading analysis with 10% profit targets"""

    def __init__(self):
        super().__init__(
            name="DayTradingAnalyst",
            description="Identifies day trading opportunities with 10% upside potential",
            model="gemini-2.0-flash"
        )

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stocks for day trading opportunities"""
        if not self.validate_input(data, ["symbol"]):
            return {"error": "Invalid input data"}

        symbol = data["symbol"]
        self.log_analysis_start(symbol)

        try:
            # Get intraday data (1min, 5min, 15min intervals)
            intraday_data = await self._get_intraday_data(symbol)

            if not intraday_data:
                return {"error": "Unable to fetch intraday data"}

            # Perform day trading specific analysis
            analysis_result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "day_trading_score": 0,
                "volatility_analysis": {},
                "momentum_indicators": {},
                "liquidity_analysis": {},
                "gap_analysis": {},
                "support_resistance": {},
                "entry_signals": [],
                "profit_targets": {},
                "risk_assessment": {},
                "trading_recommendation": "HOLD"
            }

            # Analyze different timeframes
            for interval in ["1m", "5m", "15m"]:
                if interval in intraday_data:
                    df = intraday_data[interval]

                    # Calculate day trading indicators
                    indicators = self._calculate_day_trading_indicators(df, interval)
                    analysis_result[f"indicators_{interval}"] = indicators

            # Volatility analysis for day trading
            analysis_result["volatility_analysis"] = self._analyze_volatility(intraday_data)

            # Momentum analysis
            analysis_result["momentum_indicators"] = self._analyze_momentum(intraday_data)

            # Liquidity analysis
            analysis_result["liquidity_analysis"] = self._analyze_liquidity(intraday_data)

            # Gap analysis (pre-market vs market open)
            analysis_result["gap_analysis"] = self._analyze_gaps(symbol)

            # Support/Resistance for day trading
            analysis_result["support_resistance"] = self._calculate_intraday_levels(intraday_data)

            # Generate entry signals
            analysis_result["entry_signals"] = self._generate_day_trading_signals(intraday_data)

            # Calculate 10% profit targets
            analysis_result["profit_targets"] = self._calculate_profit_targets(symbol, intraday_data)

            # Risk assessment
            analysis_result["risk_assessment"] = self._assess_day_trading_risk(intraday_data)

            # Overall day trading score
            analysis_result["day_trading_score"] = self._calculate_day_trading_score(analysis_result)

            # Trading recommendation
            analysis_result["trading_recommendation"] = self._generate_trading_recommendation(analysis_result)

            self.log_analysis_complete(symbol, f"Score: {analysis_result['day_trading_score']:.1f}")
            return analysis_result

        except Exception as e:
            logger.error(f"Day trading analysis failed for {symbol}: {str(e)}")
            return {"error": str(e)}

    async def _get_intraday_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get intraday data for multiple timeframes"""
        intraday_data = {}

        try:
            ticker = yf.Ticker(symbol)

            # Get different intervals for day trading
            intervals = {
                "1m": "1d",    # 1-minute data for last day
                "5m": "5d",    # 5-minute data for last 5 days
                "15m": "5d",   # 15-minute data for last 5 days
                "1h": "5d"     # 1-hour data for context
            }

            for interval, period in intervals.items():
                try:
                    data = ticker.history(period=period, interval=interval)
                    if not data.empty:
                        intraday_data[interval] = data
                except Exception as e:
                    logger.warning(f"Failed to get {interval} data for {symbol}: {str(e)}")

        except Exception as e:
            logger.error(f"Failed to get intraday data for {symbol}: {str(e)}")

        return intraday_data

    def _calculate_day_trading_indicators(self, df: pd.DataFrame, interval: str) -> Dict[str, Any]:
        """Calculate indicators optimized for day trading"""
        if df.empty:
            return {}

        indicators = {}

        try:
            # Short-term moving averages
            indicators["ema_9"] = df['Close'].ewm(span=9).mean().iloc[-1]
            indicators["ema_21"] = df['Close'].ewm(span=21).mean().iloc[-1]

            # VWAP (Volume Weighted Average Price) - crucial for day trading
            indicators["vwap"] = self._calculate_vwap(df)

            # RSI with short period
            indicators["rsi_14"] = self._calculate_rsi(df['Close'], 14)
            indicators["rsi_9"] = self._calculate_rsi(df['Close'], 9)  # Faster RSI

            # Stochastic for quick reversals
            stoch = self._calculate_stochastic(df['High'], df['Low'], df['Close'], 14, 3)
            indicators["stoch_k"] = stoch.get("k")
            indicators["stoch_d"] = stoch.get("d")

            # Williams %R for overbought/oversold
            indicators["williams_r"] = self._calculate_williams_r(df['High'], df['Low'], df['Close'], 14)

            # Average True Range for volatility
            indicators["atr"] = self._calculate_atr(df, 14)

            # Money Flow Index
            indicators["mfi"] = self._calculate_mfi(df, 14)

            # Price Rate of Change
            indicators["roc"] = self._calculate_roc(df['Close'], 10)

            # Volume indicators
            indicators["volume_sma"] = df['Volume'].rolling(window=20).mean().iloc[-1]
            indicators["volume_ratio"] = df['Volume'].iloc[-1] / indicators["volume_sma"]

            # Bollinger Bands for volatility breakouts
            bb = self._calculate_bollinger_bands(df['Close'], 20, 2)
            indicators["bb_upper"] = bb.get("upper")
            indicators["bb_lower"] = bb.get("lower")
            indicators["bb_percent"] = ((df['Close'].iloc[-1] - bb.get("lower", 0)) /
                                      (bb.get("upper", 1) - bb.get("lower", 1))) * 100 if bb.get("upper") and bb.get("lower") else 50

        except Exception as e:
            logger.error(f"Error calculating day trading indicators: {str(e)}")

        return indicators

    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
            return vwap.iloc[-1]
        except:
            return 0

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Money Flow Index"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']

            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()

            mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
            return mfi.iloc[-1]
        except:
            return 50

    def _calculate_roc(self, prices: pd.Series, period: int) -> float:
        """Calculate Rate of Change"""
        try:
            roc = ((prices.iloc[-1] - prices.iloc[-period-1]) / prices.iloc[-period-1]) * 100
            return roc
        except:
            return 0

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Average True Range"""
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr.iloc[-1]
        except:
            return 0

    def _analyze_volatility(self, intraday_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze volatility for day trading opportunities"""
        volatility_analysis = {}

        try:
            if "5m" in intraday_data:
                df = intraday_data["5m"]

                # Calculate various volatility measures
                returns = df['Close'].pct_change().dropna()

                volatility_analysis.update({
                    "daily_volatility": returns.std() * np.sqrt(252) * 100,  # Annualized
                    "intraday_volatility": returns.std() * np.sqrt(78) * 100,  # 5-min periods in trading day
                    "current_range": ((df['High'].iloc[-1] - df['Low'].iloc[-1]) / df['Close'].iloc[-1]) * 100,
                    "average_range": ((df['High'] - df['Low']) / df['Close']).mean() * 100,
                    "volatility_percentile": self._calculate_volatility_percentile(df),
                    "is_high_volatility": returns.std() > returns.rolling(window=20).std().mean() * 1.5
                })

        except Exception as e:
            logger.error(f"Error analyzing volatility: {str(e)}")

        return volatility_analysis

    def _calculate_volatility_percentile(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """Calculate current volatility percentile"""
        try:
            returns = df['Close'].pct_change().dropna()
            current_vol = returns.tail(5).std()  # Last 5 periods volatility
            historical_vols = returns.rolling(window=5).std().dropna()

            if len(historical_vols) > 0:
                percentile = (historical_vols < current_vol).sum() / len(historical_vols) * 100
                return percentile
        except:
            pass
        return 50

    def _analyze_momentum(self, intraday_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze momentum for day trading"""
        momentum = {}

        try:
            if "5m" in intraday_data:
                df = intraday_data["5m"]

                # Price momentum
                momentum["price_momentum_1h"] = self._calculate_price_momentum(df, 12)  # 12 * 5min = 1hr
                momentum["price_momentum_30m"] = self._calculate_price_momentum(df, 6)   # 6 * 5min = 30min
                momentum["price_momentum_15m"] = self._calculate_price_momentum(df, 3)   # 3 * 5min = 15min

                # Volume momentum
                momentum["volume_momentum"] = self._calculate_volume_momentum(df)

                # Momentum quality
                momentum["momentum_quality"] = self._assess_momentum_quality(df)

                # Acceleration
                momentum["price_acceleration"] = self._calculate_price_acceleration(df)

        except Exception as e:
            logger.error(f"Error analyzing momentum: {str(e)}")

        return momentum

    def _calculate_price_momentum(self, df: pd.DataFrame, periods: int) -> float:
        """Calculate price momentum over specified periods"""
        try:
            if len(df) > periods:
                current_price = df['Close'].iloc[-1]
                past_price = df['Close'].iloc[-periods-1]
                momentum = ((current_price - past_price) / past_price) * 100
                return momentum
        except:
            pass
        return 0

    def _calculate_volume_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume momentum indicators"""
        try:
            recent_volume = df['Volume'].tail(6).mean()  # Last 30 minutes
            average_volume = df['Volume'].rolling(window=78).mean().iloc[-1]  # Daily average

            return {
                "volume_ratio": recent_volume / average_volume if average_volume > 0 else 1,
                "volume_trend": (df['Volume'].tail(6).iloc[-1] - df['Volume'].tail(6).iloc[0]) / df['Volume'].tail(6).iloc[0] * 100
            }
        except:
            return {"volume_ratio": 1, "volume_trend": 0}

    def _assess_momentum_quality(self, df: pd.DataFrame) -> str:
        """Assess the quality of current momentum"""
        try:
            # Check if momentum is supported by volume
            price_momentum = self._calculate_price_momentum(df, 6)
            volume_momentum = self._calculate_volume_momentum(df)

            if abs(price_momentum) > 2 and volume_momentum["volume_ratio"] > 1.5:
                return "HIGH_QUALITY"
            elif abs(price_momentum) > 1 and volume_momentum["volume_ratio"] > 1.2:
                return "MEDIUM_QUALITY"
            else:
                return "LOW_QUALITY"
        except:
            return "UNKNOWN"

    def _calculate_price_acceleration(self, df: pd.DataFrame) -> float:
        """Calculate price acceleration (change in momentum)"""
        try:
            if len(df) >= 12:
                momentum_current = self._calculate_price_momentum(df, 3)
                momentum_previous = self._calculate_price_momentum(df.iloc[:-3], 3)
                acceleration = momentum_current - momentum_previous
                return acceleration
        except:
            pass
        return 0

    def _analyze_liquidity(self, intraday_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze liquidity for day trading"""
        liquidity = {}

        try:
            if "5m" in intraday_data:
                df = intraday_data["5m"]

                # Average volume analysis
                avg_volume = df['Volume'].mean()
                recent_volume = df['Volume'].tail(12).mean()  # Last hour

                # Bid-ask spread estimation (using high-low as proxy)
                spread_estimate = ((df['High'] - df['Low']) / df['Close']).mean() * 100

                liquidity.update({
                    "average_volume": avg_volume,
                    "recent_volume": recent_volume,
                    "volume_consistency": df['Volume'].std() / avg_volume if avg_volume > 0 else 0,
                    "estimated_spread": spread_estimate,
                    "liquidity_score": self._calculate_liquidity_score(df),
                    "is_liquid": avg_volume > 100000 and spread_estimate < 0.5  # Basic liquidity criteria
                })

        except Exception as e:
            logger.error(f"Error analyzing liquidity: {str(e)}")

        return liquidity

    def _calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """Calculate a liquidity score from 0-100"""
        try:
            # Factors: volume, spread, consistency
            avg_volume = df['Volume'].mean()
            spread = ((df['High'] - df['Low']) / df['Close']).mean()
            volume_consistency = 1 - (df['Volume'].std() / avg_volume) if avg_volume > 0 else 0

            # Normalize and weight factors
            volume_score = min(100, avg_volume / 10000)  # 1M volume = 100 points
            spread_score = max(0, 100 - spread * 1000)   # Lower spread = higher score
            consistency_score = volume_consistency * 100

            liquidity_score = (volume_score * 0.5 + spread_score * 0.3 + consistency_score * 0.2)
            return max(0, min(100, liquidity_score))
        except:
            return 50

    def _analyze_gaps(self, symbol: str) -> Dict[str, Any]:
        """Analyze pre-market gaps"""
        gap_analysis = {}

        try:
            ticker = yf.Ticker(symbol)

            # Get recent daily data
            daily_data = ticker.history(period="10d")

            if len(daily_data) >= 2:
                yesterday_close = daily_data['Close'].iloc[-2]
                today_open = daily_data['Open'].iloc[-1]
                current_price = daily_data['Close'].iloc[-1]

                gap_percent = ((today_open - yesterday_close) / yesterday_close) * 100

                gap_analysis.update({
                    "gap_percent": gap_percent,
                    "gap_type": "UP" if gap_percent > 0.5 else "DOWN" if gap_percent < -0.5 else "NONE",
                    "gap_size": abs(gap_percent),
                    "is_significant_gap": abs(gap_percent) > 2,
                    "gap_filled": self._check_gap_fill(yesterday_close, today_open, current_price),
                    "yesterday_close": yesterday_close,
                    "today_open": today_open,
                    "current_price": current_price
                })

        except Exception as e:
            logger.error(f"Error analyzing gaps for {symbol}: {str(e)}")

        return gap_analysis

    def _check_gap_fill(self, yesterday_close: float, today_open: float, current_price: float) -> bool:
        """Check if gap has been filled"""
        if today_open > yesterday_close:  # Gap up
            return current_price <= yesterday_close
        elif today_open < yesterday_close:  # Gap down
            return current_price >= yesterday_close
        return False

    def _calculate_intraday_levels(self, intraday_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate key intraday support and resistance levels"""
        levels = {}

        try:
            if "5m" in intraday_data:
                df = intraday_data["5m"]

                # VWAP as key level
                vwap = self._calculate_vwap(df)

                # Pivot points
                pivot_points = self._calculate_pivot_points(df)

                # Recent highs and lows
                recent_high = df['High'].tail(78).max()  # Last 6.5 hours
                recent_low = df['Low'].tail(78).min()

                # Volume profile (simplified)
                volume_levels = self._calculate_volume_levels(df)

                levels.update({
                    "vwap": vwap,
                    "pivot_points": pivot_points,
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                    "volume_levels": volume_levels,
                    "key_resistance": [recent_high, pivot_points.get("r1", 0), vwap + (vwap * 0.01)],
                    "key_support": [recent_low, pivot_points.get("s1", 0), vwap - (vwap * 0.01)]
                })

        except Exception as e:
            logger.error(f"Error calculating intraday levels: {str(e)}")

        return levels

    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate daily pivot points"""
        try:
            # Use yesterday's or current session data
            high = df['High'].max()
            low = df['Low'].min()
            close = df['Close'].iloc[-1]

            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)

            return {
                "pivot": pivot,
                "r1": r1, "r2": r2,
                "s1": s1, "s2": s2
            }
        except:
            return {}

    def _calculate_volume_levels(self, df: pd.DataFrame) -> List[float]:
        """Calculate price levels with high volume (volume profile)"""
        try:
            # Group by price ranges and sum volume
            df['price_bucket'] = pd.cut(df['Close'], bins=20)
            volume_by_price = df.groupby('price_bucket')['Volume'].sum().sort_values(ascending=False)

            # Get top volume areas
            top_volume_ranges = volume_by_price.head(3).index
            volume_levels = [(range.left + range.right) / 2 for range in top_volume_ranges]

            return volume_levels
        except:
            return []

    def _generate_day_trading_signals(self, intraday_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate specific day trading entry signals"""
        signals = []

        try:
            if "5m" in intraday_data:
                df = intraday_data["5m"]
                current_price = df['Close'].iloc[-1]

                # VWAP signals
                vwap = self._calculate_vwap(df)
                if current_price > vwap * 1.002:  # Above VWAP with momentum
                    signals.append({
                        "type": "VWAP_BREAKOUT",
                        "strength": "STRONG" if current_price > vwap * 1.005 else "MEDIUM",
                        "entry_price": current_price,
                        "target": current_price * 1.1,  # 10% target
                        "stop": vwap * 0.998
                    })

                # Momentum breakout signals
                indicators = self._calculate_day_trading_indicators(df, "5m")

                if (indicators.get("rsi_9", 50) > 70 and
                    indicators.get("volume_ratio", 1) > 1.5 and
                    current_price > indicators.get("ema_9", 0)):

                    signals.append({
                        "type": "MOMENTUM_BREAKOUT",
                        "strength": "STRONG",
                        "entry_price": current_price,
                        "target": current_price * 1.1,
                        "stop": indicators.get("ema_9", current_price * 0.98)
                    })

                # Oversold bounce signals
                if (indicators.get("rsi_9", 50) < 30 and
                    indicators.get("williams_r", -50) < -80 and
                    indicators.get("volume_ratio", 1) > 1.2):

                    signals.append({
                        "type": "OVERSOLD_BOUNCE",
                        "strength": "MEDIUM",
                        "entry_price": current_price,
                        "target": current_price * 1.08,  # Smaller target for bounce
                        "stop": current_price * 0.97
                    })

        except Exception as e:
            logger.error(f"Error generating day trading signals: {str(e)}")

        return signals

    def _calculate_profit_targets(self, symbol: str, intraday_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate 10% profit targets with optimal entry/exit points"""
        targets = {}

        try:
            if "5m" in intraday_data:
                df = intraday_data["5m"]
                current_price = df['Close'].iloc[-1]

                # 10% profit target
                profit_target = current_price * 1.10

                # Calculate optimal entry based on support levels
                support_levels = self._calculate_intraday_levels(intraday_data).get("key_support", [])
                optimal_entry = min([level for level in support_levels if level <= current_price * 1.02],
                                  default=current_price * 0.998)

                # Calculate stop loss (2% or below nearest support)
                stop_loss = min(optimal_entry * 0.98,
                              min([level for level in support_levels if level < optimal_entry],
                                  default=optimal_entry * 0.98))

                # Risk-reward calculation
                risk = optimal_entry - stop_loss
                reward = profit_target - optimal_entry
                risk_reward_ratio = reward / risk if risk > 0 else 0

                # Probability of success estimation
                volatility = self._analyze_volatility(intraday_data)
                success_probability = self._estimate_success_probability(
                    current_price, profit_target, volatility
                )

                targets.update({
                    "current_price": current_price,
                    "optimal_entry": optimal_entry,
                    "profit_target": profit_target,
                    "stop_loss": stop_loss,
                    "risk_amount": risk,
                    "reward_amount": reward,
                    "risk_reward_ratio": risk_reward_ratio,
                    "success_probability": success_probability,
                    "expected_value": (reward * success_probability) - (risk * (1 - success_probability)),
                    "time_to_target": self._estimate_time_to_target(intraday_data),
                    "is_favorable": risk_reward_ratio >= 2 and success_probability >= 0.6
                })

        except Exception as e:
            logger.error(f"Error calculating profit targets: {str(e)}")

        return targets

    def _estimate_success_probability(self, current_price: float, target: float, volatility: Dict) -> float:
        """Estimate probability of reaching 10% target"""
        try:
            target_move = ((target - current_price) / current_price) * 100
            daily_volatility = volatility.get("daily_volatility", 20)

            # Higher volatility = higher chance of reaching target
            # Adjust based on current market conditions
            base_probability = min(0.8, max(0.3, (daily_volatility / 30) * 0.7))

            # Adjust based on momentum
            if volatility.get("is_high_volatility", False):
                base_probability *= 1.2

            return min(0.9, base_probability)
        except:
            return 0.5

    def _estimate_time_to_target(self, intraday_data: Dict[str, pd.DataFrame]) -> str:
        """Estimate time to reach profit target"""
        try:
            if "5m" in intraday_data:
                df = intraday_data["5m"]
                recent_volatility = df['Close'].pct_change().tail(12).std()

                if recent_volatility > 0.005:  # High volatility
                    return "1-3 hours"
                elif recent_volatility > 0.003:
                    return "3-6 hours"
                else:
                    return "Full day"
        except:
            pass
        return "Unknown"

    def _assess_day_trading_risk(self, intraday_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess risks specific to day trading"""
        risk_assessment = {}

        try:
            if "5m" in intraday_data:
                df = intraday_data["5m"]

                # Volatility risk
                volatility = self._analyze_volatility(intraday_data)
                risk_assessment["volatility_risk"] = "HIGH" if volatility.get("is_high_volatility") else "MEDIUM"

                # Liquidity risk
                liquidity = self._analyze_liquidity(intraday_data)
                risk_assessment["liquidity_risk"] = "LOW" if liquidity.get("is_liquid") else "HIGH"

                # Time decay risk (market close approaching)
                current_hour = datetime.now().hour
                if current_hour >= 15:  # After 3 PM EST
                    risk_assessment["time_risk"] = "HIGH"
                elif current_hour >= 14:  # After 2 PM EST
                    risk_assessment["time_risk"] = "MEDIUM"
                else:
                    risk_assessment["time_risk"] = "LOW"

                # Gap risk
                gap_analysis = self._analyze_gaps("")  # Symbol will be filled by caller
                risk_assessment["gap_risk"] = "HIGH" if gap_analysis.get("is_significant_gap") else "LOW"

                # Overall risk level
                risk_factors = [
                    risk_assessment.get("volatility_risk") == "HIGH",
                    risk_assessment.get("liquidity_risk") == "HIGH",
                    risk_assessment.get("time_risk") == "HIGH",
                    risk_assessment.get("gap_risk") == "HIGH"
                ]

                high_risk_count = sum(risk_factors)
                if high_risk_count >= 3:
                    risk_assessment["overall_risk"] = "VERY_HIGH"
                elif high_risk_count >= 2:
                    risk_assessment["overall_risk"] = "HIGH"
                elif high_risk_count >= 1:
                    risk_assessment["overall_risk"] = "MEDIUM"
                else:
                    risk_assessment["overall_risk"] = "LOW"

        except Exception as e:
            logger.error(f"Error assessing day trading risk: {str(e)}")

        return risk_assessment

    def _calculate_day_trading_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate overall day trading score (0-100)"""
        score = 0

        try:
            # Volatility score (30%)
            volatility = analysis_result.get("volatility_analysis", {})
            if volatility.get("is_high_volatility"):
                score += 30
            elif volatility.get("volatility_percentile", 50) > 60:
                score += 20
            else:
                score += 10

            # Momentum score (25%)
            momentum = analysis_result.get("momentum_indicators", {})
            momentum_quality = momentum.get("momentum_quality", "LOW_QUALITY")
            if momentum_quality == "HIGH_QUALITY":
                score += 25
            elif momentum_quality == "MEDIUM_QUALITY":
                score += 15
            else:
                score += 5

            # Liquidity score (20%)
            liquidity = analysis_result.get("liquidity_analysis", {})
            if liquidity.get("is_liquid"):
                score += 20
            else:
                score += 5

            # Signal strength (15%)
            signals = analysis_result.get("entry_signals", [])
            strong_signals = [s for s in signals if s.get("strength") == "STRONG"]
            if strong_signals:
                score += 15
            elif signals:
                score += 10

            # Profit target favorability (10%)
            profit_targets = analysis_result.get("profit_targets", {})
            if profit_targets.get("is_favorable"):
                score += 10
            elif profit_targets.get("risk_reward_ratio", 0) >= 1.5:
                score += 5

        except Exception as e:
            logger.error(f"Error calculating day trading score: {str(e)}")

        return min(100, score)

    def _generate_trading_recommendation(self, analysis_result: Dict[str, Any]) -> str:
        """Generate final trading recommendation"""
        try:
            score = analysis_result.get("day_trading_score", 0)
            risk_assessment = analysis_result.get("risk_assessment", {})
            overall_risk = risk_assessment.get("overall_risk", "MEDIUM")

            if score >= 80 and overall_risk in ["LOW", "MEDIUM"]:
                return "STRONG_BUY"
            elif score >= 65 and overall_risk != "VERY_HIGH":
                return "BUY"
            elif score >= 50:
                return "WEAK_BUY"
            elif score >= 35:
                return "HOLD"
            else:
                return "AVOID"

        except:
            return "HOLD"

    # Helper methods from technical analyst (reused)
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else None
        except:
            return None

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

    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Williams %R"""
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return williams_r.iloc[-1] if not williams_r.empty else None
        except:
            return None

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