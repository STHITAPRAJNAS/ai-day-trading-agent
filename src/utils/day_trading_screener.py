from typing import List, Dict, Any, Optional
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DayTradingScreener:
    """Screener optimized for day trading opportunities with 10% upside potential"""

    def __init__(self):
        self.min_volume = 1000000  # Minimum 1M average volume
        self.min_price = 5.0       # Minimum $5 per share
        self.max_price = 500.0     # Maximum $500 per share
        self.min_volatility = 15   # Minimum 15% annualized volatility
        self.max_volatility = 100  # Maximum 100% annualized volatility

    def screen_for_day_trading(self, symbols: List[str], max_results: int = 20) -> List[Dict[str, Any]]:
        """Screen stocks for day trading opportunities"""
        candidates = []

        logger.info(f"Screening {len(symbols)} symbols for day trading opportunities...")

        for symbol in symbols:
            try:
                analysis = self._analyze_symbol_for_day_trading(symbol)
                if analysis and analysis.get("is_suitable"):
                    candidates.append(analysis)

                    if len(candidates) >= max_results * 2:  # Get extra to rank and filter
                        break

            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {str(e)}")

        # Rank by day trading score
        candidates.sort(key=lambda x: x.get("day_trading_score", 0), reverse=True)

        logger.info(f"Found {len(candidates)} day trading candidates")
        return candidates[:max_results]

    def _analyze_symbol_for_day_trading(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single symbol for day trading suitability"""
        try:
            ticker = yf.Ticker(symbol)

            # Get basic info
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')

            if not current_price:
                return None

            # Price filter
            if not (self.min_price <= current_price <= self.max_price):
                return None

            # Get volume data
            hist_data = ticker.history(period="10d", interval="1d")
            if hist_data.empty:
                return None

            avg_volume = hist_data['Volume'].mean()
            if avg_volume < self.min_volume:
                return None

            # Get intraday data for volatility analysis
            intraday_data = self._get_intraday_sample(ticker)

            # Calculate key metrics
            analysis = {
                "symbol": symbol,
                "current_price": current_price,
                "average_volume": avg_volume,
                "market_cap": info.get('marketCap'),
                "float_shares": info.get('floatShares'),
                "sector": info.get('sector'),
                "industry": info.get('industry')
            }

            # Volatility analysis
            volatility_metrics = self._calculate_volatility_metrics(hist_data, intraday_data)
            analysis.update(volatility_metrics)

            # Volatility filter
            if not (self.min_volatility <= volatility_metrics.get("annualized_volatility", 0) <= self.max_volatility):
                return None

            # Momentum analysis
            momentum_metrics = self._calculate_momentum_metrics(hist_data, intraday_data)
            analysis.update(momentum_metrics)

            # Liquidity analysis
            liquidity_metrics = self._calculate_liquidity_metrics(hist_data, intraday_data)
            analysis.update(liquidity_metrics)

            # Technical setup analysis
            technical_setup = self._analyze_technical_setup(hist_data, intraday_data)
            analysis.update(technical_setup)

            # Gap analysis
            gap_analysis = self._analyze_gap_potential(hist_data)
            analysis.update(gap_analysis)

            # 10% profit potential
            profit_potential = self._assess_profit_potential(analysis)
            analysis.update(profit_potential)

            # Overall scoring
            day_trading_score = self._calculate_day_trading_score(analysis)
            analysis["day_trading_score"] = day_trading_score

            # Suitability determination
            analysis["is_suitable"] = self._determine_suitability(analysis)

            # Risk assessment
            analysis["risk_level"] = self._assess_risk_level(analysis)

            # Time-sensitive factors
            analysis["time_factors"] = self._analyze_time_factors()

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None

    def _get_intraday_sample(self, ticker) -> Optional[pd.DataFrame]:
        """Get sample intraday data for analysis"""
        try:
            # Try to get recent intraday data
            intraday = ticker.history(period="1d", interval="5m")
            if not intraday.empty:
                return intraday

            # Fallback to hourly data
            return ticker.history(period="2d", interval="1h")
        except:
            return None

    def _calculate_volatility_metrics(self, daily_data: pd.DataFrame, intraday_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate comprehensive volatility metrics"""
        metrics = {}

        try:
            # Daily volatility
            daily_returns = daily_data['Close'].pct_change().dropna()
            annualized_vol = daily_returns.std() * np.sqrt(252) * 100

            metrics.update({
                "annualized_volatility": annualized_vol,
                "daily_volatility_avg": daily_returns.std() * 100,
                "volatility_percentile": self._calculate_volatility_percentile(daily_returns),
                "is_high_volatility": annualized_vol > 30
            })

            # Intraday volatility if available
            if intraday_data is not None and not intraday_data.empty:
                intraday_returns = intraday_data['Close'].pct_change().dropna()
                intraday_vol = intraday_returns.std() * np.sqrt(78) * 100  # 78 5-min periods in trading day

                metrics.update({
                    "intraday_volatility": intraday_vol,
                    "current_day_range": ((intraday_data['High'].max() - intraday_data['Low'].min()) /
                                         intraday_data['Close'].iloc[-1]) * 100 if len(intraday_data) > 0 else 0
                })

        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {str(e)}")

        return metrics

    def _calculate_volatility_percentile(self, returns: pd.Series, lookback: int = 20) -> float:
        """Calculate current volatility percentile"""
        try:
            rolling_vol = returns.rolling(window=5).std()
            current_vol = rolling_vol.iloc[-1]
            historical_vols = rolling_vol.dropna()

            if len(historical_vols) > 0:
                percentile = (historical_vols < current_vol).sum() / len(historical_vols) * 100
                return percentile
        except:
            pass
        return 50

    def _calculate_momentum_metrics(self, daily_data: pd.DataFrame, intraday_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate momentum metrics for day trading"""
        metrics = {}

        try:
            # Price momentum (various timeframes)
            current_price = daily_data['Close'].iloc[-1]

            if len(daily_data) >= 5:
                price_5d_ago = daily_data['Close'].iloc[-5]
                momentum_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100
                metrics["momentum_5d"] = momentum_5d

            if len(daily_data) >= 2:
                price_1d_ago = daily_data['Close'].iloc[-2]
                momentum_1d = ((current_price - price_1d_ago) / price_1d_ago) * 100
                metrics["momentum_1d"] = momentum_1d

            # Volume momentum
            recent_volume = daily_data['Volume'].tail(3).mean()
            avg_volume = daily_data['Volume'].mean()
            volume_momentum = (recent_volume / avg_volume) if avg_volume > 0 else 1

            metrics.update({
                "volume_momentum": volume_momentum,
                "is_volume_surge": volume_momentum > 1.5,
                "momentum_quality": self._assess_momentum_quality(daily_data)
            })

            # Intraday momentum if available
            if intraday_data is not None and not intraday_data.empty:
                intraday_momentum = self._calculate_intraday_momentum(intraday_data)
                metrics.update(intraday_momentum)

        except Exception as e:
            logger.error(f"Error calculating momentum metrics: {str(e)}")

        return metrics

    def _calculate_intraday_momentum(self, intraday_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate intraday momentum metrics"""
        try:
            if len(intraday_data) < 12:  # Need at least 1 hour of data
                return {}

            current_price = intraday_data['Close'].iloc[-1]

            # 1-hour momentum
            price_1h_ago = intraday_data['Close'].iloc[-12] if len(intraday_data) >= 12 else intraday_data['Close'].iloc[0]
            momentum_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100

            # 30-minute momentum
            price_30m_ago = intraday_data['Close'].iloc[-6] if len(intraday_data) >= 6 else intraday_data['Close'].iloc[0]
            momentum_30m = ((current_price - price_30m_ago) / price_30m_ago) * 100

            # Price acceleration
            momentum_recent = momentum_30m
            momentum_previous = ((price_30m_ago - price_1h_ago) / price_1h_ago) * 100 if len(intraday_data) >= 12 else 0
            acceleration = momentum_recent - momentum_previous

            return {
                "intraday_momentum_1h": momentum_1h,
                "intraday_momentum_30m": momentum_30m,
                "price_acceleration": acceleration,
                "is_accelerating": acceleration > 0.5
            }

        except:
            return {}

    def _assess_momentum_quality(self, data: pd.DataFrame) -> str:
        """Assess the quality of momentum"""
        try:
            recent_volume = data['Volume'].tail(3).mean()
            avg_volume = data['Volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

            price_momentum = ((data['Close'].iloc[-1] - data['Close'].iloc[-3]) / data['Close'].iloc[-3]) * 100

            if abs(price_momentum) > 3 and volume_ratio > 2:
                return "EXCELLENT"
            elif abs(price_momentum) > 2 and volume_ratio > 1.5:
                return "GOOD"
            elif abs(price_momentum) > 1 and volume_ratio > 1.2:
                return "FAIR"
            else:
                return "POOR"

        except:
            return "UNKNOWN"

    def _calculate_liquidity_metrics(self, daily_data: pd.DataFrame, intraday_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate liquidity metrics"""
        metrics = {}

        try:
            # Average dollar volume
            avg_volume = daily_data['Volume'].mean()
            avg_price = daily_data['Close'].mean()
            dollar_volume = avg_volume * avg_price

            # Spread estimation (using daily high-low)
            avg_spread = ((daily_data['High'] - daily_data['Low']) / daily_data['Close']).mean() * 100

            metrics.update({
                "average_dollar_volume": dollar_volume,
                "estimated_spread": avg_spread,
                "is_liquid": dollar_volume > 10000000 and avg_spread < 1.0,  # $10M+ volume, <1% spread
                "liquidity_score": self._calculate_liquidity_score(avg_volume, avg_spread, dollar_volume)
            })

        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {str(e)}")

        return metrics

    def _calculate_liquidity_score(self, volume: float, spread: float, dollar_volume: float) -> float:
        """Calculate liquidity score 0-100"""
        try:
            # Volume component (0-40 points)
            volume_score = min(40, (volume / 1000000) * 10)  # 4M volume = 40 points

            # Spread component (0-30 points)
            spread_score = max(0, 30 - (spread * 30))  # Lower spread = higher score

            # Dollar volume component (0-30 points)
            dollar_volume_score = min(30, (dollar_volume / 10000000) * 30)  # $10M = 30 points

            total_score = volume_score + spread_score + dollar_volume_score
            return min(100, total_score)

        except:
            return 50

    def _analyze_technical_setup(self, daily_data: pd.DataFrame, intraday_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze technical setup for day trading"""
        setup = {}

        try:
            # Moving averages
            ma_20 = daily_data['Close'].rolling(window=20).mean().iloc[-1] if len(daily_data) >= 20 else daily_data['Close'].mean()
            ma_10 = daily_data['Close'].rolling(window=10).mean().iloc[-1] if len(daily_data) >= 10 else daily_data['Close'].mean()
            current_price = daily_data['Close'].iloc[-1]

            # Trend analysis
            is_uptrend = current_price > ma_10 > ma_20
            is_downtrend = current_price < ma_10 < ma_20

            setup.update({
                "ma_10": ma_10,
                "ma_20": ma_20,
                "is_uptrend": is_uptrend,
                "is_downtrend": is_downtrend,
                "distance_from_ma20": ((current_price - ma_20) / ma_20) * 100
            })

            # RSI for momentum
            rsi = self._calculate_rsi(daily_data['Close'])
            setup["rsi"] = rsi

            # Bollinger Bands for volatility
            bb = self._calculate_bollinger_bands(daily_data['Close'])
            setup.update(bb)

            # Pattern recognition (simplified)
            setup["technical_pattern"] = self._identify_pattern(daily_data)

        except Exception as e:
            logger.error(f"Error analyzing technical setup: {str(e)}")

        return setup

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50
        except:
            return 50

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)

            current_price = prices.iloc[-1]
            bb_position = ((current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])) * 100

            return {
                "bb_upper": upper.iloc[-1],
                "bb_lower": lower.iloc[-1],
                "bb_position": bb_position,
                "near_bb_upper": bb_position > 80,
                "near_bb_lower": bb_position < 20
            }
        except:
            return {}

    def _identify_pattern(self, data: pd.DataFrame) -> str:
        """Identify basic chart patterns"""
        try:
            if len(data) < 5:
                return "INSUFFICIENT_DATA"

            recent_highs = data['High'].tail(5)
            recent_lows = data['Low'].tail(5)
            recent_closes = data['Close'].tail(5)

            # Simple pattern recognition
            if recent_closes.iloc[-1] > recent_closes.iloc[-2] > recent_closes.iloc[-3]:
                return "ASCENDING"
            elif recent_closes.iloc[-1] < recent_closes.iloc[-2] < recent_closes.iloc[-3]:
                return "DESCENDING"
            elif abs(recent_closes.pct_change().std()) < 0.02:
                return "CONSOLIDATION"
            else:
                return "SIDEWAYS"

        except:
            return "UNKNOWN"

    def _analyze_gap_potential(self, daily_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze potential for gaps and gap trading"""
        gap_analysis = {}

        try:
            if len(daily_data) >= 2:
                yesterday_close = daily_data['Close'].iloc[-2]
                today_open = daily_data['Open'].iloc[-1]
                current_price = daily_data['Close'].iloc[-1]

                gap_percent = ((today_open - yesterday_close) / yesterday_close) * 100

                gap_analysis.update({
                    "has_gap": abs(gap_percent) > 0.5,
                    "gap_percent": gap_percent,
                    "gap_type": "UP" if gap_percent > 0.5 else "DOWN" if gap_percent < -0.5 else "NONE",
                    "gap_filled": self._check_gap_fill(yesterday_close, today_open, current_price),
                    "gap_size_category": self._categorize_gap_size(gap_percent)
                })

        except Exception as e:
            logger.error(f"Error analyzing gap potential: {str(e)}")

        return gap_analysis

    def _check_gap_fill(self, yesterday_close: float, today_open: float, current_price: float) -> bool:
        """Check if gap has been filled"""
        if today_open > yesterday_close:  # Gap up
            return current_price <= yesterday_close
        elif today_open < yesterday_close:  # Gap down
            return current_price >= yesterday_close
        return False

    def _categorize_gap_size(self, gap_percent: float) -> str:
        """Categorize gap size"""
        abs_gap = abs(gap_percent)
        if abs_gap > 10:
            return "HUGE"
        elif abs_gap > 5:
            return "LARGE"
        elif abs_gap > 2:
            return "MEDIUM"
        elif abs_gap > 0.5:
            return "SMALL"
        else:
            return "NONE"

    def _assess_profit_potential(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess 10% profit potential"""
        potential = {}

        try:
            current_price = analysis.get("current_price", 0)
            volatility = analysis.get("annualized_volatility", 20)
            momentum_quality = analysis.get("momentum_quality", "POOR")

            # Calculate probability of 10% move
            daily_vol = volatility / np.sqrt(252)  # Daily volatility
            prob_10_percent = min(0.9, max(0.1, (daily_vol / 10) * 100))

            # Adjust for momentum
            momentum_multiplier = {
                "EXCELLENT": 1.5,
                "GOOD": 1.2,
                "FAIR": 1.0,
                "POOR": 0.7
            }.get(momentum_quality, 1.0)

            adjusted_prob = min(0.9, prob_10_percent * momentum_multiplier)

            # Time to target estimation
            if volatility > 50:
                time_estimate = "1-3 hours"
            elif volatility > 30:
                time_estimate = "3-6 hours"
            else:
                time_estimate = "Full day"

            potential.update({
                "target_price": current_price * 1.10,
                "probability_10_percent": adjusted_prob,
                "estimated_time_to_target": time_estimate,
                "is_high_potential": adjusted_prob > 0.6 and volatility > 25,
                "risk_reward_favorable": True  # Assuming 2% stop loss vs 10% target
            })

        except Exception as e:
            logger.error(f"Error assessing profit potential: {str(e)}")

        return potential

    def _calculate_day_trading_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall day trading score"""
        score = 0

        try:
            # Volatility score (25 points)
            volatility = analysis.get("annualized_volatility", 0)
            if 25 <= volatility <= 60:
                score += 25
            elif 20 <= volatility <= 80:
                score += 20
            elif volatility > 15:
                score += 10

            # Liquidity score (20 points)
            liquidity_score = analysis.get("liquidity_score", 50)
            score += (liquidity_score / 100) * 20

            # Momentum score (20 points)
            momentum_quality = analysis.get("momentum_quality", "POOR")
            momentum_scores = {"EXCELLENT": 20, "GOOD": 15, "FAIR": 10, "POOR": 5}
            score += momentum_scores.get(momentum_quality, 5)

            # Volume score (15 points)
            volume_momentum = analysis.get("volume_momentum", 1)
            if volume_momentum > 2:
                score += 15
            elif volume_momentum > 1.5:
                score += 12
            elif volume_momentum > 1.2:
                score += 8
            else:
                score += 3

            # Technical setup score (10 points)
            if analysis.get("is_uptrend") and analysis.get("rsi", 50) < 70:
                score += 10
            elif analysis.get("technical_pattern") in ["ASCENDING", "CONSOLIDATION"]:
                score += 8
            else:
                score += 3

            # Profit potential score (10 points)
            if analysis.get("is_high_potential"):
                score += 10
            elif analysis.get("probability_10_percent", 0) > 0.4:
                score += 7
            else:
                score += 3

        except Exception as e:
            logger.error(f"Error calculating day trading score: {str(e)}")

        return min(100, score)

    def _determine_suitability(self, analysis: Dict[str, Any]) -> bool:
        """Determine if stock is suitable for day trading"""
        try:
            score = analysis.get("day_trading_score", 0)
            is_liquid = analysis.get("is_liquid", False)
            volatility = analysis.get("annualized_volatility", 0)
            risk_level = analysis.get("risk_level", "HIGH")

            # Minimum criteria
            min_criteria = [
                score >= 50,
                is_liquid,
                15 <= volatility <= 100,
                risk_level != "VERY_HIGH"
            ]

            return all(min_criteria)

        except:
            return False

    def _assess_risk_level(self, analysis: Dict[str, Any]) -> str:
        """Assess risk level for day trading"""
        try:
            volatility = analysis.get("annualized_volatility", 20)
            liquidity_score = analysis.get("liquidity_score", 50)
            gap_size = analysis.get("gap_percent", 0)

            risk_factors = 0

            if volatility > 60:
                risk_factors += 2
            elif volatility > 40:
                risk_factors += 1

            if liquidity_score < 50:
                risk_factors += 2
            elif liquidity_score < 70:
                risk_factors += 1

            if abs(gap_size) > 5:
                risk_factors += 2
            elif abs(gap_size) > 2:
                risk_factors += 1

            if risk_factors >= 5:
                return "VERY_HIGH"
            elif risk_factors >= 3:
                return "HIGH"
            elif risk_factors >= 1:
                return "MEDIUM"
            else:
                return "LOW"

        except:
            return "MEDIUM"

    def _analyze_time_factors(self) -> Dict[str, Any]:
        """Analyze time-sensitive factors"""
        now = datetime.now()
        market_hour = now.hour

        # Assuming EST market hours (9:30 AM - 4:00 PM)
        is_market_open = 9 <= market_hour <= 16
        is_power_hour = market_hour >= 15  # Last hour
        is_opening_hour = market_hour <= 10  # First hour

        return {
            "current_hour": market_hour,
            "is_market_open": is_market_open,
            "is_power_hour": is_power_hour,
            "is_opening_hour": is_opening_hour,
            "optimal_time": is_market_open and not is_power_hour,
            "time_risk": "HIGH" if is_power_hour else "LOW"
        }

def get_day_trading_universes() -> Dict[str, List[str]]:
    """Get predefined universes optimized for day trading"""
    return {
        "HIGH_VOLUME_MOVERS": [
            "TSLA", "AAPL", "NVDA", "AMD", "SPY", "QQQ", "AMZN", "MSFT",
            "META", "GOOGL", "NFLX", "BABA", "UBER", "COIN", "ROKU"
        ],

        "VOLATILE_TECH": [
            "TSLA", "NVDA", "AMD", "NFLX", "ROKU", "ZOOM", "PELOTON",
            "SNOW", "PLTR", "SQ", "PYPL", "CRM", "SHOP", "TWLO"
        ],

        "MEME_STOCKS": [
            "GME", "AMC", "BB", "NOK", "SNDL", "CLOV", "WISH",
            "SPCE", "PLTR", "DKNG", "RKT", "SKLZ", "SOFI"
        ],

        "BIOTECH_MOVERS": [
            "MRNA", "BNTX", "GILD", "BIIB", "REGN", "VRTX", "ILMN",
            "AMGN", "CELG", "MYL", "TEVA", "JNJ", "PFE", "ABBV"
        ],

        "ETF_MOVERS": [
            "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLV",
            "GDX", "EWZ", "FXI", "EEM", "TLT", "HYG", "VIX"
        ],

        "PENNY_STOCKS_SAFE": [
            # Higher priced "penny" stocks with good volume
            # This would need to be populated with current candidates
        ]
    }