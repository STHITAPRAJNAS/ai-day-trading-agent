from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import logging
from .base_agent import CoordinatorAgent
from .data_collector import StockDataCollector, NewsCollector, MarketDataCollector
from .day_trading_analyst import DayTradingAnalyst
from ..utils.day_trading_screener import DayTradingScreener, get_day_trading_universes
from ..models.stock_data import StockAnalysis, DailyStockPicks, TechnicalSignal, MarketSentiment, TradingStrategy

logger = logging.getLogger(__name__)

class DayTradingCoordinator(CoordinatorAgent):
    """Specialized coordinator for day trading with 10% profit targets"""

    def __init__(self):
        # Initialize specialized agents
        self.stock_collector = StockDataCollector()
        self.news_collector = NewsCollector()
        self.market_collector = MarketDataCollector()
        self.day_trading_analyst = DayTradingAnalyst()
        self.screener = DayTradingScreener()

        super().__init__(
            name="DayTradingCoordinator",
            description="Coordinates day trading analysis for 10% profit opportunities",
            child_agents=[
                self.stock_collector,
                self.news_collector,
                self.market_collector,
                self.day_trading_analyst
            ]
        )

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate day trading picks with 10% profit potential"""
        universe = data.get("universe", "HIGH_VOLUME_MOVERS")
        max_picks = data.get("max_picks", 10)
        time_preference = data.get("time_preference", "INTRADAY")  # INTRADAY, SCALP, SWING

        self.log_analysis_start()

        try:
            # Step 1: Get appropriate stock universe for day trading
            symbols = self._get_day_trading_universe(universe)
            logger.info(f"Analyzing {len(symbols)} day trading candidates from {universe}")

            # Step 2: Screen for day trading suitability
            suitable_candidates = self.screener.screen_for_day_trading(symbols, max_picks * 3)
            logger.info(f"Found {len(suitable_candidates)} suitable day trading candidates")

            if not suitable_candidates:
                return {
                    "error": "No suitable day trading candidates found",
                    "recommendation": "Market conditions may not be favorable for day trading"
                }

            # Step 3: Collect market data for context
            market_data = await self.market_collector.analyze({})

            # Step 4: Detailed analysis of top candidates
            detailed_analyses = []
            for candidate in suitable_candidates[:max_picks * 2]:  # Analyze more than needed
                symbol = candidate["symbol"]
                try:
                    # Get current stock data
                    stock_data = await self.stock_collector.analyze({"symbols": [symbol]})
                    symbol_data = stock_data.get(symbol, {})

                    if "error" in symbol_data:
                        continue

                    # Get news data
                    news_data = await self.news_collector.analyze({"symbols": [symbol], "days_back": 1})
                    symbol_news = news_data.get(symbol, [])

                    # Perform day trading analysis
                    day_trading_analysis = await self.day_trading_analyst.analyze({"symbol": symbol})

                    if "error" not in day_trading_analysis:
                        # Combine all analysis
                        combined_analysis = await self._create_day_trading_pick(
                            symbol,
                            symbol_data,
                            symbol_news,
                            day_trading_analysis,
                            candidate,
                            market_data
                        )

                        if combined_analysis:
                            detailed_analyses.append(combined_analysis)

                except Exception as e:
                    logger.error(f"Analysis failed for {symbol}: {str(e)}")

            # Step 5: Rank and select final picks
            final_picks = self._select_top_day_trading_picks(detailed_analyses, max_picks, time_preference)

            # Step 6: Generate day trading report
            day_trading_report = self._generate_day_trading_report(final_picks, market_data, time_preference)

            result = {
                "day_trading_picks": day_trading_report,
                "total_analyzed": len(symbols),
                "candidates_screened": len(suitable_candidates),
                "final_picks_count": len(final_picks),
                "market_conditions": self._assess_market_conditions(market_data),
                "trading_session_info": self._get_trading_session_info(),
                "analysis_summary": self._generate_analysis_summary(detailed_analyses, final_picks)
            }

            self.log_analysis_complete(result_summary=f"Generated {len(final_picks)} day trading picks")
            return result

        except Exception as e:
            logger.error(f"Day trading analysis failed: {str(e)}")
            return {"error": str(e)}

    def _get_day_trading_universe(self, universe_name: str) -> List[str]:
        """Get appropriate stock universe for day trading"""
        day_trading_universes = get_day_trading_universes()

        if universe_name in day_trading_universes:
            return day_trading_universes[universe_name]

        # Default to high volume movers
        return day_trading_universes["HIGH_VOLUME_MOVERS"]

    async def _create_day_trading_pick(
        self,
        symbol: str,
        stock_data: Dict[str, Any],
        news_data: List[Dict[str, Any]],
        day_trading_data: Dict[str, Any],
        screener_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create comprehensive day trading pick"""

        try:
            current_price = stock_data.get("current_price") or day_trading_data.get("current_price")
            if not current_price:
                return None

            # Calculate combined scores
            day_trading_score = day_trading_data.get("day_trading_score", 0)
            screener_score = screener_data.get("day_trading_score", 0)
            combined_score = (day_trading_score * 0.7) + (screener_score * 0.3)

            # Generate optimized entry/exit strategies
            entry_strategy = self._generate_day_trading_entry_strategy(
                symbol, current_price, day_trading_data, screener_data
            )

            exit_strategy = self._generate_day_trading_exit_strategy(
                symbol, current_price, day_trading_data, entry_strategy
            )

            # Risk assessment
            risk_assessment = self._assess_day_trading_risk(day_trading_data, screener_data, market_data)

            # Time-sensitive factors
            time_factors = day_trading_data.get("time_factors", {})

            # News impact
            news_impact = self._assess_news_impact(news_data)

            # Create the pick
            pick = {
                "symbol": symbol,
                "analysis_timestamp": datetime.now(),
                "current_price": current_price,

                # Scoring
                "day_trading_score": combined_score,
                "screener_score": screener_score,
                "technical_score": day_trading_score,

                # Market signals
                "trading_signal": self._determine_trading_signal(day_trading_data, combined_score),
                "urgency": self._determine_urgency(day_trading_data, time_factors),

                # Strategies
                "entry_strategy": entry_strategy,
                "exit_strategy": exit_strategy,

                # Risk & Targets
                "risk_level": risk_assessment["overall_risk"],
                "profit_target": current_price * 1.10,  # 10% target
                "time_horizon": "INTRADAY",

                # Analysis details
                "volatility_analysis": day_trading_data.get("volatility_analysis", {}),
                "momentum_analysis": day_trading_data.get("momentum_indicators", {}),
                "liquidity_analysis": day_trading_data.get("liquidity_analysis", {}),
                "gap_analysis": day_trading_data.get("gap_analysis", {}),

                # Market context
                "news_impact": news_impact,
                "market_conditions": market_data.get("economic_indicators", {}),

                # Key insights
                "key_insights": self._generate_day_trading_insights(symbol, day_trading_data, screener_data),
                "risks": self._identify_day_trading_risks(day_trading_data, risk_assessment),
                "catalysts": self._identify_day_trading_catalysts(day_trading_data, news_data),

                # Metadata
                "screener_data": screener_data,
                "day_trading_data": day_trading_data
            }

            return pick

        except Exception as e:
            logger.error(f"Failed to create day trading pick for {symbol}: {str(e)}")
            return None

    def _generate_day_trading_entry_strategy(
        self,
        symbol: str,
        current_price: float,
        day_trading_data: Dict[str, Any],
        screener_data: Dict[str, Any]
    ) -> TradingStrategy:
        """Generate optimized entry strategy for day trading"""

        # Get support/resistance levels
        support_resistance = day_trading_data.get("support_resistance", {})
        vwap = support_resistance.get("vwap", current_price)

        # Get entry signals
        entry_signals = day_trading_data.get("entry_signals", [])

        # Determine optimal entry
        if entry_signals:
            # Use signal-based entry
            best_signal = max(entry_signals, key=lambda x: x.get("strength", "WEAK") == "STRONG")
            entry_price = best_signal.get("entry_price", current_price * 0.999)
            reasoning = f"Entry based on {best_signal.get('type', 'signal')} with {best_signal.get('strength', 'medium')} strength"
        else:
            # Use VWAP-based entry
            if current_price > vwap:
                entry_price = min(current_price * 0.999, vwap * 1.001)  # Slight pullback to VWAP
                reasoning = "Entry on pullback to VWAP support"
            else:
                entry_price = current_price * 1.001  # Small breakout above current
                reasoning = "Entry on breakout above resistance"

        # Calculate stop loss (tight for day trading)
        atr = day_trading_data.get("indicators_5m", {}).get("atr", current_price * 0.02)
        stop_loss = entry_price - (atr * 1.5)  # 1.5x ATR stop

        # 10% profit target
        take_profit = entry_price * 1.10

        # Risk-reward calculation
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        risk_reward_ratio = reward / risk if risk > 0 else 0

        # Confidence based on analysis quality
        volatility_analysis = day_trading_data.get("volatility_analysis", {})
        momentum_analysis = day_trading_data.get("momentum_indicators", {})

        confidence_factors = [
            volatility_analysis.get("is_high_volatility", False),
            momentum_analysis.get("momentum_quality") == "HIGH_QUALITY",
            len(entry_signals) > 0,
            risk_reward_ratio >= 2.0
        ]

        confidence = (sum(confidence_factors) / len(confidence_factors)) * 100

        return TradingStrategy(
            strategy_type="day_trading_entry",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            confidence_score=confidence,
            timeframe="Intraday (1-6 hours)",
            reasoning=reasoning
        )

    def _generate_day_trading_exit_strategy(
        self,
        symbol: str,
        current_price: float,
        day_trading_data: Dict[str, Any],
        entry_strategy: TradingStrategy
    ) -> TradingStrategy:
        """Generate optimized exit strategy for day trading"""

        # Get resistance levels
        support_resistance = day_trading_data.get("support_resistance", {})
        resistance_levels = support_resistance.get("key_resistance", [current_price * 1.05])

        # Primary exit at 10% profit target
        primary_exit = entry_strategy.take_profit

        # Secondary exits for partial profit taking
        partial_exit_1 = entry_strategy.entry_price * 1.05  # 5% partial
        partial_exit_2 = entry_strategy.entry_price * 1.08  # 8% partial

        # Time-based exit (before market close)
        time_factors = day_trading_data.get("time_factors", {})
        time_exit_reason = "Exit before 3:30 PM to avoid end-of-day volatility"

        reasoning = f"Primary exit at 10% target (${primary_exit:.2f}). " \
                   f"Consider partial exits at 5% (${partial_exit_1:.2f}) and 8% (${partial_exit_2:.2f}). " \
                   f"{time_exit_reason}"

        return TradingStrategy(
            strategy_type="day_trading_exit",
            entry_price=entry_strategy.entry_price,
            exit_price=primary_exit,
            stop_loss=entry_strategy.stop_loss,
            take_profit=primary_exit,
            risk_reward_ratio=entry_strategy.risk_reward_ratio,
            confidence_score=entry_strategy.confidence_score * 0.95,
            timeframe="Intraday exit",
            reasoning=reasoning
        )

    def _assess_day_trading_risk(
        self,
        day_trading_data: Dict[str, Any],
        screener_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive day trading risk assessment"""

        risk_factors = []

        # Volatility risk
        volatility_analysis = day_trading_data.get("volatility_analysis", {})
        if volatility_analysis.get("annualized_volatility", 0) > 60:
            risk_factors.append("High volatility (>60%)")

        # Liquidity risk
        liquidity_analysis = day_trading_data.get("liquidity_analysis", {})
        if not liquidity_analysis.get("is_liquid", True):
            risk_factors.append("Poor liquidity")

        # Time risk
        current_hour = datetime.now().hour
        if current_hour >= 15:  # After 3 PM EST
            risk_factors.append("Late trading session")

        # Gap risk
        gap_analysis = day_trading_data.get("gap_analysis", {})
        if gap_analysis.get("gap_size_category") in ["LARGE", "HUGE"]:
            risk_factors.append("Significant price gap")

        # Market risk
        market_trend = market_data.get("economic_indicators", {}).get("market_trend", "Sideways")
        if "Down" in market_trend:
            risk_factors.append("Bearish market conditions")

        # Overall risk assessment
        if len(risk_factors) >= 3:
            overall_risk = "VERY_HIGH"
        elif len(risk_factors) >= 2:
            overall_risk = "HIGH"
        elif len(risk_factors) >= 1:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"

        return {
            "overall_risk": overall_risk,
            "risk_factors": risk_factors,
            "risk_count": len(risk_factors)
        }

    def _determine_trading_signal(self, day_trading_data: Dict[str, Any], score: float) -> str:
        """Determine trading signal based on analysis"""
        entry_signals = day_trading_data.get("entry_signals", [])
        trading_recommendation = day_trading_data.get("trading_recommendation", "HOLD")

        if score >= 80 and entry_signals:
            return "STRONG_BUY"
        elif score >= 65 and (entry_signals or trading_recommendation in ["BUY", "STRONG_BUY"]):
            return "BUY"
        elif score >= 50:
            return "WEAK_BUY"
        else:
            return "HOLD"

    def _determine_urgency(self, day_trading_data: Dict[str, Any], time_factors: Dict[str, Any]) -> str:
        """Determine urgency level for the trade"""
        current_hour = datetime.now().hour
        entry_signals = day_trading_data.get("entry_signals", [])
        momentum_quality = day_trading_data.get("momentum_indicators", {}).get("momentum_quality", "LOW_QUALITY")

        if current_hour >= 15:  # After 3 PM
            return "LOW"  # Getting late for day trading
        elif current_hour <= 10 and entry_signals and momentum_quality == "HIGH_QUALITY":
            return "HIGH"  # Good early opportunity
        elif entry_signals:
            return "MEDIUM"
        else:
            return "LOW"

    def _assess_news_impact(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess news impact on day trading"""
        if not news_data:
            return {"impact": "NONE", "sentiment": "NEUTRAL"}

        # Simple sentiment analysis (would be enhanced with NLP in production)
        recent_news = [item for item in news_data if item.get("published_at")]
        if not recent_news:
            return {"impact": "NONE", "sentiment": "NEUTRAL"}

        # Count positive/negative keywords
        positive_words = ["surge", "gain", "profit", "beat", "strong", "growth", "upgrade"]
        negative_words = ["drop", "fall", "loss", "miss", "weak", "decline", "downgrade"]

        total_positive = 0
        total_negative = 0

        for news_item in recent_news:
            title = news_item.get("title", "").lower()
            content = news_item.get("content", "").lower()
            text = title + " " + content

            total_positive += sum(1 for word in positive_words if word in text)
            total_negative += sum(1 for word in negative_words if word in text)

        if total_positive > total_negative * 1.5:
            return {"impact": "POSITIVE", "sentiment": "BULLISH", "news_count": len(recent_news)}
        elif total_negative > total_positive * 1.5:
            return {"impact": "NEGATIVE", "sentiment": "BEARISH", "news_count": len(recent_news)}
        else:
            return {"impact": "NEUTRAL", "sentiment": "NEUTRAL", "news_count": len(recent_news)}

    def _generate_day_trading_insights(
        self,
        symbol: str,
        day_trading_data: Dict[str, Any],
        screener_data: Dict[str, Any]
    ) -> List[str]:
        """Generate key insights for day trading"""
        insights = []

        # Volatility insights
        volatility_analysis = day_trading_data.get("volatility_analysis", {})
        if volatility_analysis.get("is_high_volatility"):
            insights.append("High volatility environment provides 10% move opportunity")

        # Volume insights
        momentum_indicators = day_trading_data.get("momentum_indicators", {})
        if momentum_indicators.get("momentum_quality") == "HIGH_QUALITY":
            insights.append("Strong volume-confirmed momentum breakout")

        # Gap insights
        gap_analysis = day_trading_data.get("gap_analysis", {})
        if gap_analysis.get("has_gap") and not gap_analysis.get("gap_filled"):
            gap_type = gap_analysis.get("gap_type", "")
            insights.append(f"Unfilled {gap_type.lower()} gap provides directional bias")

        # VWAP insights
        support_resistance = day_trading_data.get("support_resistance", {})
        vwap = support_resistance.get("vwap")
        current_price = day_trading_data.get("current_price", 0)
        if vwap and current_price:
            if current_price > vwap * 1.01:
                insights.append("Trading above VWAP with institutional support")
            elif current_price < vwap * 0.99:
                insights.append("Below VWAP - potential bounce opportunity")

        return insights[:3]  # Return top 3 insights

    def _identify_day_trading_risks(
        self,
        day_trading_data: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Identify specific day trading risks"""
        return risk_assessment.get("risk_factors", [])

    def _identify_day_trading_catalysts(
        self,
        day_trading_data: Dict[str, Any],
        news_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify potential catalysts for day trading moves"""
        catalysts = []

        # Technical catalysts
        entry_signals = day_trading_data.get("entry_signals", [])
        for signal in entry_signals:
            if signal.get("strength") == "STRONG":
                catalysts.append(f"{signal.get('type', 'Technical')} breakout signal")

        # Volume catalysts
        momentum_indicators = day_trading_data.get("momentum_indicators", {})
        if momentum_indicators.get("volume_momentum", 1) > 2:
            catalysts.append("Exceptional volume surge")

        # News catalysts
        if news_data:
            catalysts.append("Recent news catalyst")

        return catalysts[:3]

    def _select_top_day_trading_picks(
        self,
        analyses: List[Dict[str, Any]],
        max_picks: int,
        time_preference: str
    ) -> List[Dict[str, Any]]:
        """Select and rank top day trading picks"""

        # Filter by time preference and quality
        filtered_picks = []
        for analysis in analyses:
            score = analysis.get("day_trading_score", 0)
            risk_level = analysis.get("risk_level", "HIGH")

            # Quality filters
            if score >= 60 and risk_level != "VERY_HIGH":
                # Adjust score based on time preference
                if time_preference == "SCALP":
                    # Prefer higher volatility for scalping
                    volatility = analysis.get("volatility_analysis", {}).get("annualized_volatility", 0)
                    if volatility > 40:
                        analysis["adjusted_score"] = score * 1.1
                    else:
                        analysis["adjusted_score"] = score * 0.9
                elif time_preference == "SWING":
                    # Prefer momentum quality for swing trades
                    momentum_quality = analysis.get("momentum_analysis", {}).get("momentum_quality", "POOR")
                    if momentum_quality in ["EXCELLENT", "GOOD"]:
                        analysis["adjusted_score"] = score * 1.1
                    else:
                        analysis["adjusted_score"] = score * 0.9
                else:  # INTRADAY
                    analysis["adjusted_score"] = score

                filtered_picks.append(analysis)

        # Sort by adjusted score
        filtered_picks.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)

        return filtered_picks[:max_picks]

    def _generate_day_trading_report(
        self,
        picks: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        time_preference: str
    ) -> Dict[str, Any]:
        """Generate comprehensive day trading report"""

        if not picks:
            return {
                "date": datetime.now(),
                "picks": [],
                "market_overview": "No suitable day trading opportunities found",
                "key_themes": ["Poor market conditions for day trading"],
                "risk_factors": ["Low volatility", "Poor liquidity", "Unfavorable market conditions"],
                "generated_at": datetime.now(),
                "trading_session": self._get_trading_session_info()
            }

        # Generate market overview
        market_overview = self._generate_day_trading_market_overview(market_data)

        # Extract key themes
        key_themes = self._extract_day_trading_themes(picks)

        # Risk factors
        risk_factors = self._extract_common_risk_factors(picks)

        return {
            "date": datetime.now(),
            "picks": picks,
            "market_overview": market_overview,
            "key_themes": key_themes,
            "risk_factors": risk_factors,
            "time_preference": time_preference,
            "trading_session": self._get_trading_session_info(),
            "generated_at": datetime.now()
        }

    def _assess_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall market conditions for day trading"""
        vix_data = market_data.get("VIX", {})
        spy_data = market_data.get("SPY", {})

        vix_level = vix_data.get("current_price", 20)
        spy_change = spy_data.get("change_percent", 0)

        if vix_level > 25 and abs(spy_change) > 1:
            condition = "EXCELLENT"  # High volatility, good for day trading
        elif vix_level > 20 or abs(spy_change) > 0.5:
            condition = "GOOD"
        elif vix_level < 15 and abs(spy_change) < 0.3:
            condition = "POOR"  # Low volatility, poor for day trading
        else:
            condition = "FAIR"

        return {
            "overall_condition": condition,
            "vix_level": vix_level,
            "market_movement": spy_change,
            "volatility_environment": "HIGH" if vix_level > 25 else "MEDIUM" if vix_level > 15 else "LOW"
        }

    def _get_trading_session_info(self) -> Dict[str, Any]:
        """Get current trading session information"""
        now = datetime.now()
        hour = now.hour

        # Assuming EST
        if 9 <= hour < 10:
            session = "OPENING"
        elif 10 <= hour < 14:
            session = "MIDDAY"
        elif 14 <= hour < 15:
            session = "AFTERNOON"
        elif 15 <= hour < 16:
            session = "POWER_HOUR"
        else:
            session = "AFTER_HOURS"

        return {
            "current_time": now,
            "session": session,
            "is_market_open": 9 <= hour < 16,
            "optimal_for_day_trading": 9 <= hour < 15
        }

    def _generate_day_trading_market_overview(self, market_data: Dict[str, Any]) -> str:
        """Generate market overview for day trading"""
        spy_data = market_data.get("SPY", {})
        vix_data = market_data.get("VIX", {})

        spy_change = spy_data.get("change_percent", 0)
        vix_level = vix_data.get("current_price", 20)

        trend = "bullish" if spy_change > 0.5 else "bearish" if spy_change < -0.5 else "neutral"
        volatility = "high" if vix_level > 25 else "moderate" if vix_level > 15 else "low"

        return f"Market showing {trend} bias with {volatility} volatility (VIX: {vix_level:.1f}). " \
               f"S&P 500 {spy_change:+.2f}% today. " \
               f"{'Excellent' if vix_level > 25 else 'Good' if vix_level > 20 else 'Challenging'} " \
               f"conditions for day trading."

    def _extract_day_trading_themes(self, picks: List[Dict[str, Any]]) -> List[str]:
        """Extract key themes from day trading picks"""
        themes = []

        # Analyze common patterns
        high_vol_count = sum(1 for pick in picks
                           if pick.get("volatility_analysis", {}).get("is_high_volatility", False))

        gap_count = sum(1 for pick in picks
                       if pick.get("gap_analysis", {}).get("has_gap", False))

        momentum_count = sum(1 for pick in picks
                           if pick.get("momentum_analysis", {}).get("momentum_quality") in ["EXCELLENT", "GOOD"])

        if high_vol_count >= len(picks) * 0.6:
            themes.append("High volatility environment driving opportunities")

        if gap_count >= len(picks) * 0.4:
            themes.append("Gap trading opportunities prevalent")

        if momentum_count >= len(picks) * 0.5:
            themes.append("Strong momentum breakouts dominating")

        if not themes:
            themes.append("Mixed technical setups across various sectors")

        return themes

    def _extract_common_risk_factors(self, picks: List[Dict[str, Any]]) -> List[str]:
        """Extract common risk factors across picks"""
        all_risks = []
        for pick in picks:
            all_risks.extend(pick.get("risks", []))

        # Count occurrences
        from collections import Counter
        risk_counts = Counter(all_risks)

        # Return most common risks
        common_risks = [risk for risk, count in risk_counts.most_common(3)]

        if not common_risks:
            common_risks = ["Standard day trading risks apply"]

        return common_risks

    def _generate_analysis_summary(self, all_analyses: List[Dict], final_picks: List[Dict]) -> Dict[str, Any]:
        """Generate summary of day trading analysis"""
        if not all_analyses:
            return {}

        scores = [analysis.get("day_trading_score", 0) for analysis in all_analyses]

        return {
            "total_analyzed": len(all_analyses),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "top_score": max(scores) if scores else 0,
            "picks_above_80": len([s for s in scores if s >= 80]),
            "picks_above_70": len([s for s in scores if s >= 70]),
            "picks_above_60": len([s for s in scores if s >= 60]),
            "final_picks_avg_score": sum(pick.get("day_trading_score", 0) for pick in final_picks) / len(final_picks) if final_picks else 0
        }