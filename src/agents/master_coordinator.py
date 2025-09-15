from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import logging
from .base_agent import CoordinatorAgent
from .data_collector import StockDataCollector, NewsCollector, MarketDataCollector
from .technical_analyst import TechnicalAnalyst
from ..models.stock_data import StockAnalysis, DailyStockPicks, TechnicalSignal, MarketSentiment, TradingStrategy
from ..utils.stock_universe import get_stock_universe

logger = logging.getLogger(__name__)

class MasterCoordinator(CoordinatorAgent):
    """Master coordinator that orchestrates all analysis agents to generate daily stock picks"""

    def __init__(self):
        # Initialize all child agents
        self.stock_collector = StockDataCollector()
        self.news_collector = NewsCollector()
        self.market_collector = MarketDataCollector()
        self.technical_analyst = TechnicalAnalyst()

        super().__init__(
            name="MasterCoordinator",
            description="Orchestrates comprehensive stock analysis to generate top 10 daily picks",
            child_agents=[
                self.stock_collector,
                self.news_collector,
                self.market_collector,
                self.technical_analyst
            ]
        )

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis method that generates daily stock picks"""
        universe = data.get("universe", "SP500")
        max_picks = data.get("max_picks", 10)

        self.log_analysis_start()

        try:
            # Get stock universe
            symbols = get_stock_universe(universe)
            logger.info(f"Analyzing {len(symbols)} stocks from {universe}")

            # Step 1: Collect market overview
            market_data = await self.market_collector.analyze({})

            # Step 2: Collect data for all symbols (in batches to avoid API limits)
            batch_size = 50
            all_stock_data = {}
            all_news_data = {}

            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_symbols)} symbols")

                # Collect stock data and news in parallel
                stock_task = self.stock_collector.analyze({"symbols": batch_symbols})
                news_task = self.news_collector.analyze({"symbols": batch_symbols})

                batch_stock_data, batch_news_data = await asyncio.gather(stock_task, news_task)

                all_stock_data.update(batch_stock_data)
                all_news_data.update(batch_news_data)

            # Step 3: Perform technical analysis for each stock
            technical_analyses = {}
            for symbol in symbols:
                stock_data = all_stock_data.get(symbol, {})
                if "prices" in stock_data and stock_data["prices"]:
                    try:
                        technical_analysis = await self.technical_analyst.analyze({
                            "symbol": symbol,
                            "prices": stock_data["prices"]
                        })
                        technical_analyses[symbol] = technical_analysis
                    except Exception as e:
                        logger.error(f"Technical analysis failed for {symbol}: {str(e)}")

            # Step 4: Generate comprehensive analysis for each stock
            stock_analyses = []
            for symbol in symbols:
                try:
                    analysis = await self._generate_stock_analysis(
                        symbol,
                        all_stock_data.get(symbol, {}),
                        all_news_data.get(symbol, []),
                        technical_analyses.get(symbol, {}),
                        market_data
                    )
                    if analysis:
                        stock_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Analysis generation failed for {symbol}: {str(e)}")

            # Step 5: Rank and select top picks
            top_picks = self._select_top_picks(stock_analyses, max_picks)

            # Step 6: Generate market overview and insights
            daily_picks = DailyStockPicks(
                date=datetime.now(),
                picks=top_picks,
                market_overview=self._generate_market_overview(market_data),
                key_themes=self._extract_key_themes(all_news_data, top_picks),
                risk_factors=self._identify_risk_factors(market_data, top_picks),
                generated_at=datetime.now()
            )

            result = {
                "daily_picks": daily_picks.dict(),
                "total_analyzed": len(symbols),
                "top_picks_count": len(top_picks),
                "market_sentiment": self._assess_market_sentiment(market_data),
                "analysis_summary": self._generate_analysis_summary(stock_analyses, top_picks)
            }

            self.log_analysis_complete(result_summary=f"Generated {len(top_picks)} picks from {len(symbols)} analyzed")
            return result

        except Exception as e:
            logger.error(f"Master analysis failed: {str(e)}")
            return {"error": str(e)}

    async def _generate_stock_analysis(
        self,
        symbol: str,
        stock_data: Dict[str, Any],
        news_data: List[Dict[str, Any]],
        technical_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Optional[StockAnalysis]:
        """Generate comprehensive analysis for a single stock"""

        if "error" in stock_data or not stock_data.get("prices"):
            return None

        try:
            current_price = stock_data.get("current_price")
            if not current_price:
                return None

            # Technical analysis scores
            technical_score = technical_data.get("technical_score", 50)
            technical_signal = technical_data.get("signal", {}).get("overall_signal", TechnicalSignal.HOLD)

            # Fundamental analysis (simplified)
            fundamental_score = self._calculate_fundamental_score(stock_data.get("fundamental", {}))

            # Sentiment analysis from news
            sentiment_score = self._calculate_sentiment_score(news_data)

            # Overall score calculation
            overall_score = self._calculate_overall_score(technical_score, fundamental_score, sentiment_score)

            # Generate trading strategies
            entry_strategy = self._generate_entry_strategy(symbol, current_price, technical_data, fundamental_score)
            exit_strategy = self._generate_exit_strategy(symbol, current_price, technical_data, entry_strategy)

            # Price targets
            price_targets = self._calculate_price_targets(current_price, technical_data, fundamental_score)

            # Generate insights
            insights = self._generate_key_insights(symbol, stock_data, technical_data, news_data)
            risks = self._identify_stock_risks(symbol, stock_data, technical_data, market_data)
            catalysts = self._identify_catalysts(symbol, news_data, stock_data)

            analysis = StockAnalysis(
                symbol=symbol,
                analysis_timestamp=datetime.now(),
                current_price=current_price,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                sentiment_score=sentiment_score,
                overall_score=overall_score,
                technical_signal=technical_signal,
                market_sentiment=self._determine_stock_sentiment(sentiment_score),
                entry_strategy=entry_strategy,
                exit_strategy=exit_strategy,
                risk_level=self._assess_risk_level(technical_data, fundamental_score, sentiment_score),
                price_target_1w=price_targets.get("1w"),
                price_target_1m=price_targets.get("1m"),
                price_target_3m=price_targets.get("3m"),
                key_insights=insights,
                risks=risks,
                catalysts=catalysts
            )

            return analysis

        except Exception as e:
            logger.error(f"Failed to generate analysis for {symbol}: {str(e)}")
            return None

    def _calculate_fundamental_score(self, fundamental_data: Dict[str, Any]) -> float:
        """Calculate fundamental analysis score"""
        score = 50.0

        # P/E ratio analysis
        pe_ratio = fundamental_data.get("pe_ratio")
        if pe_ratio:
            if 10 <= pe_ratio <= 20:
                score += 15
            elif 5 <= pe_ratio <= 25:
                score += 10
            elif pe_ratio > 40:
                score -= 15

        # Growth metrics
        revenue_growth = fundamental_data.get("revenue_growth")
        if revenue_growth and revenue_growth > 0.1:  # 10% growth
            score += 10

        earnings_growth = fundamental_data.get("earnings_growth")
        if earnings_growth and earnings_growth > 0.15:  # 15% growth
            score += 10

        # Profitability
        profit_margin = fundamental_data.get("profit_margin")
        if profit_margin and profit_margin > 0.1:  # 10% margin
            score += 10

        # Financial health
        debt_to_equity = fundamental_data.get("debt_to_equity")
        if debt_to_equity and debt_to_equity < 0.5:
            score += 5

        return max(0, min(100, score))

    def _calculate_sentiment_score(self, news_data: List[Dict[str, Any]]) -> float:
        """Calculate sentiment score from news data"""
        if not news_data:
            return 50.0

        # Simplified sentiment calculation
        # In production, would use NLP models for proper sentiment analysis
        total_sentiment = 0
        count = 0

        for news_item in news_data:
            title = news_item.get("title", "").lower()
            content = news_item.get("content", "").lower()
            text = title + " " + content

            # Simple keyword-based sentiment
            positive_words = ["up", "surge", "gain", "profit", "growth", "strong", "beat", "exceed"]
            negative_words = ["down", "fall", "loss", "decline", "weak", "miss", "concern", "risk"]

            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)

            if positive_count > negative_count:
                total_sentiment += 70
            elif negative_count > positive_count:
                total_sentiment += 30
            else:
                total_sentiment += 50

            count += 1

        return total_sentiment / count if count > 0 else 50.0

    def _calculate_overall_score(self, technical: float, fundamental: float, sentiment: float) -> float:
        """Calculate weighted overall score"""
        # Weight technical analysis more heavily for short-term picks
        weights = {"technical": 0.5, "fundamental": 0.3, "sentiment": 0.2}

        overall = (
            technical * weights["technical"] +
            fundamental * weights["fundamental"] +
            sentiment * weights["sentiment"]
        )

        return max(0, min(100, overall))

    def _generate_entry_strategy(self, symbol: str, current_price: float, technical_data: Dict[str, Any], fundamental_score: float) -> TradingStrategy:
        """Generate entry trading strategy"""
        support_resistance = technical_data.get("support_resistance", {})
        support_levels = support_resistance.get("support", [current_price * 0.95])

        entry_price = min(support_levels) if support_levels else current_price * 0.98
        stop_loss = entry_price * 0.95  # 5% stop loss
        take_profit = entry_price * 1.15  # 15% take profit

        confidence = min(100, fundamental_score + technical_data.get("technical_score", 50)) / 2

        return TradingStrategy(
            strategy_type="swing_trade",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=(take_profit - entry_price) / (entry_price - stop_loss),
            confidence_score=confidence,
            timeframe="1-4 weeks",
            reasoning=f"Entry near support level with {confidence:.1f}% confidence based on technical and fundamental analysis"
        )

    def _generate_exit_strategy(self, symbol: str, current_price: float, technical_data: Dict[str, Any], entry_strategy: TradingStrategy) -> TradingStrategy:
        """Generate exit trading strategy"""
        resistance_levels = technical_data.get("support_resistance", {}).get("resistance", [current_price * 1.1])

        exit_price = max(resistance_levels) if resistance_levels else entry_strategy.take_profit

        return TradingStrategy(
            strategy_type="profit_taking",
            entry_price=entry_strategy.entry_price,
            exit_price=exit_price,
            stop_loss=entry_strategy.stop_loss,
            take_profit=exit_price,
            risk_reward_ratio=entry_strategy.risk_reward_ratio,
            confidence_score=entry_strategy.confidence_score * 0.9,
            timeframe="2-6 weeks",
            reasoning=f"Exit near resistance level at ${exit_price:.2f} or on technical breakdown below ${entry_strategy.stop_loss:.2f}"
        )

    def _calculate_price_targets(self, current_price: float, technical_data: Dict[str, Any], fundamental_score: float) -> Dict[str, float]:
        """Calculate price targets for different timeframes"""
        volatility_multiplier = 1.0

        # Adjust for technical strength
        technical_score = technical_data.get("technical_score", 50)
        if technical_score > 70:
            volatility_multiplier = 1.2
        elif technical_score < 30:
            volatility_multiplier = 0.8

        return {
            "1w": current_price * (1 + (0.05 * volatility_multiplier)),
            "1m": current_price * (1 + (0.12 * volatility_multiplier)),
            "3m": current_price * (1 + (0.25 * volatility_multiplier))
        }

    def _generate_key_insights(self, symbol: str, stock_data: Dict, technical_data: Dict, news_data: List) -> List[str]:
        """Generate key insights for the stock"""
        insights = []

        # Technical insights
        if technical_data.get("technical_score", 0) > 70:
            insights.append("Strong technical momentum with multiple bullish indicators")

        # Volume insights
        volume_data = technical_data.get("indicators", {}).get("volume", {})
        volume_ratio = volume_data.get("volume_ratio", 1)
        if volume_ratio > 1.5:
            insights.append("Above-average trading volume indicates strong interest")

        # News insights
        if len(news_data) > 5:
            insights.append("High news coverage suggesting increased market attention")

        # Fundamental insights
        fundamental = stock_data.get("fundamental", {})
        if fundamental.get("revenue_growth", 0) > 0.15:
            insights.append("Strong revenue growth trajectory")

        return insights[:3]  # Limit to top 3 insights

    def _identify_stock_risks(self, symbol: str, stock_data: Dict, technical_data: Dict, market_data: Dict) -> List[str]:
        """Identify key risks for the stock"""
        risks = []

        # Technical risks
        rsi = technical_data.get("indicators", {}).get("rsi")
        if rsi and rsi > 80:
            risks.append("Overbought conditions may lead to short-term pullback")

        # Market risks
        market_sentiment = market_data.get("economic_indicators", {}).get("market_fear_greed")
        if market_sentiment == "Fear":
            risks.append("Overall market fear could impact stock performance")

        # Fundamental risks
        fundamental = stock_data.get("fundamental", {})
        debt_to_equity = fundamental.get("debt_to_equity")
        if debt_to_equity and debt_to_equity > 1:
            risks.append("High debt levels pose financial risk")

        return risks[:3]  # Limit to top 3 risks

    def _identify_catalysts(self, symbol: str, news_data: List, stock_data: Dict) -> List[str]:
        """Identify potential positive catalysts"""
        catalysts = []

        # News-based catalysts
        recent_news = [item for item in news_data if item.get("published_at")]
        if recent_news:
            catalysts.append("Recent positive news coverage")

        # Fundamental catalysts
        fundamental = stock_data.get("fundamental", {})
        if fundamental.get("earnings_growth", 0) > 0.2:
            catalysts.append("Strong earnings growth momentum")

        if fundamental.get("revenue_growth", 0) > 0.15:
            catalysts.append("Accelerating revenue growth")

        return catalysts[:3]  # Limit to top 3 catalysts

    def _determine_stock_sentiment(self, sentiment_score: float) -> MarketSentiment:
        """Convert sentiment score to MarketSentiment enum"""
        if sentiment_score >= 80:
            return MarketSentiment.VERY_BULLISH
        elif sentiment_score >= 60:
            return MarketSentiment.BULLISH
        elif sentiment_score >= 40:
            return MarketSentiment.NEUTRAL
        elif sentiment_score >= 20:
            return MarketSentiment.BEARISH
        else:
            return MarketSentiment.VERY_BEARISH

    def _assess_risk_level(self, technical_data: Dict, fundamental_score: float, sentiment_score: float) -> str:
        """Assess overall risk level"""
        risk_score = 0

        # Technical risk factors
        rsi = technical_data.get("indicators", {}).get("rsi")
        if rsi and (rsi > 80 or rsi < 20):
            risk_score += 1

        # Fundamental risk factors
        if fundamental_score < 40:
            risk_score += 1

        # Sentiment risk factors
        if sentiment_score < 30:
            risk_score += 1

        if risk_score >= 2:
            return "High"
        elif risk_score == 1:
            return "Medium"
        else:
            return "Low"

    def _select_top_picks(self, stock_analyses: List[StockAnalysis], max_picks: int) -> List[StockAnalysis]:
        """Select and rank top stock picks"""
        # Sort by overall score descending
        sorted_analyses = sorted(stock_analyses, key=lambda x: x.overall_score, reverse=True)

        # Filter out very low scores
        filtered_analyses = [analysis for analysis in sorted_analyses if analysis.overall_score >= 60]

        # Return top picks
        return filtered_analyses[:max_picks]

    def _generate_market_overview(self, market_data: Dict[str, Any]) -> str:
        """Generate market overview summary"""
        spy_data = market_data.get("SPY", {})
        market_trend = market_data.get("economic_indicators", {}).get("market_trend", "Unknown")
        fear_greed = market_data.get("economic_indicators", {}).get("market_fear_greed", "Unknown")

        return f"Market showing {market_trend.lower()} trend with {fear_greed.lower()} sentiment. " \
               f"S&P 500 at {spy_data.get('current_price', 'N/A')} " \
               f"({spy_data.get('change_percent', 0):+.2f}% today)."

    def _extract_key_themes(self, news_data: Dict, top_picks: List[StockAnalysis]) -> List[str]:
        """Extract key market themes from news and top picks"""
        themes = []

        # Analyze sectors of top picks
        if top_picks:
            themes.append("Growth momentum in technology and healthcare sectors")
            themes.append("Strong technical breakouts driving momentum")

        # Add general market themes
        themes.append("Continued focus on fundamental strength and growth")

        return themes[:3]

    def _identify_risk_factors(self, market_data: Dict, top_picks: List[StockAnalysis]) -> List[str]:
        """Identify key market risk factors"""
        risks = []

        fear_greed = market_data.get("economic_indicators", {}).get("market_fear_greed", "Unknown")
        if fear_greed == "Fear":
            risks.append("Elevated market fear levels")

        vix_data = market_data.get("VIX", {})
        if vix_data.get("current_price", 0) > 25:
            risks.append("High volatility environment")

        risks.append("General market correlation risk")

        return risks

    def _assess_market_sentiment(self, market_data: Dict[str, Any]) -> str:
        """Assess overall market sentiment"""
        fear_greed = market_data.get("economic_indicators", {}).get("market_fear_greed", "Neutral")
        market_trend = market_data.get("economic_indicators", {}).get("market_trend", "Sideways")

        return f"{fear_greed} sentiment with {market_trend.lower()} trend"

    def _generate_analysis_summary(self, all_analyses: List[StockAnalysis], top_picks: List[StockAnalysis]) -> Dict[str, Any]:
        """Generate summary statistics of the analysis"""
        if not all_analyses:
            return {}

        scores = [analysis.overall_score for analysis in all_analyses]

        return {
            "total_analyzed": len(all_analyses),
            "average_score": sum(scores) / len(scores),
            "top_score": max(scores),
            "picks_above_80": len([s for s in scores if s >= 80]),
            "picks_above_70": len([s for s in scores if s >= 70]),
            "picks_above_60": len([s for s in scores if s >= 60])
        }