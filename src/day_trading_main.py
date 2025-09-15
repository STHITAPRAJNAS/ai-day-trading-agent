#!/usr/bin/env python3
"""
Day Trading Stock Analysis Agent
Specialized for day trading with 10% profit targets

This application identifies stocks with strong potential for 10% intraday moves
based on sophisticated technical analysis, volume patterns, and momentum indicators.
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from agents.day_trading_coordinator import DayTradingCoordinator
from utils.day_trading_screener import DayTradingScreener, get_day_trading_universes
from utils.real_time_alerts import RealTimeAlertManager, setup_day_trading_alerts, setup_breakout_monitoring
from models.stock_data import DailyStockPicks

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('day_trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DayTradingApp:
    """Main application class for day trading analysis"""

    def __init__(self):
        self.coordinator = DayTradingCoordinator()
        self.screener = DayTradingScreener()
        self.alert_manager = RealTimeAlertManager(
            notification_channels=["console", "slack"] if os.getenv("SLACK_WEBHOOK_URL") else ["console"]
        )
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load application configuration"""
        return {
            "universe": os.getenv("DAY_TRADING_UNIVERSE", "HIGH_VOLUME_MOVERS"),
            "max_picks": int(os.getenv("MAX_DAY_TRADING_PICKS", "10")),
            "profit_target": float(os.getenv("PROFIT_TARGET_PERCENT", "10.0")),
            "stop_loss": float(os.getenv("STOP_LOSS_PERCENT", "2.0")),
            "time_preference": os.getenv("TIME_PREFERENCE", "INTRADAY"),  # SCALP, INTRADAY, SWING
            "output_dir": "day_trading_outputs",
            "enable_alerts": os.getenv("ENABLE_REAL_TIME_ALERTS", "True").lower() == "true",
            "auto_monitor": os.getenv("AUTO_MONITOR_PICKS", "True").lower() == "true"
        }

    async def run_day_trading_analysis(
        self,
        universe: Optional[str] = None,
        max_picks: Optional[int] = None,
        time_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive day trading analysis

        Args:
            universe: Day trading universe to analyze
            max_picks: Maximum number of picks to generate
            time_preference: SCALP, INTRADAY, or SWING

        Returns:
            Analysis results with day trading picks
        """
        universe = universe or self.config["universe"]
        max_picks = max_picks or self.config["max_picks"]
        time_preference = time_preference or self.config["time_preference"]

        logger.info(f"Starting day trading analysis for {universe} universe")
        logger.info(f"Target: {self.config['profit_target']}% profit with {self.config['stop_loss']}% stop loss")

        try:
            # Check market conditions
            market_conditions = await self._check_market_conditions()
            if not market_conditions["suitable_for_day_trading"]:
                logger.warning(f"Market conditions not ideal for day trading: {market_conditions['reason']}")

            # Run the coordinator analysis
            result = await self.coordinator.analyze({
                "universe": universe,
                "max_picks": max_picks,
                "time_preference": time_preference
            })

            # Process and enhance results
            enhanced_result = await self._enhance_day_trading_results(result)

            # Save results
            await self._save_day_trading_results(enhanced_result)

            # Set up real-time monitoring if enabled
            if self.config["enable_alerts"] and enhanced_result.get("day_trading_picks", {}).get("picks"):
                await self._setup_real_time_monitoring(enhanced_result["day_trading_picks"]["picks"])

            # Generate summary report
            summary = self._generate_day_trading_summary(enhanced_result)
            logger.info("Day trading analysis completed successfully")
            logger.info(f"Summary: {summary}")

            return enhanced_result

        except Exception as e:
            logger.error(f"Day trading analysis failed: {str(e)}")
            raise

    async def _check_market_conditions(self) -> Dict[str, Any]:
        """Check if market conditions are suitable for day trading"""
        try:
            # Check market hours
            current_hour = datetime.now().hour
            is_market_hours = 9 <= current_hour <= 16  # Assuming EST

            # Check if it's a weekday
            is_weekday = datetime.now().weekday() < 5

            # Basic volatility check (would enhance with VIX data)
            suitable = is_market_hours and is_weekday

            reason = ""
            if not is_weekday:
                reason = "Market closed (weekend)"
            elif not is_market_hours:
                reason = f"Outside market hours (current: {current_hour}:00)"
            else:
                reason = "Market conditions favorable"

            return {
                "suitable_for_day_trading": suitable,
                "is_market_hours": is_market_hours,
                "is_weekday": is_weekday,
                "current_hour": current_hour,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"Error checking market conditions: {str(e)}")
            return {"suitable_for_day_trading": False, "reason": "Error checking conditions"}

    async def _enhance_day_trading_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with additional day trading insights"""
        if "error" in result:
            return result

        day_trading_picks = result.get("day_trading_picks", {})
        picks = day_trading_picks.get("picks", [])

        # Enhance each pick with additional information
        for pick in picks:
            # Add realistic time estimates
            pick["estimated_time_to_target"] = self._estimate_time_to_target(pick)

            # Add position sizing recommendation
            pick["position_sizing"] = self._calculate_position_sizing(pick)

            # Add trade management plan
            pick["trade_management"] = self._generate_trade_management_plan(pick)

            # Add risk warnings
            pick["risk_warnings"] = self._generate_risk_warnings(pick)

        # Add session timing analysis
        result["session_analysis"] = self._analyze_trading_session()

        # Add market microstructure insights
        result["microstructure_insights"] = await self._get_microstructure_insights()

        return result

    def _estimate_time_to_target(self, pick: Dict[str, Any]) -> Dict[str, str]:
        """Estimate time to reach profit target"""
        volatility_analysis = pick.get("volatility_analysis", {})
        momentum_analysis = pick.get("momentum_analysis", {})

        volatility = volatility_analysis.get("annualized_volatility", 25)
        momentum_quality = momentum_analysis.get("momentum_quality", "FAIR")

        # Base estimate on volatility and momentum
        if volatility > 50 and momentum_quality == "EXCELLENT":
            estimate = "30 minutes - 2 hours"
            confidence = "HIGH"
        elif volatility > 30 and momentum_quality in ["EXCELLENT", "GOOD"]:
            estimate = "1 - 4 hours"
            confidence = "MEDIUM"
        elif volatility > 20:
            estimate = "2 - 6 hours"
            confidence = "MEDIUM"
        else:
            estimate = "Full trading day"
            confidence = "LOW"

        return {
            "estimate": estimate,
            "confidence": confidence,
            "basis": f"Volatility: {volatility:.1f}%, Momentum: {momentum_quality}"
        }

    def _calculate_position_sizing(self, pick: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate recommended position sizing"""
        entry_strategy = pick.get("entry_strategy", {})
        risk_level = pick.get("risk_level", "MEDIUM")

        entry_price = entry_strategy.get("entry_price", 0)
        stop_loss = entry_strategy.get("stop_loss", 0)

        # Risk per share
        risk_per_share = entry_price - stop_loss if entry_price > stop_loss else entry_price * 0.02

        # Risk multipliers based on risk level
        risk_multipliers = {
            "LOW": 1.0,
            "MEDIUM": 0.75,
            "HIGH": 0.5,
            "VERY_HIGH": 0.25
        }

        risk_multiplier = risk_multipliers.get(risk_level, 0.75)

        # Assuming 2% account risk per trade
        account_risk_percent = 2.0 * risk_multiplier

        return {
            "account_risk_percent": account_risk_percent,
            "risk_per_share": risk_per_share,
            "risk_multiplier": risk_multiplier,
            "recommendation": f"Risk {account_risk_percent:.1f}% of account",
            "notes": f"Adjusted for {risk_level} risk level"
        }

    def _generate_trade_management_plan(self, pick: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trade management plan"""
        entry_strategy = pick.get("entry_strategy", {})
        current_price = pick.get("current_price", 0)
        profit_target = pick.get("profit_target", 0)

        plan = {
            "entry_plan": {
                "strategy": "Limit order near support or on momentum breakout",
                "max_chase": f"${current_price * 1.005:.2f}",  # 0.5% above current
                "timeout": "30 minutes after signal"
            },
            "profit_taking": {
                "target_1": f"${current_price * 1.05:.2f} (50% position)",
                "target_2": f"${current_price * 1.08:.2f} (30% position)",
                "final_target": f"${profit_target:.2f} (20% position)"
            },
            "stop_management": {
                "initial_stop": f"${entry_strategy.get('stop_loss', 0):.2f}",
                "break_even": "Move to break-even after 5% profit",
                "trailing_stop": "Consider trailing stop after 7% profit"
            },
            "time_management": {
                "max_hold_time": "6 hours intraday",
                "exit_before": "3:30 PM EST",
                "weekend_rule": "Close all positions by Friday 3:00 PM"
            }
        }

        return plan

    def _generate_risk_warnings(self, pick: Dict[str, Any]) -> List[str]:
        """Generate specific risk warnings"""
        warnings = []

        risk_level = pick.get("risk_level", "MEDIUM")
        if risk_level in ["HIGH", "VERY_HIGH"]:
            warnings.append(f"⚠️ {risk_level} risk - consider reduced position size")

        # Volatility warnings
        volatility_analysis = pick.get("volatility_analysis", {})
        if volatility_analysis.get("annualized_volatility", 0) > 60:
            warnings.append("⚠️ High volatility - expect large price swings")

        # Liquidity warnings
        liquidity_analysis = pick.get("liquidity_analysis", {})
        if not liquidity_analysis.get("is_liquid", True):
            warnings.append("⚠️ Lower liquidity - use limit orders only")

        # Time warnings
        current_hour = datetime.now().hour
        if current_hour >= 15:
            warnings.append("⚠️ Late session - reduced time to target")

        # Gap warnings
        gap_analysis = pick.get("gap_analysis", {})
        if gap_analysis.get("gap_size_category") in ["LARGE", "HUGE"]:
            warnings.append("⚠️ Large gap - higher than normal risk")

        return warnings

    def _analyze_trading_session(self) -> Dict[str, Any]:
        """Analyze current trading session characteristics"""
        current_time = datetime.now()
        hour = current_time.hour

        if 9 <= hour < 10:
            session = "OPENING"
            characteristics = "High volatility, gap fills, breakout opportunities"
            optimal_for = "Gap trading, momentum breakouts"
        elif 10 <= hour < 11:
            session = "MORNING"
            characteristics = "Settling into trend, volume normalizing"
            optimal_for = "Trend continuation, pullback entries"
        elif 11 <= hour < 14:
            session = "MIDDAY"
            characteristics = "Lower volatility, range-bound trading"
            optimal_for = "Range trading, mean reversion"
        elif 14 <= hour < 15:
            session = "AFTERNOON"
            characteristics = "Renewed interest, positioning for close"
            optimal_for = "Breakout attempts, fresh momentum"
        elif 15 <= hour < 16:
            session = "POWER_HOUR"
            characteristics = "High volume, final positioning, volatility"
            optimal_for = "Quick scalps, avoid new positions"
        else:
            session = "AFTER_HOURS"
            characteristics = "Thin volume, wider spreads"
            optimal_for = "Limited opportunities, exit planning"

        return {
            "current_session": session,
            "characteristics": characteristics,
            "optimal_for": optimal_for,
            "hour": hour,
            "recommendations": self._get_session_recommendations(session)
        }

    def _get_session_recommendations(self, session: str) -> List[str]:
        """Get session-specific recommendations"""
        recommendations = {
            "OPENING": [
                "Monitor for gap fills and reversals",
                "Wait for first 30 minutes to settle",
                "Focus on high-volume breakouts"
            ],
            "MORNING": [
                "Look for pullbacks in trending stocks",
                "Trade with the established trend",
                "Monitor for momentum continuation"
            ],
            "MIDDAY": [
                "Consider range trading strategies",
                "Look for mean reversion opportunities",
                "Reduce position sizes in choppy conditions"
            ],
            "AFTERNOON": [
                "Watch for fresh breakouts",
                "Monitor institutional activity",
                "Prepare for power hour volatility"
            ],
            "POWER_HOUR": [
                "Focus on quick scalps only",
                "Avoid new swing positions",
                "Close positions before market close"
            ],
            "AFTER_HOURS": [
                "Avoid new positions",
                "Plan for next day's trades",
                "Review and analyze today's performance"
            ]
        }

        return recommendations.get(session, ["Exercise caution"])

    async def _get_microstructure_insights(self) -> Dict[str, Any]:
        """Get market microstructure insights"""
        # This would integrate with more sophisticated data in production
        return {
            "market_regime": "Normal volatility regime",
            "institutional_flow": "Neutral to slightly bullish",
            "sector_rotation": "Technology showing relative strength",
            "volume_profile": "Average volume, no significant anomalies",
            "recommendations": [
                "Standard day trading approaches should work",
                "Monitor for any sector-specific news",
                "Watch for volume spikes in tech names"
            ]
        }

    async def _save_day_trading_results(self, result: Dict[str, Any]) -> None:
        """Save day trading results to files"""
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save complete results
        results_file = os.path.join(output_dir, f"day_trading_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # Save picks summary
        if "day_trading_picks" in result:
            picks_file = os.path.join(output_dir, f"day_trading_picks_{timestamp}.json")
            with open(picks_file, 'w') as f:
                json.dump(result["day_trading_picks"], f, indent=2, default=str)

        # Save human-readable summary
        summary_file = os.path.join(output_dir, f"day_trading_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(self._generate_detailed_day_trading_summary(result))

        # Save alert configuration if monitoring enabled
        if self.config["enable_alerts"]:
            alert_config_file = os.path.join(output_dir, f"alert_config_{timestamp}.json")
            alert_config = self._generate_alert_config(result)
            with open(alert_config_file, 'w') as f:
                json.dump(alert_config, f, indent=2)

        logger.info(f"Day trading results saved to {output_dir}")

    async def _setup_real_time_monitoring(self, picks: List[Dict[str, Any]]) -> None:
        """Set up real-time monitoring for selected picks"""
        if not self.config["auto_monitor"]:
            return

        try:
            # Set up alerts for each pick
            for pick in picks:
                symbol = pick["symbol"]
                entry_price = pick.get("entry_strategy", {}).get("entry_price", pick.get("current_price", 0))

                if entry_price > 0:
                    self.alert_manager.add_day_trading_alerts(
                        symbol,
                        entry_price,
                        self.config["profit_target"],
                        self.config["stop_loss"]
                    )

                    # Add breakout alerts
                    support_resistance = pick.get("support_resistance", {})
                    resistance_levels = support_resistance.get("key_resistance", [])
                    support_levels = support_resistance.get("key_support", [])

                    if resistance_levels and support_levels:
                        self.alert_manager.add_breakout_alerts(
                            symbol,
                            max(resistance_levels),
                            min(support_levels)
                        )

            # Start monitoring
            await self.alert_manager.start_monitoring()

            logger.info(f"Real-time monitoring started for {len(picks)} picks")

        except Exception as e:
            logger.error(f"Failed to set up real-time monitoring: {str(e)}")

    def _generate_alert_config(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alert configuration for manual setup"""
        picks = result.get("day_trading_picks", {}).get("picks", [])

        alert_config = {
            "timestamp": datetime.now().isoformat(),
            "profit_target": self.config["profit_target"],
            "stop_loss": self.config["stop_loss"],
            "symbols": {}
        }

        for pick in picks:
            symbol = pick["symbol"]
            entry_price = pick.get("entry_strategy", {}).get("entry_price", pick.get("current_price", 0))

            alert_config["symbols"][symbol] = {
                "entry_price": entry_price,
                "profit_target": entry_price * (1 + self.config["profit_target"] / 100),
                "stop_loss": entry_price * (1 - self.config["stop_loss"] / 100),
                "support_resistance": pick.get("support_resistance", {}),
                "risk_level": pick.get("risk_level", "MEDIUM")
            }

        return alert_config

    def _generate_day_trading_summary(self, result: Dict[str, Any]) -> str:
        """Generate concise summary"""
        if "error" in result:
            return f"Analysis failed: {result['error']}"

        picks_count = result.get("final_picks_count", 0)
        total_analyzed = result.get("total_analyzed", 0)
        market_conditions = result.get("market_conditions", {})

        return f"Generated {picks_count} day trading picks from {total_analyzed} analyzed. Market: {market_conditions.get('overall_condition', 'Unknown')}"

    def _generate_detailed_day_trading_summary(self, result: Dict[str, Any]) -> str:
        """Generate detailed human-readable summary"""
        if "error" in result:
            return f"Day Trading Analysis Failed\n=========================\nError: {result['error']}\n"

        summary = []
        summary.append("DAY TRADING ANALYSIS REPORT")
        summary.append("=" * 50)
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Target: {self.config['profit_target']}% profit, {self.config['stop_loss']}% stop loss")
        summary.append("")

        # Market conditions
        market_conditions = result.get("market_conditions", {})
        if market_conditions:
            summary.append("MARKET CONDITIONS")
            summary.append("-" * 20)
            summary.append(f"Overall: {market_conditions.get('overall_condition', 'Unknown')}")
            summary.append(f"VIX Level: {market_conditions.get('vix_level', 'N/A')}")
            summary.append(f"Market Movement: {market_conditions.get('market_movement', 0):+.2f}%")
            summary.append("")

        # Session analysis
        session_analysis = result.get("session_analysis", {})
        if session_analysis:
            summary.append("TRADING SESSION")
            summary.append("-" * 18)
            summary.append(f"Current Session: {session_analysis.get('current_session', 'Unknown')}")
            summary.append(f"Characteristics: {session_analysis.get('characteristics', 'N/A')}")
            summary.append(f"Optimal For: {session_analysis.get('optimal_for', 'N/A')}")
            summary.append("")

        # Top picks
        day_trading_picks = result.get("day_trading_picks", {})
        picks = day_trading_picks.get("picks", [])

        if picks:
            summary.append(f"TOP {len(picks)} DAY TRADING PICKS")
            summary.append("-" * 35)

            for i, pick in enumerate(picks, 1):
                symbol = pick['symbol']
                current_price = pick['current_price']
                score = pick['day_trading_score']
                signal = pick['trading_signal']
                urgency = pick['urgency']
                risk_level = pick['risk_level']

                summary.append(f"{i:2d}. {symbol:6s} - ${current_price:8.2f} (Score: {score:5.1f}/100)")
                summary.append(f"      Signal: {signal:15s} Urgency: {urgency:6s} Risk: {risk_level}")

                # Entry/exit info
                entry_strategy = pick.get('entry_strategy', {})
                if entry_strategy:
                    entry_price = entry_strategy.get('entry_price', 0)
                    stop_loss = entry_strategy.get('stop_loss', 0)
                    take_profit = entry_strategy.get('take_profit', 0)
                    summary.append(f"      Entry: ${entry_price:.2f} | Stop: ${stop_loss:.2f} | Target: ${take_profit:.2f}")

                # Time estimate
                time_estimate = pick.get('estimated_time_to_target', {})
                if time_estimate:
                    summary.append(f"      Time Est: {time_estimate.get('estimate', 'Unknown')}")

                # Key insight
                insights = pick.get('key_insights', [])
                if insights:
                    summary.append(f"      Insight: {insights[0][:60]}{'...' if len(insights[0]) > 60 else ''}")

                # Risk warnings
                warnings = pick.get('risk_warnings', [])
                if warnings:
                    summary.append(f"      Warning: {warnings[0]}")

                summary.append("")

        # Key insights
        microstructure = result.get("microstructure_insights", {})
        if microstructure:
            summary.append("MARKET INSIGHTS")
            summary.append("-" * 17)
            summary.append(f"Market Regime: {microstructure.get('market_regime', 'N/A')}")
            summary.append(f"Institutional Flow: {microstructure.get('institutional_flow', 'N/A')}")
            summary.append(f"Sector Rotation: {microstructure.get('sector_rotation', 'N/A')}")
            summary.append("")

        # Analysis summary
        analysis_summary = result.get("analysis_summary", {})
        if analysis_summary:
            summary.append("ANALYSIS STATISTICS")
            summary.append("-" * 22)
            summary.append(f"Total analyzed: {analysis_summary.get('total_analyzed', 0)} stocks")
            summary.append(f"Candidates screened: {result.get('candidates_screened', 0)}")
            summary.append(f"Final picks: {result.get('final_picks_count', 0)}")
            summary.append(f"Average score: {analysis_summary.get('final_picks_avg_score', 0):.1f}/100")
            summary.append("")

        # Monitoring status
        if self.config["enable_alerts"]:
            summary.append("REAL-TIME MONITORING")
            summary.append("-" * 22)
            summary.append("✓ Real-time alerts enabled")
            summary.append(f"✓ Monitoring {len(picks)} symbols")
            summary.append(f"✓ Profit target: {self.config['profit_target']}%")
            summary.append(f"✓ Stop loss: {self.config['stop_loss']}%")
            summary.append("")

        summary.append("📁 Detailed results saved to: day_trading_outputs/")
        summary.append("")
        summary.append("⚠️  IMPORTANT DISCLAIMERS:")
        summary.append("   • For educational purposes only")
        summary.append("   • Not financial advice")
        summary.append("   • Day trading involves significant risk")
        summary.append("   • Past performance does not guarantee future results")

        return "\n".join(summary)

    async def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status"""
        if not self.alert_manager:
            return {"error": "Alert manager not initialized"}

        return {
            "is_monitoring": self.alert_manager.is_running,
            "statistics": self.alert_manager.get_statistics(),
            "active_alerts": [alert.__dict__ for alert in self.alert_manager.get_active_alerts()],
            "recent_alerts": [alert.__dict__ for alert in self.alert_manager.get_alert_history(2)]
        }

    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.alert_manager and self.alert_manager.is_running:
            await self.alert_manager.stop_monitoring()
            logger.info("Stopped real-time monitoring")

async def main():
    """Main day trading application entry point"""
    app = DayTradingApp()

    print("🎯 Day Trading Stock Analysis Agent")
    print("Optimized for 10% Profit Targets")
    print("=" * 50)

    # Show available universes
    universes = get_day_trading_universes()
    print("\n📊 Available Day Trading Universes:")
    for name, symbols in universes.items():
        print(f"  • {name}: {len(symbols)} stocks")
    print()

    # Show current configuration
    print("⚙️ Configuration:")
    print(f"  • Universe: {app.config['universe']}")
    print(f"  • Max Picks: {app.config['max_picks']}")
    print(f"  • Profit Target: {app.config['profit_target']}%")
    print(f"  • Stop Loss: {app.config['stop_loss']}%")
    print(f"  • Time Preference: {app.config['time_preference']}")
    print(f"  • Real-time Alerts: {'✓' if app.config['enable_alerts'] else '✗'}")
    print()

    # Check market conditions
    print("🕐 Checking market conditions...")
    market_check = await app._check_market_conditions()
    if market_check["suitable_for_day_trading"]:
        print(f"✅ {market_check['reason']}")
    else:
        print(f"⚠️  {market_check['reason']}")
        print("   Analysis will continue but conditions may not be optimal.")
    print()

    print("🔍 Running day trading analysis...")
    print("This analysis focuses on stocks with 10% upside potential.")
    print()

    try:
        # Run the analysis
        result = await app.run_day_trading_analysis()

        # Display results
        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            return 1

        day_trading_picks = result.get("day_trading_picks", {})
        picks = day_trading_picks.get("picks", [])

        if not picks:
            print("⚠️  No day trading opportunities found.")
            print("   Market conditions may not be favorable.")
            print("   Try again during higher volatility periods.")
            return 0

        print("✅ Day trading analysis complete!")
        print("=" * 40)
        print()

        # Show market overview
        market_overview = day_trading_picks.get("market_overview", "")
        if market_overview:
            print(f"📈 Market Overview:")
            print(f"   {market_overview}")
            print()

        # Show session info
        session_analysis = result.get("session_analysis", {})
        if session_analysis:
            session = session_analysis.get("current_session", "Unknown")
            optimal_for = session_analysis.get("optimal_for", "N/A")
            print(f"🕐 Trading Session: {session}")
            print(f"   Optimal for: {optimal_for}")
            print()

        # Show top picks
        print(f"🎯 Top {len(picks)} Day Trading Picks:")
        print("-" * 45)

        for i, pick in enumerate(picks, 1):
            symbol = pick['symbol']
            price = pick['current_price']
            score = pick['day_trading_score']
            signal = pick['trading_signal']
            urgency = pick['urgency']

            urgency_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            emoji = urgency_emoji.get(urgency, "⚪")

            print(f"{i:2d}. {symbol:6s} - ${price:8.2f} (Score: {score:5.1f}) {emoji}")
            print(f"      Signal: {signal:12s} Urgency: {urgency}")

            # Entry info
            entry_strategy = pick.get('entry_strategy', {})
            if entry_strategy:
                entry = entry_strategy.get('entry_price', 0)
                target = entry_strategy.get('take_profit', 0)
                stop = entry_strategy.get('stop_loss', 0)
                rr_ratio = entry_strategy.get('risk_reward_ratio', 0)
                print(f"      Entry: ${entry:.2f} → Target: ${target:.2f} (R/R: {rr_ratio:.1f})")

            # Time estimate
            time_est = pick.get('estimated_time_to_target', {})
            if time_est:
                estimate = time_est.get('estimate', 'Unknown')
                confidence = time_est.get('confidence', 'LOW')
                print(f"      Time: {estimate} ({confidence} confidence)")

            print()

        # Show analysis stats
        analysis_summary = result.get("analysis_summary", {})
        if analysis_summary:
            print("📊 Analysis Summary:")
            print(f"   • Analyzed: {analysis_summary.get('total_analyzed', 0)} stocks")
            print(f"   • Screened: {result.get('candidates_screened', 0)} candidates")
            print(f"   • Selected: {len(picks)} final picks")
            print(f"   • Avg Score: {analysis_summary.get('final_picks_avg_score', 0):.1f}/100")
            print()

        # Monitoring status
        if app.config["enable_alerts"]:
            print("🔔 Real-time Monitoring:")
            print(f"   • Monitoring {len(picks)} symbols")
            print(f"   • Profit alerts at {app.config['profit_target']}%")
            print(f"   • Stop loss alerts at {app.config['stop_loss']}%")
            print(f"   • Notifications: {'Slack + Console' if os.getenv('SLACK_WEBHOOK_URL') else 'Console'}")
            print()

        print("📁 Detailed results saved to: day_trading_outputs/")
        print()
        print("Next steps:")
        print("  • Review individual pick details in saved files")
        print("  • Set up limit orders near recommended entry prices")
        print("  • Monitor alerts for profit targets and stop losses")
        print("  • Consider position sizing based on risk assessments")

        # Keep monitoring running if enabled
        if app.config["enable_alerts"] and app.alert_manager.is_running:
            print()
            print("🔄 Real-time monitoring is active...")
            print("   Press Ctrl+C to stop monitoring and exit")

            try:
                while True:
                    await asyncio.sleep(10)
                    # Could add periodic status updates here
            except KeyboardInterrupt:
                print("\n\n🛑 Stopping monitoring...")
                await app.stop_monitoring()

        return 0

    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        logger.exception("Application failed")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 Application interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        exit(1)