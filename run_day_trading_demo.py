#!/usr/bin/env python3
"""
Day Trading Demo for Stock Analysis Agent
Quick demonstration of day trading capabilities with 10% profit targets
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from day_trading_main import DayTradingApp
from utils.day_trading_screener import get_day_trading_universes

async def run_day_trading_demo():
    """Run a focused day trading demo"""

    print("🎯 Day Trading Stock Analysis Agent - DEMO")
    print("Optimized for 10% Profit Targets")
    print("=" * 55)
    print()

    # Initialize the app
    app = DayTradingApp()

    # Show demo configuration
    print("🔧 Demo Configuration:")
    print("  • Universe: HIGH_VOLUME_MOVERS (focused set)")
    print("  • Max Picks: 5 (quick demo)")
    print("  • Profit Target: 10%")
    print("  • Stop Loss: 2%")
    print("  • Focus: Intraday momentum plays")
    print()

    # Show what makes this different
    print("🎯 Day Trading Focus:")
    print("  • Intraday volatility analysis (1m, 5m, 15m timeframes)")
    print("  • VWAP-based entry/exit strategies")
    print("  • Volume surge detection")
    print("  • Gap analysis and momentum breakouts")
    print("  • Real-time profit target calculations")
    print("  • Risk management with tight stops")
    print()

    # Check market timing
    current_hour = datetime.now().hour
    market_status = ""
    if 9 <= current_hour <= 16:
        market_status = "🟢 MARKET OPEN - Optimal for day trading"
    elif current_hour < 9:
        market_status = "🟡 PRE-MARKET - Limited data availability"
    else:
        market_status = "🔴 AFTER HOURS - Using last available data"

    print(f"⏰ Market Status: {market_status}")
    print(f"   Current time: {datetime.now().strftime('%H:%M:%S EST')}")
    print()

    # Show available universes
    universes = get_day_trading_universes()
    print("📊 Day Trading Universes Available:")
    for name, symbols in list(universes.items())[:3]:  # Show first 3
        print(f"  • {name}: {len(symbols)} liquid, volatile stocks")
    print()

    print("🔍 Running day trading analysis...")
    print("   Focusing on stocks with strong 10% move potential...")
    print("   This may take 2-3 minutes for detailed analysis...")
    print()

    try:
        # Run focused day trading analysis
        result = await app.run_day_trading_analysis(
            universe="HIGH_VOLUME_MOVERS",  # High-volume, volatile stocks
            max_picks=5,                    # Fewer picks for demo
            time_preference="INTRADAY"      # Intraday focus
        )

        # Display results
        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            print()
            print("Common issues:")
            print("  • Market closed (weekends)")
            print("  • Network connectivity")
            print("  • API rate limits")
            print("  • Low volatility environment")
            return

        day_trading_picks = result.get("day_trading_picks", {})
        picks = day_trading_picks.get("picks", [])

        if not picks:
            print("⚠️  No day trading opportunities found")
            print()
            print("Possible reasons:")
            print("  • Low market volatility today")
            print("  • All stocks below quality threshold")
            print("  • Market conditions not favorable")
            print("  • Outside optimal trading hours")
            print()
            print("💡 Day trading works best during:")
            print("  • High volatility periods (VIX > 20)")
            print("  • Market open hours (9:30 AM - 4:00 PM EST)")
            print("  • News-driven market sessions")
            print("  • Earnings seasons")
            return

        print("✅ Day Trading Analysis Complete!")
        print("=" * 45)
        print()

        # Market conditions
        market_overview = day_trading_picks.get("market_overview", "")
        if market_overview:
            print(f"📈 Market Conditions:")
            print(f"   {market_overview}")
            print()

        # Trading session analysis
        session_analysis = result.get("session_analysis", {})
        if session_analysis:
            current_session = session_analysis.get("current_session", "Unknown")
            optimal_for = session_analysis.get("optimal_for", "Unknown")
            print(f"🕐 Trading Session: {current_session}")
            print(f"   Best for: {optimal_for}")
            print()

        # Top day trading picks
        print(f"🎯 TOP {len(picks)} DAY TRADING PICKS")
        print("=" * 35)
        print()

        for i, pick in enumerate(picks, 1):
            symbol = pick['symbol']
            price = pick['current_price']
            score = pick['day_trading_score']
            signal = pick['trading_signal']
            urgency = pick['urgency']
            risk_level = pick['risk_level']

            # Urgency indicators
            urgency_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            emoji = urgency_emoji.get(urgency, "⚪")

            print(f"{i}. {symbol} - ${price:.2f} (Score: {score:.1f}/100) {emoji}")
            print(f"   Signal: {signal} | Urgency: {urgency} | Risk: {risk_level}")

            # Entry/Exit Strategy
            entry_strategy = pick.get('entry_strategy', {})
            if entry_strategy:
                entry_price = entry_strategy.get('entry_price', 0)
                stop_loss = entry_strategy.get('stop_loss', 0)
                take_profit = entry_strategy.get('take_profit', 0)
                risk_reward = entry_strategy.get('risk_reward_ratio', 0)

                print(f"   📍 Entry: ${entry_price:.2f}")
                print(f"   🎯 Target: ${take_profit:.2f} (+10.0%)")
                print(f"   🛑 Stop: ${stop_loss:.2f} (-{((entry_price - stop_loss) / entry_price * 100):.1f}%)")
                print(f"   📊 Risk/Reward: 1:{risk_reward:.1f}")

            # Time Estimate
            time_estimate = pick.get('estimated_time_to_target', {})
            if time_estimate:
                estimate = time_estimate.get('estimate', 'Unknown')
                confidence = time_estimate.get('confidence', 'LOW')
                print(f"   ⏱️  Time to Target: {estimate} ({confidence} confidence)")

            # Key Insights
            insights = pick.get('key_insights', [])
            if insights:
                print(f"   💡 Key Insight: {insights[0]}")

            # Volatility & Volume Info
            volatility_analysis = pick.get('volatility_analysis', {})
            momentum_analysis = pick.get('momentum_analysis', {})

            vol_level = volatility_analysis.get('annualized_volatility', 0)
            is_high_vol = volatility_analysis.get('is_high_volatility', False)
            momentum_quality = momentum_analysis.get('momentum_quality', 'UNKNOWN')

            print(f"   📈 Volatility: {vol_level:.1f}% {'(HIGH)' if is_high_vol else '(NORMAL)'}")
            print(f"   🚀 Momentum: {momentum_quality}")

            # Risk Warnings
            warnings = pick.get('risk_warnings', [])
            if warnings:
                print(f"   ⚠️  Warning: {warnings[0]}")

            print()

        # Analysis Statistics
        analysis_summary = result.get("analysis_summary", {})
        if analysis_summary:
            print("📊 Analysis Statistics:")
            total_analyzed = analysis_summary.get('total_analyzed', 0)
            candidates_screened = result.get('candidates_screened', 0)
            avg_score = analysis_summary.get('final_picks_avg_score', 0)

            print(f"   • Total stocks analyzed: {total_analyzed}")
            print(f"   • Candidates passing screen: {candidates_screened}")
            print(f"   • Final picks selected: {len(picks)}")
            print(f"   • Average pick score: {avg_score:.1f}/100")
            print()

        # Market Microstructure
        microstructure = result.get("microstructure_insights", {})
        if microstructure:
            print("🔬 Market Insights:")
            print(f"   • Regime: {microstructure.get('market_regime', 'N/A')}")
            print(f"   • Flow: {microstructure.get('institutional_flow', 'N/A')}")
            print(f"   • Sector Focus: {microstructure.get('sector_rotation', 'N/A')}")
            print()

        # Real-time monitoring status
        if app.config.get("enable_alerts", False):
            print("🔔 Real-Time Monitoring:")
            print("   ✅ Profit target alerts enabled")
            print("   ✅ Stop loss alerts enabled")
            print("   ✅ Volume surge detection active")
            print("   ✅ Momentum breakout alerts set")
            print()

        print("📁 Detailed Results:")
        print("   • Complete analysis saved to: day_trading_outputs/")
        print("   • Alert configurations generated")
        print("   • Trade management plans included")
        print()

        print("🎯 Day Trading Workflow:")
        print("   1. Review picks and select 1-2 highest conviction")
        print("   2. Set limit orders near recommended entry prices")
        print("   3. Monitor for profit targets (10%) and stops (2%)")
        print("   4. Use trailing stops after 5-7% profit")
        print("   5. Close all positions before 3:30 PM EST")
        print()

        print("⚠️  Important Reminders:")
        print("   • This is for educational/demo purposes only")
        print("   • Day trading involves significant risk")
        print("   • Never risk more than you can afford to lose")
        print("   • Past performance doesn't guarantee future results")
        print("   • Consider paper trading first")
        print()

        print("🎉 Demo completed successfully!")
        print()
        print("Next Steps:")
        print("  • Run full analysis: cd src && python day_trading_main.py")
        print("  • View dashboard: cd src/dashboard && python app.py")
        print("  • Set up scheduling: cd src && python scheduler.py schedule")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print()
        print("Troubleshooting:")
        print("  • Check internet connection")
        print("  • Ensure dependencies installed: uv sync")
        print("  • Try during market hours for best results")
        print("  • Check logs for detailed error information")

        import traceback
        print(f"\nDetailed error:")
        traceback.print_exc()

def main():
    """Main demo function"""
    print("Starting Day Trading Demo...")
    print()

    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("❌ Please run this script from the project root directory")
        print("   cd stock-analysis-agent")
        print("   python run_day_trading_demo.py")
        return

    # Run the demo
    try:
        asyncio.run(run_day_trading_demo())
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")

if __name__ == "__main__":
    main()