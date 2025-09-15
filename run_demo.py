#!/usr/bin/env python3
"""
Quick Demo Runner for Stock Analysis Agent
Run this to see the agent in action with a small sample
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from main import StockAnalysisApp
from utils.stock_universe import get_universe_info

async def run_demo():
    """Run a quick demo of the stock analysis system"""

    print("🚀 Stock Analysis Agent Demo")
    print("=" * 50)
    print()

    # Initialize the app
    app = StockAnalysisApp()

    # Show available universes
    print("📊 Available Stock Universes:")
    universes = get_universe_info()
    for name, info in list(universes.items())[:4]:  # Show first 4
        print(f"  • {name}: {info['description']} ({info['count']} stocks)")
    print()

    # Run analysis on a small universe for demo
    print("🔍 Running analysis on MEGA_CAP universe (20 stocks)...")
    print("This may take 2-3 minutes...")
    print()

    try:
        # Run the analysis
        result = await app.run_daily_analysis(
            universe="MEGA_CAP",  # Smaller universe for faster demo
            max_picks=5           # Fewer picks for demo
        )

        # Display results
        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            return

        daily_picks = result.get("daily_picks", {})
        picks = daily_picks.get("picks", [])

        if not picks:
            print("⚠️  No picks generated. This might be due to:")
            print("   • Market being closed")
            print("   • Insufficient data")
            print("   • Network connectivity issues")
            return

        print("✅ Analysis Complete!")
        print("=" * 30)
        print()

        # Market overview
        market_overview = daily_picks.get("market_overview", "")
        if market_overview:
            print(f"📈 Market Overview:")
            print(f"   {market_overview}")
            print()

        # Top picks
        print(f"🎯 Top {len(picks)} Stock Picks:")
        print("-" * 40)

        for i, pick in enumerate(picks, 1):
            symbol = pick['symbol']
            price = pick['current_price']
            score = pick['overall_score']
            signal = pick['technical_signal']
            risk = pick['risk_level']

            print(f"{i:2d}. {symbol:6s} - ${price:8.2f} (Score: {score:5.1f}/100)")
            print(f"      Signal: {signal:15s} Risk: {risk}")

            # Entry strategy
            entry = pick.get('entry_strategy', {})
            if entry:
                entry_price = entry.get('entry_price', 0)
                stop_loss = entry.get('stop_loss', 0)
                take_profit = entry.get('take_profit', 0)
                print(f"      Entry: ${entry_price:.2f} | Stop: ${stop_loss:.2f} | Target: ${take_profit:.2f}")

            # Key insights
            insights = pick.get('key_insights', [])
            if insights:
                print(f"      Insight: {insights[0][:50]}{'...' if len(insights[0]) > 50 else ''}")

            print()

        # Analysis summary
        analysis_summary = result.get("analysis_summary", {})
        if analysis_summary:
            print("📊 Analysis Summary:")
            print(f"   • Total analyzed: {analysis_summary.get('total_analyzed', 0)} stocks")
            print(f"   • Average score: {analysis_summary.get('average_score', 0):.1f}/100")
            print(f"   • High quality picks (70+): {analysis_summary.get('picks_above_70', 0)}")
            print()

        # Key themes
        key_themes = daily_picks.get("key_themes", [])
        if key_themes:
            print("💡 Key Market Themes:")
            for theme in key_themes[:3]:
                print(f"   • {theme}")
            print()

        # Risk factors
        risk_factors = daily_picks.get("risk_factors", [])
        if risk_factors:
            print("⚠️  Risk Factors:")
            for risk in risk_factors[:3]:
                print(f"   • {risk}")
            print()

        print("📁 Detailed results saved to: outputs/")
        print()
        print("🎉 Demo completed successfully!")
        print()
        print("Next steps:")
        print("  • Run full analysis: cd src && python main.py")
        print("  • Start dashboard: cd src/dashboard && python app.py")
        print("  • Schedule daily runs: cd src && python scheduler.py schedule")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print()
        print("Troubleshooting:")
        print("  • Check internet connection")
        print("  • Ensure all dependencies are installed: uv sync")
        print("  • Check logs for detailed error information")

        import traceback
        print("\nDetailed error:")
        traceback.print_exc()

def main():
    """Main demo function"""
    print("Starting Stock Analysis Agent Demo...")
    print()

    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("❌ Please run this script from the project root directory")
        print("   cd stock-analysis-agent")
        print("   python run_demo.py")
        return

    # Run the demo
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")

if __name__ == "__main__":
    main()