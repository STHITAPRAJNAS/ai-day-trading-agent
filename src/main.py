#!/usr/bin/env python3
"""
Sophisticated Stock Analysis Agent
Main application entry point using Google ADK

This application analyzes stocks and generates daily picks with entry/exit strategies
based on technical analysis, fundamental data, and news sentiment.
"""

import asyncio
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from agents.master_coordinator import MasterCoordinator
from utils.stock_universe import get_stock_universe, get_universe_info
from models.stock_data import DailyStockPicks

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class StockAnalysisApp:
    """Main application class for the stock analysis system"""

    def __init__(self):
        self.coordinator = MasterCoordinator()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load application configuration"""
        return {
            "universe": os.getenv("STOCK_UNIVERSE", "SP500_SAMPLE"),
            "max_picks": int(os.getenv("MAX_PICKS", "10")),
            "analysis_interval": int(os.getenv("ANALYSIS_INTERVAL", "3600")),  # seconds
            "output_dir": "outputs",
            "debug": os.getenv("DEBUG", "True").lower() == "true"
        }

    async def run_daily_analysis(self, universe: Optional[str] = None, max_picks: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete daily stock analysis

        Args:
            universe: Stock universe to analyze (optional, uses config default)
            max_picks: Maximum number of picks to generate (optional, uses config default)

        Returns:
            Analysis results with daily picks
        """
        universe = universe or self.config["universe"]
        max_picks = max_picks or self.config["max_picks"]

        logger.info(f"Starting daily analysis for {universe} universe")
        logger.info(f"Generating top {max_picks} stock picks")

        try:
            # Run the master coordinator analysis
            result = await self.coordinator.analyze({
                "universe": universe,
                "max_picks": max_picks
            })

            # Save results
            await self._save_results(result)

            # Generate summary report
            summary = self._generate_summary_report(result)
            logger.info("Daily analysis completed successfully")
            logger.info(f"Summary: {summary}")

            return result

        except Exception as e:
            logger.error(f"Daily analysis failed: {str(e)}")
            raise

    async def _save_results(self, result: Dict[str, Any]) -> None:
        """Save analysis results to files"""
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save complete results as JSON
        results_file = os.path.join(output_dir, f"analysis_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # Save daily picks summary
        if "daily_picks" in result:
            picks_file = os.path.join(output_dir, f"daily_picks_{timestamp}.json")
            with open(picks_file, 'w') as f:
                json.dump(result["daily_picks"], f, indent=2, default=str)

        # Save human-readable summary
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(self._generate_detailed_summary(result))

        logger.info(f"Results saved to {output_dir}")

    def _generate_summary_report(self, result: Dict[str, Any]) -> str:
        """Generate a concise summary of the analysis"""
        if "error" in result:
            return f"Analysis failed: {result['error']}"

        analysis_summary = result.get("analysis_summary", {})
        total_analyzed = analysis_summary.get("total_analyzed", 0)
        top_picks_count = result.get("top_picks_count", 0)
        avg_score = analysis_summary.get("average_score", 0)

        return f"Analyzed {total_analyzed} stocks, generated {top_picks_count} picks with avg score {avg_score:.1f}"

    def _generate_detailed_summary(self, result: Dict[str, Any]) -> str:
        """Generate a detailed human-readable summary"""
        if "error" in result:
            return f"Analysis Failed\n================\nError: {result['error']}\n"

        summary = []
        summary.append("DAILY STOCK ANALYSIS REPORT")
        summary.append("=" * 50)
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")

        # Market overview
        daily_picks = result.get("daily_picks", {})
        if daily_picks.get("market_overview"):
            summary.append("MARKET OVERVIEW")
            summary.append("-" * 20)
            summary.append(daily_picks["market_overview"])
            summary.append("")

        # Top picks
        picks = daily_picks.get("picks", [])
        if picks:
            summary.append(f"TOP {len(picks)} STOCK PICKS")
            summary.append("-" * 30)

            for i, pick in enumerate(picks, 1):
                summary.append(f"{i}. {pick['symbol']} - ${pick['current_price']:.2f}")
                summary.append(f"   Overall Score: {pick['overall_score']:.1f}/100")
                summary.append(f"   Signal: {pick['technical_signal']}")
                summary.append(f"   Sentiment: {pick['market_sentiment']}")
                summary.append(f"   Risk Level: {pick['risk_level']}")

                # Entry strategy
                entry = pick.get("entry_strategy", {})
                if entry:
                    summary.append(f"   Entry: ${entry.get('entry_price', 0):.2f} | Stop: ${entry.get('stop_loss', 0):.2f} | Target: ${entry.get('take_profit', 0):.2f}")

                # Key insights
                insights = pick.get("key_insights", [])
                if insights:
                    summary.append(f"   Insights: {', '.join(insights[:2])}")

                summary.append("")

        # Key themes
        key_themes = daily_picks.get("key_themes", [])
        if key_themes:
            summary.append("KEY MARKET THEMES")
            summary.append("-" * 20)
            for theme in key_themes:
                summary.append(f"• {theme}")
            summary.append("")

        # Risk factors
        risk_factors = daily_picks.get("risk_factors", [])
        if risk_factors:
            summary.append("RISK FACTORS")
            summary.append("-" * 15)
            for risk in risk_factors:
                summary.append(f"• {risk}")
            summary.append("")

        # Analysis statistics
        analysis_summary = result.get("analysis_summary", {})
        if analysis_summary:
            summary.append("ANALYSIS STATISTICS")
            summary.append("-" * 20)
            summary.append(f"Total stocks analyzed: {analysis_summary.get('total_analyzed', 0)}")
            summary.append(f"Average score: {analysis_summary.get('average_score', 0):.1f}")
            summary.append(f"Stocks scoring 80+: {analysis_summary.get('picks_above_80', 0)}")
            summary.append(f"Stocks scoring 70+: {analysis_summary.get('picks_above_70', 0)}")
            summary.append(f"Stocks scoring 60+: {analysis_summary.get('picks_above_60', 0)}")

        return "\n".join(summary)

    def get_available_universes(self) -> Dict[str, Dict]:
        """Get information about available stock universes"""
        return get_universe_info()

    def print_universe_info(self) -> None:
        """Print available stock universes"""
        universes = self.get_available_universes()

        print("\nAVAILABLE STOCK UNIVERSES")
        print("=" * 40)

        for name, info in universes.items():
            print(f"{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Count: {info['count']} stocks")
            print(f"  Focus: {info['focus']}")
            print()

async def main():
    """Main application entry point"""
    app = StockAnalysisApp()

    # Print welcome message
    print("🚀 Sophisticated Stock Analysis Agent")
    print("Powered by Google Agent Development Kit")
    print("=" * 50)

    # Print available universes
    app.print_universe_info()

    # Default analysis
    print("Running daily analysis with default settings...")
    print(f"Universe: {app.config['universe']}")
    print(f"Max Picks: {app.config['max_picks']}")
    print()

    try:
        # Run the analysis
        result = await app.run_daily_analysis()

        # Print summary
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)

        daily_picks = result.get("daily_picks", {})
        picks = daily_picks.get("picks", [])

        if picks:
            print(f"\n🎯 TOP {len(picks)} STOCK PICKS:")
            print("-" * 30)

            for i, pick in enumerate(picks, 1):
                print(f"{i:2d}. {pick['symbol']:5s} - ${pick['current_price']:7.2f} "
                      f"(Score: {pick['overall_score']:5.1f}) - {pick['technical_signal']}")

            print(f"\n📊 Market Overview: {daily_picks.get('market_overview', 'N/A')}")

            print(f"\n💡 Key Insights:")
            for theme in daily_picks.get("key_themes", [])[:3]:
                print(f"   • {theme}")

            print(f"\n⚠️  Risk Factors:")
            for risk in daily_picks.get("risk_factors", [])[:3]:
                print(f"   • {risk}")

        print(f"\n📁 Detailed results saved to: {app.config['output_dir']}/")
        print("\nRun completed successfully! 🎉")

    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        logger.exception("Application failed")
        return 1

    return 0

if __name__ == "__main__":
    try:
        # Run the async main function
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        exit(1)