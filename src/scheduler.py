#!/usr/bin/env python3
"""
Stock Analysis Scheduler
Automated daily analysis runner with scheduling capabilities
"""

import asyncio
import schedule
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

from main import StockAnalysisApp
from utils.stock_universe import get_universe_info

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class StockAnalysisScheduler:
    """Scheduler for automated stock analysis"""

    def __init__(self):
        self.app = StockAnalysisApp()
        self.is_running = False
        self.last_run = None
        self.config = {
            "daily_time": os.getenv("DAILY_ANALYSIS_TIME", "09:30"),  # Market open time
            "weekend_analysis": os.getenv("WEEKEND_ANALYSIS", "True").lower() == "true",
            "universe": os.getenv("STOCK_UNIVERSE", "SP500_SAMPLE"),
            "max_picks": int(os.getenv("MAX_PICKS", "10")),
            "email_alerts": os.getenv("EMAIL_ALERTS", "False").lower() == "true",
            "slack_webhook": os.getenv("SLACK_WEBHOOK_URL", None)
        }

    async def run_scheduled_analysis(self, trigger_type: str = "scheduled"):
        """Run the scheduled analysis"""
        if self.is_running:
            logger.warning("Analysis already running, skipping...")
            return

        self.is_running = True
        start_time = datetime.now()

        try:
            logger.info(f"Starting {trigger_type} analysis at {start_time}")

            # Run the analysis
            result = await self.app.run_daily_analysis(
                universe=self.config["universe"],
                max_picks=self.config["max_picks"]
            )

            # Process results
            await self._process_results(result, trigger_type)

            self.last_run = start_time
            duration = datetime.now() - start_time

            logger.info(f"Analysis completed successfully in {duration}")

            # Send notifications if configured
            await self._send_notifications(result, duration)

        except Exception as e:
            logger.error(f"Scheduled analysis failed: {str(e)}")
            await self._send_error_notification(str(e))

        finally:
            self.is_running = False

    async def _process_results(self, result: dict, trigger_type: str):
        """Process and save analysis results"""
        daily_picks = result.get("daily_picks", {})
        picks = daily_picks.get("picks", [])

        if picks:
            logger.info(f"Generated {len(picks)} picks:")
            for i, pick in enumerate(picks[:5], 1):  # Log top 5
                logger.info(f"  {i}. {pick['symbol']} - ${pick['current_price']:.2f} "
                           f"(Score: {pick['overall_score']:.1f})")

        # Save additional metadata
        result["scheduler_metadata"] = {
            "trigger_type": trigger_type,
            "run_time": datetime.now().isoformat(),
            "scheduler_config": self.config
        }

    async def _send_notifications(self, result: dict, duration: timedelta):
        """Send notifications about analysis results"""
        picks = result.get("daily_picks", {}).get("picks", [])

        if not picks:
            logger.warning("No picks generated - skipping notifications")
            return

        # Prepare notification message
        message = self._format_notification_message(result, duration)

        # Send Slack notification
        if self.config["slack_webhook"]:
            await self._send_slack_notification(message)

        # Send email notification
        if self.config["email_alerts"]:
            await self._send_email_notification(message, result)

    def _format_notification_message(self, result: dict, duration: timedelta) -> str:
        """Format notification message"""
        daily_picks = result.get("daily_picks", {})
        picks = daily_picks.get("picks", [])
        analysis_summary = result.get("analysis_summary", {})

        message_lines = [
            "🚀 Daily Stock Analysis Complete",
            f"⏰ Duration: {duration}",
            f"📊 Analyzed: {analysis_summary.get('total_analyzed', 0)} stocks",
            f"🎯 Top Picks: {len(picks)}",
            f"📈 Avg Score: {analysis_summary.get('average_score', 0):.1f}",
            "",
            "🔥 Top 5 Picks:"
        ]

        for i, pick in enumerate(picks[:5], 1):
            message_lines.append(
                f"{i}. {pick['symbol']} - ${pick['current_price']:.2f} "
                f"(Score: {pick['overall_score']:.1f}) - {pick['technical_signal']}"
            )

        market_overview = daily_picks.get("market_overview", "")
        if market_overview:
            message_lines.extend(["", f"📰 Market: {market_overview}"])

        return "\n".join(message_lines)

    async def _send_slack_notification(self, message: str):
        """Send Slack notification"""
        try:
            import requests
            import json

            webhook_url = self.config["slack_webhook"]
            payload = {
                "text": message,
                "username": "Stock Analysis Bot",
                "icon_emoji": ":chart_with_upwards_trend:"
            }

            response = requests.post(webhook_url, json=payload)
            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
            else:
                logger.error(f"Failed to send Slack notification: {response.status_code}")

        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")

    async def _send_email_notification(self, message: str, result: dict):
        """Send email notification"""
        try:
            # Email implementation would go here
            # For now, just log that we would send an email
            logger.info("Email notification would be sent here")
            # In production, integrate with SendGrid, AWS SES, or similar service

        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")

    async def _send_error_notification(self, error_message: str):
        """Send error notification"""
        error_msg = f"❌ Stock Analysis Failed\n\nError: {error_message}\nTime: {datetime.now()}"

        if self.config["slack_webhook"]:
            await self._send_slack_notification(error_msg)

        logger.error("Error notification sent")

    def schedule_daily_analysis(self):
        """Schedule daily analysis"""
        analysis_time = self.config["daily_time"]

        # Schedule for weekdays
        schedule.every().monday.at(analysis_time).do(self._run_async_analysis, "daily_weekday")
        schedule.every().tuesday.at(analysis_time).do(self._run_async_analysis, "daily_weekday")
        schedule.every().wednesday.at(analysis_time).do(self._run_async_analysis, "daily_weekday")
        schedule.every().thursday.at(analysis_time).do(self._run_async_analysis, "daily_weekday")
        schedule.every().friday.at(analysis_time).do(self._run_async_analysis, "daily_weekday")

        # Weekend analysis if enabled
        if self.config["weekend_analysis"]:
            schedule.every().saturday.at("10:00").do(self._run_async_analysis, "weekend")
            schedule.every().sunday.at("10:00").do(self._run_async_analysis, "weekend")

        logger.info(f"Scheduled daily analysis at {analysis_time}")
        if self.config["weekend_analysis"]:
            logger.info("Weekend analysis enabled")

    def schedule_intraday_analysis(self):
        """Schedule intraday analysis during market hours"""
        # Market hours analysis (every 4 hours during trading)
        schedule.every().day.at("09:30").do(self._run_async_analysis, "market_open")
        schedule.every().day.at("13:30").do(self._run_async_analysis, "midday")
        schedule.every().day.at("15:30").do(self._run_async_analysis, "market_close")

        logger.info("Scheduled intraday analysis")

    def _run_async_analysis(self, trigger_type: str):
        """Wrapper to run async analysis from sync scheduler"""
        # Check if it's a trading day for weekday analyses
        if trigger_type == "daily_weekday" and not self._is_trading_day():
            logger.info("Skipping analysis - market closed (holiday)")
            return

        # Run async analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_scheduled_analysis(trigger_type))
        finally:
            loop.close()

    def _is_trading_day(self) -> bool:
        """Check if today is a trading day (simplified)"""
        # In production, would check against market calendar API
        today = datetime.now()
        return today.weekday() < 5  # Monday = 0, Friday = 4

    def run_manual_analysis(self, universe: Optional[str] = None):
        """Run manual analysis immediately"""
        logger.info("Running manual analysis...")

        universe = universe or self.config["universe"]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_scheduled_analysis("manual"))
        finally:
            loop.close()

    def get_schedule_status(self) -> dict:
        """Get current schedule status"""
        return {
            "is_running": self.is_running,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_runs": [str(job) for job in schedule.jobs],
            "config": self.config
        }

    def start_scheduler(self):
        """Start the scheduler in a loop"""
        logger.info("Starting Stock Analysis Scheduler...")

        # Set up schedules
        self.schedule_daily_analysis()

        # Optional: add intraday analysis
        if os.getenv("INTRADAY_ANALYSIS", "False").lower() == "true":
            self.schedule_intraday_analysis()

        logger.info(f"Scheduler started with {len(schedule.jobs)} jobs")

        # Run scheduler loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")

def main():
    """Main scheduler entry point"""
    scheduler = StockAnalysisScheduler()

    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "run":
            # Run manual analysis
            universe = sys.argv[2] if len(sys.argv) > 2 else None
            scheduler.run_manual_analysis(universe)

        elif command == "schedule":
            # Start scheduler
            scheduler.start_scheduler()

        elif command == "status":
            # Show status
            status = scheduler.get_schedule_status()
            print("\nScheduler Status:")
            print("=" * 30)
            print(f"Running: {status['is_running']}")
            print(f"Last Run: {status['last_run']}")
            print(f"Scheduled Jobs: {len(status['next_runs'])}")
            for job in status['next_runs']:
                print(f"  - {job}")

        else:
            print("Usage:")
            print("  python scheduler.py run [universe]    - Run manual analysis")
            print("  python scheduler.py schedule          - Start scheduler")
            print("  python scheduler.py status            - Show status")

    else:
        print("Stock Analysis Scheduler")
        print("========================")
        print()
        print("Commands:")
        print("  run [universe]  - Run analysis immediately")
        print("  schedule        - Start automated scheduler")
        print("  status          - Show current status")
        print()
        print("Environment Variables:")
        print("  DAILY_ANALYSIS_TIME  - Time for daily analysis (default: 09:30)")
        print("  WEEKEND_ANALYSIS     - Enable weekend analysis (default: True)")
        print("  STOCK_UNIVERSE       - Stock universe to analyze (default: SP500_SAMPLE)")
        print("  MAX_PICKS           - Maximum picks to generate (default: 10)")
        print("  SLACK_WEBHOOK_URL   - Slack webhook for notifications")
        print("  EMAIL_ALERTS        - Enable email alerts (default: False)")

if __name__ == "__main__":
    main()