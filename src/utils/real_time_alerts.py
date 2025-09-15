import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging
import json
import os
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    symbol: str
    alert_type: str
    condition: str
    target_value: float
    current_value: float
    triggered_at: datetime
    message: str
    urgency: str  # LOW, MEDIUM, HIGH
    metadata: Dict[str, Any]

class AlertCondition:
    """Base class for alert conditions"""

    def __init__(self, symbol: str, condition_type: str):
        self.symbol = symbol
        self.condition_type = condition_type

    def check(self, data: Dict[str, Any]) -> Optional[Alert]:
        """Check if condition is met"""
        raise NotImplementedError

class PriceAlert(AlertCondition):
    """Price-based alerts"""

    def __init__(self, symbol: str, target_price: float, direction: str):
        super().__init__(symbol, "PRICE")
        self.target_price = target_price
        self.direction = direction  # "ABOVE" or "BELOW"

    def check(self, data: Dict[str, Any]) -> Optional[Alert]:
        current_price = data.get("current_price", 0)

        if self.direction == "ABOVE" and current_price >= self.target_price:
            return Alert(
                id=f"{self.symbol}_PRICE_{datetime.now().timestamp()}",
                symbol=self.symbol,
                alert_type="PRICE_BREAKOUT",
                condition=f"Price above ${self.target_price:.2f}",
                target_value=self.target_price,
                current_value=current_price,
                triggered_at=datetime.now(),
                message=f"{self.symbol} broke above ${self.target_price:.2f}, now at ${current_price:.2f}",
                urgency="HIGH",
                metadata={"direction": self.direction, "breakout_percentage": ((current_price - self.target_price) / self.target_price) * 100}
            )

        elif self.direction == "BELOW" and current_price <= self.target_price:
            return Alert(
                id=f"{self.symbol}_PRICE_{datetime.now().timestamp()}",
                symbol=self.symbol,
                alert_type="PRICE_BREAKDOWN",
                condition=f"Price below ${self.target_price:.2f}",
                target_value=self.target_price,
                current_value=current_price,
                triggered_at=datetime.now(),
                message=f"{self.symbol} broke below ${self.target_price:.2f}, now at ${current_price:.2f}",
                urgency="HIGH",
                metadata={"direction": self.direction, "breakdown_percentage": ((self.target_price - current_price) / self.target_price) * 100}
            )

        return None

class VolumeAlert(AlertCondition):
    """Volume-based alerts"""

    def __init__(self, symbol: str, volume_multiplier: float):
        super().__init__(symbol, "VOLUME")
        self.volume_multiplier = volume_multiplier

    def check(self, data: Dict[str, Any]) -> Optional[Alert]:
        current_volume = data.get("current_volume", 0)
        avg_volume = data.get("avg_volume", 1)

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        if volume_ratio >= self.volume_multiplier:
            urgency = "HIGH" if volume_ratio >= 3 else "MEDIUM"

            return Alert(
                id=f"{self.symbol}_VOLUME_{datetime.now().timestamp()}",
                symbol=self.symbol,
                alert_type="VOLUME_SURGE",
                condition=f"Volume {self.volume_multiplier}x above average",
                target_value=self.volume_multiplier,
                current_value=volume_ratio,
                triggered_at=datetime.now(),
                message=f"{self.symbol} volume surge: {volume_ratio:.1f}x average ({current_volume:,.0f} vs {avg_volume:,.0f})",
                urgency=urgency,
                metadata={"volume_ratio": volume_ratio, "current_volume": current_volume, "avg_volume": avg_volume}
            )

        return None

class MomentumAlert(AlertCondition):
    """Momentum-based alerts"""

    def __init__(self, symbol: str, momentum_threshold: float, timeframe: str):
        super().__init__(symbol, "MOMENTUM")
        self.momentum_threshold = momentum_threshold
        self.timeframe = timeframe

    def check(self, data: Dict[str, Any]) -> Optional[Alert]:
        momentum = data.get(f"momentum_{self.timeframe}", 0)

        if abs(momentum) >= self.momentum_threshold:
            direction = "UP" if momentum > 0 else "DOWN"
            urgency = "HIGH" if abs(momentum) >= self.momentum_threshold * 1.5 else "MEDIUM"

            return Alert(
                id=f"{self.symbol}_MOMENTUM_{datetime.now().timestamp()}",
                symbol=self.symbol,
                alert_type="MOMENTUM_BREAKOUT",
                condition=f"{direction} momentum >= {self.momentum_threshold}%",
                target_value=self.momentum_threshold,
                current_value=abs(momentum),
                triggered_at=datetime.now(),
                message=f"{self.symbol} strong {direction.lower()} momentum: {momentum:+.2f}% in {self.timeframe}",
                urgency=urgency,
                metadata={"momentum": momentum, "direction": direction, "timeframe": self.timeframe}
            )

        return None

class GapAlert(AlertCondition):
    """Gap-based alerts"""

    def __init__(self, symbol: str, gap_threshold: float):
        super().__init__(symbol, "GAP")
        self.gap_threshold = gap_threshold

    def check(self, data: Dict[str, Any]) -> Optional[Alert]:
        gap_percent = data.get("gap_percent", 0)

        if abs(gap_percent) >= self.gap_threshold:
            direction = "UP" if gap_percent > 0 else "DOWN"
            urgency = "HIGH" if abs(gap_percent) >= self.gap_threshold * 2 else "MEDIUM"

            return Alert(
                id=f"{self.symbol}_GAP_{datetime.now().timestamp()}",
                symbol=self.symbol,
                alert_type="GAP_ALERT",
                condition=f"{direction} gap >= {self.gap_threshold}%",
                target_value=self.gap_threshold,
                current_value=abs(gap_percent),
                triggered_at=datetime.now(),
                message=f"{self.symbol} significant {direction.lower()} gap: {gap_percent:+.2f}%",
                urgency=urgency,
                metadata={"gap_percent": gap_percent, "direction": direction}
            )

        return None

class ProfitTargetAlert(AlertCondition):
    """Profit target alerts for active positions"""

    def __init__(self, symbol: str, entry_price: float, target_percentage: float):
        super().__init__(symbol, "PROFIT_TARGET")
        self.entry_price = entry_price
        self.target_percentage = target_percentage
        self.target_price = entry_price * (1 + target_percentage / 100)

    def check(self, data: Dict[str, Any]) -> Optional[Alert]:
        current_price = data.get("current_price", 0)

        profit_percent = ((current_price - self.entry_price) / self.entry_price) * 100

        if profit_percent >= self.target_percentage:
            return Alert(
                id=f"{self.symbol}_PROFIT_{datetime.now().timestamp()}",
                symbol=self.symbol,
                alert_type="PROFIT_TARGET",
                condition=f"Profit target {self.target_percentage}% reached",
                target_value=self.target_percentage,
                current_value=profit_percent,
                triggered_at=datetime.now(),
                message=f"{self.symbol} profit target reached: {profit_percent:.2f}% (entry: ${self.entry_price:.2f}, current: ${current_price:.2f})",
                urgency="HIGH",
                metadata={
                    "entry_price": self.entry_price,
                    "current_price": current_price,
                    "profit_percent": profit_percent,
                    "target_price": self.target_price
                }
            )

        return None

class StopLossAlert(AlertCondition):
    """Stop loss alerts for risk management"""

    def __init__(self, symbol: str, entry_price: float, stop_percentage: float):
        super().__init__(symbol, "STOP_LOSS")
        self.entry_price = entry_price
        self.stop_percentage = stop_percentage
        self.stop_price = entry_price * (1 - stop_percentage / 100)

    def check(self, data: Dict[str, Any]) -> Optional[Alert]:
        current_price = data.get("current_price", 0)

        loss_percent = ((self.entry_price - current_price) / self.entry_price) * 100

        if current_price <= self.stop_price:
            return Alert(
                id=f"{self.symbol}_STOP_{datetime.now().timestamp()}",
                symbol=self.symbol,
                alert_type="STOP_LOSS",
                condition=f"Stop loss triggered at ${self.stop_price:.2f}",
                target_value=self.stop_price,
                current_value=current_price,
                triggered_at=datetime.now(),
                message=f"{self.symbol} STOP LOSS: -{loss_percent:.2f}% (entry: ${self.entry_price:.2f}, current: ${current_price:.2f})",
                urgency="HIGH",
                metadata={
                    "entry_price": self.entry_price,
                    "current_price": current_price,
                    "loss_percent": loss_percent,
                    "stop_price": self.stop_price
                }
            )

        return None

class RealTimeAlertManager:
    """Manages real-time alerts for day trading"""

    def __init__(self, notification_channels: List[str] = None):
        self.alert_conditions: Dict[str, List[AlertCondition]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels = notification_channels or ["console"]
        self.is_running = False
        self.monitoring_task = None

        # Load configuration
        self.config = self._load_config()

        # Notification handlers
        self.notification_handlers = {
            "console": self._console_notification,
            "slack": self._slack_notification,
            "email": self._email_notification,
            "webhook": self._webhook_notification
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load alert configuration"""
        return {
            "monitoring_interval": 30,  # seconds
            "max_alerts_per_symbol": 5,
            "alert_cooldown": 300,  # 5 minutes
            "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
            "email_config": {
                "enabled": os.getenv("EMAIL_ALERTS", "False").lower() == "true",
                "smtp_server": os.getenv("SMTP_SERVER"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "username": os.getenv("EMAIL_USERNAME"),
                "password": os.getenv("EMAIL_PASSWORD"),
                "recipients": os.getenv("ALERT_RECIPIENTS", "").split(",")
            }
        }

    def add_day_trading_alerts(self, symbol: str, entry_price: float, target_percentage: float = 10.0, stop_percentage: float = 2.0):
        """Add comprehensive day trading alerts for a position"""
        if symbol not in self.alert_conditions:
            self.alert_conditions[symbol] = []

        # Profit target alert
        profit_alert = ProfitTargetAlert(symbol, entry_price, target_percentage)
        self.alert_conditions[symbol].append(profit_alert)

        # Stop loss alert
        stop_alert = StopLossAlert(symbol, entry_price, stop_percentage)
        self.alert_conditions[symbol].append(stop_alert)

        # Volume surge alert
        volume_alert = VolumeAlert(symbol, 2.0)  # 2x volume
        self.alert_conditions[symbol].append(volume_alert)

        # Momentum alerts
        momentum_5m = MomentumAlert(symbol, 3.0, "5m")  # 3% in 5 minutes
        momentum_15m = MomentumAlert(symbol, 5.0, "15m")  # 5% in 15 minutes
        self.alert_conditions[symbol].extend([momentum_5m, momentum_15m])

        logger.info(f"Added day trading alerts for {symbol} (entry: ${entry_price:.2f})")

    def add_breakout_alerts(self, symbol: str, resistance_level: float, support_level: float):
        """Add breakout alerts for key levels"""
        if symbol not in self.alert_conditions:
            self.alert_conditions[symbol] = []

        # Resistance breakout
        resistance_alert = PriceAlert(symbol, resistance_level, "ABOVE")
        self.alert_conditions[symbol].append(resistance_alert)

        # Support breakdown
        support_alert = PriceAlert(symbol, support_level, "BELOW")
        self.alert_conditions[symbol].append(support_alert)

        logger.info(f"Added breakout alerts for {symbol} (resistance: ${resistance_level:.2f}, support: ${support_level:.2f})")

    def add_gap_alerts(self, symbols: List[str], gap_threshold: float = 2.0):
        """Add gap alerts for multiple symbols"""
        for symbol in symbols:
            if symbol not in self.alert_conditions:
                self.alert_conditions[symbol] = []

            gap_alert = GapAlert(symbol, gap_threshold)
            self.alert_conditions[symbol].append(gap_alert)

        logger.info(f"Added gap alerts for {len(symbols)} symbols (threshold: {gap_threshold}%)")

    async def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_running:
            logger.warning("Alert monitoring is already running")
            return

        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started real-time alert monitoring")

    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped real-time alert monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                await self._check_all_alerts()
                await asyncio.sleep(self.config["monitoring_interval"])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(10)  # Wait before retrying

    async def _check_all_alerts(self):
        """Check all alert conditions"""
        for symbol, conditions in self.alert_conditions.items():
            try:
                # Get current data for symbol
                current_data = await self._get_current_data(symbol)

                if not current_data:
                    continue

                # Check each condition
                for condition in conditions:
                    alert = condition.check(current_data)
                    if alert:
                        await self._process_alert(alert)

            except Exception as e:
                logger.error(f"Error checking alerts for {symbol}: {str(e)}")

    async def _get_current_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)

            # Get current price and volume
            hist_1d = ticker.history(period="1d", interval="1m")
            if hist_1d.empty:
                return None

            current_price = hist_1d['Close'].iloc[-1]
            current_volume = hist_1d['Volume'].iloc[-1]

            # Get average volume
            hist_5d = ticker.history(period="5d", interval="1d")
            avg_volume = hist_5d['Volume'].mean() if not hist_5d.empty else current_volume

            # Calculate momentum for different timeframes
            momentum_5m = 0
            momentum_15m = 0

            if len(hist_1d) >= 5:
                price_5m_ago = hist_1d['Close'].iloc[-5]
                momentum_5m = ((current_price - price_5m_ago) / price_5m_ago) * 100

            if len(hist_1d) >= 15:
                price_15m_ago = hist_1d['Close'].iloc[-15]
                momentum_15m = ((current_price - price_15m_ago) / price_15m_ago) * 100

            # Calculate gap (if market just opened)
            gap_percent = 0
            if len(hist_5d) >= 2:
                yesterday_close = hist_5d['Close'].iloc[-2]
                today_open = hist_5d['Open'].iloc[-1]
                gap_percent = ((today_open - yesterday_close) / yesterday_close) * 100

            return {
                "symbol": symbol,
                "current_price": current_price,
                "current_volume": current_volume,
                "avg_volume": avg_volume,
                "momentum_5m": momentum_5m,
                "momentum_15m": momentum_15m,
                "gap_percent": gap_percent,
                "timestamp": datetime.now()
            }

        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {str(e)}")
            return None

    async def _process_alert(self, alert: Alert):
        """Process a triggered alert"""
        # Check for duplicate alerts (cooldown)
        alert_key = f"{alert.symbol}_{alert.alert_type}"

        if alert_key in self.active_alerts:
            last_alert_time = self.active_alerts[alert_key].triggered_at
            cooldown_period = timedelta(seconds=self.config["alert_cooldown"])

            if datetime.now() - last_alert_time < cooldown_period:
                return  # Skip duplicate alert

        # Store alert
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)

        # Send notifications
        await self._send_notifications(alert)

        logger.info(f"Alert triggered: {alert.message}")

    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        for channel in self.notification_channels:
            try:
                handler = self.notification_handlers.get(channel)
                if handler:
                    await handler(alert)
            except Exception as e:
                logger.error(f"Error sending {channel} notification: {str(e)}")

    async def _console_notification(self, alert: Alert):
        """Send console notification"""
        urgency_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        emoji = urgency_emoji.get(alert.urgency, "⚪")

        print(f"\n{emoji} ALERT [{alert.urgency}] - {alert.triggered_at.strftime('%H:%M:%S')}")
        print(f"Symbol: {alert.symbol}")
        print(f"Type: {alert.alert_type}")
        print(f"Message: {alert.message}")
        print("-" * 50)

    async def _slack_notification(self, alert: Alert):
        """Send Slack notification"""
        webhook_url = self.config.get("slack_webhook")
        if not webhook_url:
            return

        urgency_colors = {"LOW": "good", "MEDIUM": "warning", "HIGH": "danger"}
        color = urgency_colors.get(alert.urgency, "warning")

        payload = {
            "attachments": [{
                "color": color,
                "title": f"🚨 {alert.symbol} Alert - {alert.alert_type}",
                "text": alert.message,
                "fields": [
                    {"title": "Urgency", "value": alert.urgency, "short": True},
                    {"title": "Time", "value": alert.triggered_at.strftime('%H:%M:%S'), "short": True},
                    {"title": "Condition", "value": alert.condition, "short": False}
                ],
                "timestamp": alert.triggered_at.timestamp()
            }]
        }

        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Slack notification failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Slack notification error: {str(e)}")

    async def _email_notification(self, alert: Alert):
        """Send email notification"""
        email_config = self.config.get("email_config", {})
        if not email_config.get("enabled"):
            return

        # Email implementation would go here
        # For now, just log that we would send an email
        logger.info(f"Would send email notification for {alert.symbol} alert")

    async def _webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        if not webhook_url:
            return

        payload = {
            "alert": {
                "id": alert.id,
                "symbol": alert.symbol,
                "type": alert.alert_type,
                "message": alert.message,
                "urgency": alert.urgency,
                "triggered_at": alert.triggered_at.isoformat(),
                "metadata": alert.metadata
            }
        }

        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Webhook notification failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Webhook notification error: {str(e)}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.triggered_at >= cutoff_time]

    def clear_alerts(self, symbol: str = None):
        """Clear alerts for a symbol or all symbols"""
        if symbol:
            # Remove conditions for specific symbol
            if symbol in self.alert_conditions:
                del self.alert_conditions[symbol]

            # Remove active alerts for symbol
            keys_to_remove = [key for key in self.active_alerts.keys() if key.startswith(symbol)]
            for key in keys_to_remove:
                del self.active_alerts[key]

            logger.info(f"Cleared alerts for {symbol}")
        else:
            # Clear all alerts
            self.alert_conditions.clear()
            self.active_alerts.clear()
            logger.info("Cleared all alerts")

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.alert_history)
        alerts_24h = len(self.get_alert_history(24))

        # Alert types breakdown
        alert_types = {}
        for alert in self.alert_history:
            alert_types[alert.alert_type] = alert_types.get(alert.alert_type, 0) + 1

        # Symbols with most alerts
        symbol_counts = {}
        for alert in self.alert_history:
            symbol_counts[alert.symbol] = symbol_counts.get(alert.symbol, 0) + 1

        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_alerts": total_alerts,
            "alerts_24h": alerts_24h,
            "active_alerts": len(self.active_alerts),
            "monitored_symbols": len(self.alert_conditions),
            "alert_types": alert_types,
            "top_symbols": dict(top_symbols),
            "is_monitoring": self.is_running
        }

# Convenience functions for quick setup
async def setup_day_trading_alerts(symbols_with_entries: Dict[str, float], alert_manager: RealTimeAlertManager):
    """Quick setup for day trading alerts"""
    for symbol, entry_price in symbols_with_entries.items():
        alert_manager.add_day_trading_alerts(symbol, entry_price)

    if not alert_manager.is_running:
        await alert_manager.start_monitoring()

async def setup_breakout_monitoring(breakout_setups: Dict[str, Dict[str, float]], alert_manager: RealTimeAlertManager):
    """Quick setup for breakout monitoring"""
    for symbol, levels in breakout_setups.items():
        resistance = levels.get("resistance")
        support = levels.get("support")
        if resistance and support:
            alert_manager.add_breakout_alerts(symbol, resistance, support)

    if not alert_manager.is_running:
        await alert_manager.start_monitoring()