#!/usr/bin/env python3
"""
Simple Test Script for Day Trading Agent
Tests core functionality without ADK dependencies
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_basic_functionality():
    """Test basic components without ADK"""
    print("🎯 Testing Day Trading Agent - Basic Functionality")
    print("=" * 60)

    # Test 1: Environment setup
    print("\n1. Testing Environment Setup...")
    gemini_key = os.getenv('GOOGLE_API_KEY', 'not_set')
    print(f"   ✅ Gemini API Key: {'✓ Set' if gemini_key != 'not_set' and gemini_key != 'your_google_api_key_here' else '❌ Not configured'}")

    # Test 2: Data collection
    print("\n2. Testing Stock Data Collection...")
    try:
        # Test with a popular stock
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1d", interval="5m")

        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            print(f"   ✅ AAPL Data: ${current_price:.2f}, Volume: {volume:,.0f}")

            # Test intraday analysis
            if len(hist) >= 12:  # At least 1 hour of data
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(78) * 100  # Intraday vol
                momentum_1h = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-12]) / hist['Close'].iloc[-12]) * 100

                print(f"   ✅ Technical Analysis:")
                print(f"      • Intraday Volatility: {volatility:.1f}%")
                print(f"      • 1-Hour Momentum: {momentum_1h:+.2f}%")

                # VWAP calculation
                typical_price = (hist['High'] + hist['Low'] + hist['Close']) / 3
                vwap = (typical_price * hist['Volume']).cumsum() / hist['Volume'].cumsum()
                current_vwap = vwap.iloc[-1]
                vwap_position = "Above" if current_price > current_vwap else "Below"

                print(f"      • VWAP: ${current_vwap:.2f} ({vwap_position})")

                # Day trading signals
                signals = []
                if volatility > 30:
                    signals.append("High volatility environment")
                if abs(momentum_1h) > 2:
                    signals.append(f"Strong {'bullish' if momentum_1h > 0 else 'bearish'} momentum")
                if current_price > current_vwap * 1.005:
                    signals.append("Above VWAP with momentum")

                if signals:
                    print(f"   🎯 Day Trading Signals:")
                    for signal in signals:
                        print(f"      • {signal}")
                else:
                    print(f"   ⚠️  No strong signals detected")

        else:
            print("   ❌ Failed to fetch data")

    except Exception as e:
        print(f"   ❌ Data collection error: {str(e)}")

    # Test 3: Technical indicators
    print("\n3. Testing Technical Indicators...")
    try:
        symbols = ["TSLA", "NVDA", "AAPL"]
        day_trading_candidates = []

        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="5m")

            if not data.empty and len(data) >= 20:
                current_price = data['Close'].iloc[-1]

                # RSI calculation
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]

                # Volume analysis
                avg_volume = data['Volume'].mean()
                current_volume = data['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

                # Calculate score
                score = 50  # Base score
                if 30 <= current_rsi <= 70:
                    score += 20  # Good RSI range
                if volume_ratio > 1.5:
                    score += 15  # High volume
                if len(data) >= 12:
                    momentum = ((current_price - data['Close'].iloc[-12]) / data['Close'].iloc[-12]) * 100
                    if abs(momentum) > 2:
                        score += 15  # Strong momentum

                print(f"   📊 {symbol}: ${current_price:.2f} (Score: {score:.0f}/100)")
                print(f"      RSI: {current_rsi:.1f}, Volume: {volume_ratio:.1f}x")

                if score >= 70:
                    day_trading_candidates.append({
                        'symbol': symbol,
                        'price': current_price,
                        'score': score,
                        'rsi': current_rsi,
                        'volume_ratio': volume_ratio
                    })

    except Exception as e:
        print(f"   ❌ Technical analysis error: {str(e)}")

    # Test 4: Day trading recommendations
    print("\n4. Day Trading Recommendations...")
    if day_trading_candidates:
        print(f"   🎯 Found {len(day_trading_candidates)} day trading candidates:")

        for i, candidate in enumerate(day_trading_candidates, 1):
            symbol = candidate['symbol']
            price = candidate['price']
            score = candidate['score']

            # Calculate 10% profit target
            target_price = price * 1.10
            stop_loss = price * 0.98  # 2% stop
            risk_reward = (target_price - price) / (price - stop_loss)

            print(f"\n   {i}. {symbol} - ${price:.2f} (Score: {score:.0f}/100)")
            print(f"      🎯 Target: ${target_price:.2f} (+10%)")
            print(f"      🛑 Stop: ${stop_loss:.2f} (-2%)")
            print(f"      📊 Risk/Reward: 1:{risk_reward:.1f}")

            # Time estimate
            if candidate['volume_ratio'] > 2 and score > 80:
                time_est = "1-3 hours"
            elif score > 75:
                time_est = "2-4 hours"
            else:
                time_est = "4-6 hours"

            print(f"      ⏱️  Estimated time: {time_est}")

    else:
        print("   ⚠️  No strong day trading candidates found")
        print("      Market may be in low volatility or ranging conditions")

    # Test 5: Market session analysis
    print("\n5. Market Session Analysis...")
    current_hour = datetime.now().hour

    if 9 <= current_hour < 10:
        session = "Opening Hour"
        strategy = "Focus on gap trading and momentum breakouts"
    elif 10 <= current_hour < 14:
        session = "Midday Session"
        strategy = "Look for range trading and VWAP bounces"
    elif 14 <= current_hour < 16:
        session = "Power Hour"
        strategy = "Quick scalps, avoid new swing positions"
    else:
        session = "After Hours"
        strategy = "Limited opportunities, plan for tomorrow"

    print(f"   🕐 Current Session: {session}")
    print(f"   💡 Strategy: {strategy}")

    print("\n" + "=" * 60)
    print("✅ Basic functionality test completed!")
    print("\n💡 Next Steps:")
    print("   1. Set your Gemini API key in .env file")
    print("   2. Run during market hours for best results")
    print("   3. Test with different stock universes")
    print("   4. Set up Slack webhooks for alerts")

    return True

def test_profit_targets():
    """Test 10% profit target calculations"""
    print("\n🎯 Testing 10% Profit Target System...")

    test_stocks = [
        ("TSLA", 250.00),
        ("NVDA", 450.00),
        ("AAPL", 180.00)
    ]

    for symbol, price in test_stocks:
        target = price * 1.10  # 10% profit
        stop = price * 0.98    # 2% stop loss
        risk = price - stop
        reward = target - price
        rr_ratio = reward / risk if risk > 0 else 0

        print(f"   📊 {symbol} @ ${price:.2f}")
        print(f"      Target: ${target:.2f} | Stop: ${stop:.2f} | R/R: 1:{rr_ratio:.1f}")

if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        test_profit_targets()

        print("\n🎉 All tests completed!")
        print("\nTo run the full system:")
        print("   python run_day_trading_demo.py")

    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()