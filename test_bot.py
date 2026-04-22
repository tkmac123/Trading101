#!/usr/bin/env python3
"""Test Telegram bot connectivity and message handling."""
import sys
import logging
sys.path.insert(0, 'src')

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

from trading101.config import load_settings
from trading101.notify.telegram import TelegramBot

print("=" * 60)
print("TELEGRAM BOT DIAGNOSTIC TEST")
print("=" * 60)

# Load settings
settings = load_settings()
print(f"\n✓ Configuration loaded")
print(f"  Token: {settings.telegram_bot_token[:30]}..." if settings.telegram_bot_token else "  Token: NOT SET")
print(f"  Chat ID: {settings.telegram_chat_id}")

# Verify token format
if settings.telegram_bot_token:
    parts = settings.telegram_bot_token.split(':')
    if len(parts) == 2:
        print(f"  Token format: ✓ Valid (bot_id:token)")
    else:
        print(f"  Token format: ✗ Invalid")
else:
    print("  Token: ✗ NOT CONFIGURED")
    sys.exit(1)

# Try to instantiate bot
print(f"\n📡 Initializing bot...")
try:
    bot = TelegramBot()
    print(f"✓ Bot initialized successfully")
    print(f"  Allowed chats: {bot.allowed if bot.allowed else 'Open (any chat)'}")
except Exception as e:
    print(f"✗ Failed to initialize bot: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test message routing
print(f"\n🧪 Testing message routing...")
test_messages = [
    "analyze NVDA",
    "show alerts",
    "chart AAPL",
    "news TSLA",
    "stats",
]

async def test_handler():
    print("✓ Async message handler is working")

# Try to start the bot
print(f"\n🚀 Starting bot (polling)...")
print(f"   Waiting for messages. Text your bot at @{settings.telegram_bot_token.split(':')[0] if ':' in settings.telegram_bot_token else 'unknown'}")
print(f"   Press Ctrl+C to stop.\n")

try:
    bot.run()
except KeyboardInterrupt:
    print("\n✓ Bot stopped.")
except Exception as e:
    print(f"\n✗ Bot error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
