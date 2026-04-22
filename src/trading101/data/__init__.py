"""Data adapters: market prices, news, social sentiment."""
from .market import MarketData, get_market_data
from .news import NewsItem, fetch_news
from .social import SocialMention, fetch_social_mentions

__all__ = [
    "MarketData",
    "get_market_data",
    "NewsItem",
    "fetch_news",
    "SocialMention",
    "fetch_social_mentions",
]
