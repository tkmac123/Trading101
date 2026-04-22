"""Tests for Alpaca integration."""
from unittest.mock import MagicMock, patch

import pytest

from trading101.integrations.alpaca import AlpacaClient, AlpacaAccount, AlpacaPosition


@pytest.fixture
def mock_client():
    """Create a mock Alpaca client for testing."""
    with patch("trading101.integrations.alpaca.TradingClient"):
        client = AlpacaClient(api_key="test_key", api_secret="test_secret")
        client.client = MagicMock()
        return client


def test_alpaca_account_parsing(mock_client):
    """Test account snapshot parsing."""
    mock_acc = MagicMock()
    mock_acc.id = "test_account"
    mock_acc.buying_power = 50000.0
    mock_acc.cash = 25000.0
    mock_acc.portfolio_value = 100000.0
    mock_acc.last_equity = 100000.0
    mock_client.client.get_account.return_value = mock_acc

    acc = mock_client.get_account()
    assert acc is not None
    assert acc.account_id == "test_account"
    assert acc.buying_power == 50000.0
    assert acc.cash == 25000.0
    assert acc.portfolio_value == 100000.0
    assert acc.is_paper is True


def test_alpaca_position_parsing(mock_client):
    """Test position snapshot parsing."""
    mock_pos = MagicMock()
    mock_pos.symbol = "AAPL"
    mock_pos.qty = 10.0
    mock_pos.avg_fill_price = 150.0
    mock_pos.current_price = 155.0
    mock_pos.unrealized_pl = 50.0
    mock_pos.unrealized_plpc = 3.33
    mock_client.client.get_all_positions.return_value = [mock_pos]

    positions = mock_client.get_positions()
    assert positions is not None
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].side == "long"
    assert positions[0].unrealized_pl == 50.0


def test_alpaca_position_short(mock_client):
    """Test short position detection."""
    mock_pos = MagicMock()
    mock_pos.symbol = "TSLA"
    mock_pos.qty = -5.0
    mock_pos.avg_fill_price = 250.0
    mock_pos.current_price = 245.0
    mock_pos.unrealized_pl = 25.0
    mock_pos.unrealized_plpc = 2.0
    mock_client.client.get_all_positions.return_value = [mock_pos]

    positions = mock_client.get_positions()
    assert len(positions) == 1
    assert positions[0].side == "short"


def test_alpaca_account_error_handling(mock_client):
    """Test graceful error handling when account fetch fails."""
    mock_client.client.get_account.side_effect = Exception("API error")
    acc = mock_client.get_account()
    assert acc is None


def test_alpaca_positions_error_handling(mock_client):
    """Test graceful error handling when positions fetch fails."""
    mock_client.client.get_all_positions.side_effect = Exception("API error")
    positions = mock_client.get_positions()
    assert positions is None
