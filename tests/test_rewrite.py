from unittest.mock import MagicMock, patch

from api.rewrite import rewrite_paragraph, target_direction


def test_target_direction_above_mean():
    assert target_direction(1.5, 0.0) == "simpler and easier to read"


def test_target_direction_below_mean():
    assert target_direction(-0.5, 0.0) == "more detailed and intellectually rich"


def test_rewrite_paragraph_returns_llm_output():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "  Rewritten text.  "

    with patch("api.rewrite._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = rewrite_paragraph("Original.", "simpler and easier to read", "Too complex at 0.1.")

    assert result == "Rewritten text."


def test_rewrite_paragraph_returns_fallback_on_error():
    with patch("api.rewrite._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("timeout")
        mock_get_client.return_value = mock_client

        result = rewrite_paragraph("Original.", "simpler", "reason")

    assert result == "[Rewrite unavailable]"
