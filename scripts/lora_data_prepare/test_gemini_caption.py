"""Tests for Gemini caption service configuration."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lora_data_prepare import gemini_caption


class TestGeminiServiceConfig(unittest.TestCase):
    """Verify Gemini service setup paths."""

    def setUp(self) -> None:
        gemini_caption._gemini_service = None

    def tearDown(self) -> None:
        gemini_caption._gemini_service = None

    def test_get_gemini_service_uses_environment_key(self) -> None:
        """Create the service from GEMINI_API_KEY when no explicit key is passed."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            service = gemini_caption.get_gemini_service()

        self.assertEqual(service.api_key, "test-key")

    def test_get_gemini_service_raises_without_any_key(self) -> None:
        """Reject service creation when no explicit or environment key is available."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                gemini_caption.get_gemini_service()

    def test_transcribe_audio_returns_plain_text(self) -> None:
        """Return plain transcription text from a successful response payload."""
        service = gemini_caption.GeminiService(api_key="test-key")
        fake_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "hello world\nthis is a song"
                            }
                        ]
                    }
                }
            ]
        }

        with patch.object(service, "generate_content", return_value=fake_response):
            result = service.transcribe_audio("sample.wav", use_upload=False)

        self.assertEqual(result, "hello world\nthis is a song")

    def test_get_endpoint_accepts_prefixed_model_name(self) -> None:
        """Normalize model names returned from the ListModels API."""
        service = gemini_caption.GeminiService(api_key="test-key")

        endpoint = service._get_endpoint("generate", "models/gemini-2.5-flash")

        self.assertTrue(endpoint.endswith("/v1beta/models/gemini-2.5-flash:generateContent"))

    def test_generate_content_retries_after_quota_response(self) -> None:
        """Retry quota-limited requests using the suggested delay."""
        service = gemini_caption.GeminiService(api_key="test-key")

        quota_response = MagicMock()
        quota_response.status_code = 429
        quota_response.headers = {}
        quota_response.json.return_value = {
            "error": {"message": "Please retry in 1.5s."}
        }
        quota_response.text = '{"error":{"message":"Please retry in 1.5s."}}'

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {"candidates": []}

        with patch("scripts.lora_data_prepare.gemini_caption.requests.post", side_effect=[quota_response, ok_response]) as mock_post:
            with patch("scripts.lora_data_prepare.gemini_caption.time.sleep") as mock_sleep:
                result = service.generate_content("hello", model_name="models/gemini-2.5-flash")

        self.assertEqual(result, {"candidates": []})
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(1.5)


if __name__ == "__main__":
    unittest.main()