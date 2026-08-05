"""Tests for LoRA/LoKr lifecycle loading behavior."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from acestep.core.generation.handler.lora import lifecycle


class AdapterNameTest(unittest.TestCase):
    def test_decimal_checkpoint_name_is_pytorch_safe(self):
        self.assertEqual(
            lifecycle._default_adapter_name_from_path("/tmp/epoch_25_loss_0.8750"),
            "epoch_25_loss_0_8750",
        )

    def test_empty_checkpoint_name_uses_default(self):
        self.assertEqual(lifecycle._default_adapter_name_from_path("/"), "default")


class _DummyDecoder:
    """Minimal decoder stub for lifecycle loader tests."""

    def __init__(self) -> None:
        self._weights = {"w": torch.zeros(1)}

    def state_dict(self):
        """Return a tiny state dict suitable for backup/restore paths."""
        return self._weights

    def load_state_dict(self, state_dict, strict=False):
        """Pretend to restore weights and report no key mismatches."""
        self._weights = state_dict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def to(self, *_args, **_kwargs):
        """Match torch module ``to`` chaining."""
        return self

    def eval(self):
        """Match torch module ``eval`` API."""
        return self


class _DummyHandler:
    """Handler stub exposing the attributes used by ``load_lora``."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(decoder=_DummyDecoder())
        self.device = "cpu"
        self.dtype = torch.float32
        self.quantization = None
        self._base_decoder = None
        self.lora_loaded = False
        self.use_lora = False
        self.lora_scale = 1.0
        self._lora_active_adapter = None
        self._lora_service = SimpleNamespace(
            registry={},
            scale_state={},
            active_adapter=None,
            last_scale_report={},
        )

    def _ensure_lora_registry(self):
        """Satisfy lifecycle hook without side effects."""
        return None

    def _rebuild_lora_registry(self, lora_path=None):
        """Return deterministic empty registry output."""
        _ = lora_path
        return 0, []

    def _debug_lora_registry_snapshot(self):
        """Return simple debug payload."""
        return {}

    def add_lora(self, lora_path, adapter_name=None, exclude_modules=None):
        """Forward to lifecycle implementation to mimic mixin wiring."""
        return lifecycle.add_lora(
            self, lora_path, adapter_name=adapter_name,
            exclude_modules=exclude_modules,
        )


class LifecycleTests(unittest.TestCase):
    """Coverage for LoKr path detection and load branching."""

    def test_resolve_lokr_weights_from_directory(self):
        """Directory containing ``lokr_weights.safetensors`` should resolve."""
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / lifecycle.LOKR_WEIGHTS_FILENAME
            weights.write_bytes(b"")
            resolved = lifecycle._resolve_lokr_weights_path(str(Path(tmp)))
            self.assertEqual(resolved, str(weights))

    def test_resolve_lokr_weights_from_file(self):
        """Direct ``lokr_weights.safetensors`` file should resolve."""
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / lifecycle.LOKR_WEIGHTS_FILENAME
            weights.write_bytes(b"")
            resolved = lifecycle._resolve_lokr_weights_path(str(weights))
            self.assertEqual(resolved, str(weights))

    def test_resolve_lokr_weights_from_custom_safetensors_name(self):
        """Directory should resolve custom LyCORIS safetensors filenames when metadata matches."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp)
            custom = adapter_dir / "custom_lycoris.safetensors"
            custom.write_bytes(b"")

            with patch(
                "acestep.core.generation.handler.lora.lifecycle._is_lokr_safetensors",
                side_effect=lambda path: path == str(custom),
            ):
                resolved = lifecycle._resolve_lokr_weights_path(str(adapter_dir))

        self.assertEqual(resolved, str(custom))

    def test_load_lora_accepts_lokr_directory_without_adapter_config(self):
        """LoKr directory should bypass PEFT config-file requirement."""
        handler = _DummyHandler()
        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp) / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            weights = adapter_dir / lifecycle.LOKR_WEIGHTS_FILENAME
            weights.write_bytes(b"")
            with patch("acestep.core.generation.handler.lora.lifecycle._load_lokr_adapter") as mock_load_lokr:
                message = lifecycle.load_lora(handler, str(adapter_dir))

        self.assertEqual(message, f"✅ LoKr loaded from {weights}")
        mock_load_lokr.assert_called_once_with(handler.model.decoder, str(weights))

    def test_load_lora_invalid_adapter_message_mentions_lokr(self):
        """Invalid adapter error should mention both LoRA and LoKr expectations."""
        handler = _DummyHandler()
        with tempfile.TemporaryDirectory() as tmp:
            message = lifecycle.load_lora(handler, tmp)
        self.assertIn("adapter_config.json", message)
        self.assertIn(lifecycle.LOKR_WEIGHTS_FILENAME, message)

    def test_load_lokr_adapter_recreates_with_dora_when_weight_decompose_enabled(self):
        """Weight-decompose config should request a second LyCORIS net with DoRA enabled."""
        decoder = _DummyDecoder()
        base_net = Mock()
        dora_net = Mock()
        create_lycoris = Mock(side_effect=[base_net, dora_net])
        fake_lycoris = SimpleNamespace(
            LycorisNetwork=SimpleNamespace(apply_preset=Mock()),
            create_lycoris=create_lycoris,
        )
        config = lifecycle.LoKRConfig(weight_decompose=True)

        with patch.dict("sys.modules", {"lycoris": fake_lycoris}):
            with patch("acestep.core.generation.handler.lora.lifecycle._load_lokr_config", return_value=config):
                result = lifecycle._load_lokr_adapter(decoder, "weights.safetensors")

        self.assertIs(result, dora_net)
        self.assertEqual(create_lycoris.call_count, 2)
        self.assertNotIn("dora_wd", create_lycoris.call_args_list[0].kwargs)
        self.assertTrue(create_lycoris.call_args_list[1].kwargs["dora_wd"])
        dora_net.apply_to.assert_called_once_with()
        dora_net.load_weights.assert_called_once_with("weights.safetensors")
        self.assertIs(decoder._lycoris_net, dora_net)

    def test_load_lokr_adapter_uses_base_net_when_dora_not_supported(self):
        """DoRA create failures should warn and keep the initially created LyCORIS net."""
        decoder = _DummyDecoder()
        base_net = Mock()
        create_lycoris = Mock(side_effect=[base_net, RuntimeError("unsupported")])
        fake_lycoris = SimpleNamespace(
            LycorisNetwork=SimpleNamespace(apply_preset=Mock()),
            create_lycoris=create_lycoris,
        )
        config = lifecycle.LoKRConfig(weight_decompose=True)

        with patch.dict("sys.modules", {"lycoris": fake_lycoris}):
            with patch("acestep.core.generation.handler.lora.lifecycle._load_lokr_config", return_value=config):
                with patch("acestep.core.generation.handler.lora.lifecycle.logger.warning") as mock_warning:
                    result = lifecycle._load_lokr_adapter(decoder, "weights.safetensors")

        self.assertIs(result, base_net)
        self.assertEqual(create_lycoris.call_count, 2)
        mock_warning.assert_called_once()
        base_net.apply_to.assert_called_once_with()
        base_net.load_weights.assert_called_once_with("weights.safetensors")
        self.assertIs(decoder._lycoris_net, base_net)

    def test_exact_load_excludes_requested_modules_in_memory(self):
        """Strict callers can exclude new architecture modules without editing artifacts."""
        handler = _DummyHandler()
        captured = {}

        class FakePeftConfig:
            @classmethod
            def from_pretrained(cls, path):
                captured["config_path"] = path
                return SimpleNamespace(exclude_modules=None)

        class FakePeftModel(_DummyDecoder):
            @classmethod
            def from_pretrained(cls, decoder, path, **kwargs):
                captured["decoder"] = decoder
                captured["path"] = path
                captured["kwargs"] = kwargs
                return cls()

        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}")
            fake_peft = SimpleNamespace(
                PeftConfig=FakePeftConfig, PeftModel=FakePeftModel)
            with patch.dict("sys.modules", {"peft": fake_peft}):
                message = lifecycle.load_lora(
                    handler, str(adapter_dir),
                    exclude_modules=["timing_cross_attn.q_proj",
                                     "timing_cross_attn.k_proj",
                                     "timing_cross_attn.v_proj",
                                     "timing_cross_attn.o_proj"],
                )

        self.assertTrue(message.startswith("✅"), message)
        config = captured["kwargs"]["config"]
        self.assertEqual(
            config.exclude_modules,
            ["timing_cross_attn.k_proj", "timing_cross_attn.o_proj",
             "timing_cross_attn.q_proj", "timing_cross_attn.v_proj"],
        )
        self.assertEqual(captured["config_path"], str(adapter_dir))
        self.assertFalse(captured["kwargs"]["is_trainable"])

    def test_unload_lora_restores_lokr_adapter_before_state_restore(self):
        """Unload should call LyCORIS restore() and then restore decoder weights."""
        handler = _DummyHandler()
        handler.lora_loaded = True
        handler._base_decoder = {"w": torch.ones(1)}
        events = []

        lycoris_net = SimpleNamespace(restore=Mock(side_effect=lambda: events.append("restore")))
        handler.model.decoder._lycoris_net = lycoris_net
        handler.model.decoder.load_state_dict = Mock(
            side_effect=lambda *_args, **_kwargs: events.append("load_state_dict") or SimpleNamespace(
                missing_keys=[], unexpected_keys=[]
            )
        )

        message = lifecycle.unload_lora(handler)

        self.assertEqual(message, "✅ LoRA unloaded, using base model")
        self.assertEqual(events, ["restore", "load_state_dict"])
        self.assertIsNone(handler.model.decoder._lycoris_net)
        self.assertFalse(handler.lora_loaded)
        self.assertFalse(handler.use_lora)

    def test_unload_lora_fails_when_lokr_restore_raises(self):
        """Unload should fail fast if LyCORIS restore() raises an exception."""
        handler = _DummyHandler()
        handler.lora_loaded = True
        handler._base_decoder = {"w": torch.ones(1)}
        handler.model.decoder._lycoris_net = SimpleNamespace(restore=Mock(side_effect=RuntimeError("restore failed")))
        handler.model.decoder.load_state_dict = Mock(
            return_value=SimpleNamespace(missing_keys=[], unexpected_keys=[])
        )

        message = lifecycle.unload_lora(handler)

        self.assertIn("❌ Failed to unload LoRA", message)
        self.assertIn("restore failed", message)
        handler.model.decoder.load_state_dict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
