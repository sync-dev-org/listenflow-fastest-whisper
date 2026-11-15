"""Saved-plan damage fixtures preserve the tensoracle v2 envelope checksums."""

import hashlib
import json

import pytest

from test_engine_gpu import _damage_saved_plugin


@pytest.mark.req("SP-LFFASTESTWHISPER-008")
@pytest.mark.parametrize("damage", ["missing-plugin", "plugin-version"])
def test_plugin_damage_keeps_a_valid_v2_envelope(tmp_path, damage):
    """Only the serialized creator changes; the native cache can still verify its envelope."""
    original = b"prefix:LffwFp16Fc1\x01X1Xlistenflow.fastest_whisper:suffix"
    header = {
        "conditions": {"name": "listenflow-fastest-whisper-decoder"},
        "plan_sha256": hashlib.sha256(original).hexdigest(),
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path = tmp_path / "owned.engine"
    path.write_bytes(b"TENSORACLE" + len(encoded).to_bytes(8, "big") + encoded + original)
    assert _damage_saved_plugin(path, damage)
    contents = path.read_bytes()
    assert contents[:10] == b"TENSORACLE"
    size = int.from_bytes(contents[10:18], "big")
    updated = json.loads(contents[18 : 18 + size])
    plan = contents[18 + size :]
    assert updated["conditions"] == header["conditions"]
    assert updated["plan_sha256"] == hashlib.sha256(plan).hexdigest()
    assert plan != original
    assert len(contents) == path.stat().st_size == 18 + len(encoded) + len(original)
