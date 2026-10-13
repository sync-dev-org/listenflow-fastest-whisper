#!/usr/bin/env bash
set -euo pipefail

# Usage: bash tests/fixtures/generate_synthetic_speech.sh FIXTURE_SOURCE_WAV
# Conversion qualified with FFmpeg 8.0.1-3ubuntu2, libswresample 6.1.100.
if [[ $# -ne 1 ]]; then
  printf 'Usage: %s FIXTURE_SOURCE_WAV\n' "$0" >&2
  exit 2
fi

check_source() {
  local path="$1" expected="$2" digest_output actual
  digest_output="$(sha256sum "$path")"
  actual="${digest_output%% *}"
  if [[ "$actual" != "$expected" ]]; then
    printf 'SHA-256 mismatch for %s: expected %s, got %s\n' "$path" "$expected" "$actual" >&2
    exit 1
  fi
}

check_source "$1" 5b99d9424e03c56e358101e03fd82e442fc0fa42d39df5b1db6e48b7966f60ab

fixture_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ffmpeg -y -hide_banner -loglevel error -i "$1" \
  -map_metadata -1 -ac 1 -ar 16000 -c:a pcm_s16le \
  -fflags +bitexact -flags:a +bitexact -f wav \
  "${fixture_dir}/small_boat_16khz_mono_s16le.wav"
