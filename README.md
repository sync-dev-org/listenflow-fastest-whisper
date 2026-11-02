# listenflow-fastest-whisper

`listenflow-fastest-whisper` is the standalone distribution for
`listenflow.fastest_whisper`. Its current fp32 path builds the Whisper large-v3
encoder and decoder directly from a Hugging Face checkpoint with TensorRT's Python
network definition API. It does not load ONNX or distribute an engine or weights.
The numerical entry points establish a correctness reference. `FastestWhisperEngine`
implements the listenflow stream contract and persists its TensorRT engines through
tensoracle. It transcribes all 100 language codes supported by Whisper large-v3.

## Install

Requires Python 3.12 or 3.13 and Linux x86_64, including WSL2. The supported
NVIDIA GPU compute capabilities are sm86, sm89, and sm120. The host supplies an
NVIDIA driver compatible with CUDA 13. Package dependencies include
`cuda-python>=13.0` and `tensorrt-cu13>=11.0`.

Install from PyPI:

```console
pip install listenflow-fastest-whisper
```

The package installs listenflow core (`listenflow>=0.22.0`) and tensoracle from
PyPI as normal dependencies.

The Linux wheel includes the native fc1 plugin and its sm86, sm89, and sm120
cubins. Runtime installation does not require a TensorRT SDK, Triton, or the
tensoracle producer. Source development requires the independent native build
described in [the C++ build guide](https://github.com/sync-dev-org/listenflow-fastest-whisper/blob/main/cpp/README.md).

## Use the engine

Import this module explicitly with `import listenflow.fastest_whisper` or
`from listenflow.fastest_whisper import FastestWhisperEngine`. It is absent from
listenflow core's module table, so `import listenflow` alone followed by
`listenflow.fastest_whisper` attribute access raises `AttributeError`.

Provision a local Hugging Face large-v3 checkpoint directory and pass its absolute
path through `OfflineModelDirectorySource`. The checkpoint directory contains
`config.json`, `tokenizer.json`, and either `model.safetensors` or the fp32
safetensors shards and `model.safetensors.index.fp32.json`. The implementation
reads layer counts, widths, mel bins, positions, and vocabulary size from
`config.json`. When indexed fp32 shards are present, it uses them in preference to
`model.safetensors`.

```python
from collections.abc import AsyncIterator
from contextlib import aclosing

from listenflow import Final, OfflineModelDirectorySource
from listenflow.fastest_whisper import FastestWhisperEngine


async def transcribe(
    checkpoint_directory: str,
    pcm_chunks: AsyncIterator[bytes],
) -> None:
    engine = FastestWhisperEngine(
        model_source=OfflineModelDirectorySource(directory_path=checkpoint_directory),
        precision="fp16",
        device=0,
        language="en",
    )
    async with engine:
        async with aclosing(engine.stream(pcm_chunks)) as events:
            async for event in events:
                if isinstance(event, Final):
                    print(event.utterance_id, event.ts_offset_s, event.text)
```

The caller supplies 16 kHz mono PCM chunks in the selected input format. The
[engine contract](#engine-contract) describes preparation, closure, language,
errors, and TensorRT engine reuse.

## Engine contract

`FastestWhisperEngine` is the listenflow-facing entry point. Construct it with an
`OfflineModelDirectorySource`, a required precision (`fp32`, `fp16`, or `fp8`), a required
CUDA device number, and a required language argument: a large-v3 language code
such as `en`, `ja`, or `yue`, a list such as `["ja", "en"]` to detect only among
those languages, or `None` for automatic detection among all large-v3 languages.
An optional `initial_prompt` supplies text hints for recognition. It accepts 16 kHz mono
PCM in s16le by default or little-endian f32 when `input_format=EngineInputFormat.PCM_F32`.
`boundary_settings` defaults to `UtteranceBoundarySettings()`. The async
`prepare()` checks CUDA, TensorRT, the device number, and GPU generation, loads or
builds the mel, encoder, and decoder TensorRT engines, and runs a trial inference.
FP8 requires sm89 or sm120; selecting it on sm86 raises an environment error
that offers fp32 and fp16. Its linear inputs and weights use E4M3 Q/DQ, while
KV cache, logits and surrounding operations retain fp16 types. The package
includes activation scales calibrated on the official large-v3 weights with
English speech and multilingual FLEURS validation speech; preparation does not
run calibration. See
[accuracy by language](#accuracy-by-language) for measured precision differences.
Weight variants are accepted, but their accuracy with these scales has not been
established.
Changes to these scales or the FP8 graph invalidate saved FP8 engines automatically.
Call it before reading
`stream()`, or use `async with engine` to prepare and release the model automatically.
`stream()` yields listenflow transcript events. `health()` reports `state`, `busy`,
`last_progress_monotonic_s`, stream counters, and hint application. One instance
runs on one event loop and accepts one active stream at a time.

The constructor accepts these decoding keyword arguments, with the same names,
defaults, and accepted values as the numerical transcription entries:

| Argument | Default | Effect |
| --- | --- | --- |
| `beam_size` | `1` | Candidate beam width at temperature zero; 1 uses greedy decoding. |
| `temperature` | `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` | A single temperature or a sequence tried in order when quality checks fail. |
| `best_of` | `5` | Number of sampled candidates at positive temperatures. |
| `compression_ratio_threshold` | `2.4` | Retry when text exceeds this compression ratio; `None` disables this check. |
| `logprob_threshold` | `-1.0` | Retry below this average log probability; `None` disables this check. |
| `no_speech_threshold` | `0.6` | End retries for probable silence with low log probability and skip silent long-form windows; `None` disables this check. |

For example, `beam_size=2, temperature=(0.0, 0.4), best_of=3` first uses a beam
of two candidates, then samples three candidates at temperature 0.4 if quality
checks fail. `temperature=0.0` selects one attempt without retries. Beam width
and `best_of` must be positive integers, excluding bool. Temperatures must be
finite nonnegative numbers, excluding bool; a temperature sequence must be
nonempty. Use a numeric sequence such as a list or tuple; text, bytes, dict,
set, and generator values are rejected. Thresholds accept finite numbers,
excluding bool, or `None`.
Invalid decoding settings raise `ConfigurationError` during construction,
before `prepare()` or GPU access. The engine copies the temperature sequence
at construction and uses the selected settings for every utterance and stream,
including all three precisions, long utterances, and recognition with `initial_prompt`.

The engine constructor also accepts these phrase and repetition rejection settings:

| Argument | Default | Effect |
| --- | --- | --- |
| `reject_phrases` | `[]` | A list of strings identifying phrases to reject during decoding. |
| `reject_repetition` | `False` | Reject consecutive repetitions of the same token sequence at the generated tail. |
| `on_rejected_exhausted` | `"pass"` | Emit the last rejected attempt's text after retries are exhausted; `"drop"` suppresses that window's text instead. Silence is handled before either choice. |
| `phrase_hit_no_speech` | `True` | Treat a phrase-rejected window as silence when its no-speech probability exceeds `no_speech_threshold`, without requiring low average log probability. |

The engine copies the phrase list at construction and uses these settings for
every utterance and stream; callers do not change them per recognition call.
With `reject_phrases=[]` and `reject_repetition=False`, rejection is disabled and
the engine preserves its existing output selection.

Phrase matching normalizes both text and phrases by lowercasing, removing Unicode
punctuation, collapsing consecutive whitespace to one space, and trimming surrounding
whitespace. A match must start and end at word boundaries. Word characters are
letters, numbers, underscores, and Unicode combining marks; other characters and
the ends of the text form boundaries. During generation, a phrase matches only
after its following boundary is confirmed. For example, `DON’T forget to subscribe`
matches `don't forget to subscribe`; `foo,bar` becomes `foobar`, and `go` does not
match `going`. The engine does not segment text into words for languages that do
not use spaces between words.

Invalid rejection settings raise `ConfigurationError` during construction, before
`prepare()` or GPU access: `reject_phrases` must be a list of strings, each phrase
must remain nonempty after normalization, `reject_repetition` and
`phrase_hit_no_speech` must be booleans, and `on_rejected_exhausted` must be
`"pass"` or `"drop"`.

Each candidate stops generating when a phrase or repetition match is confirmed,
but remains in the existing candidate ranking. If the selected candidate is
rejected, the engine first applies the silence rules to that window, then retries
at the next temperature if the window is not silent. This also applies with
`beam_size` greater than one. The temperature sequence determines the retry count;
there is no separate limit. After the last rejected, nonsilent attempt, the engine
uses `on_rejected_exhausted`. If the last attempt fails only the compression-ratio
or average-log-probability check, the engine retains the official implementation's
behavior regardless of this setting. With `phrase_hit_no_speech=False` or
`no_speech_threshold=None`, the engine uses the standard silence rule. An empty
phrase list makes `phrase_hit_no_speech` have no effect on output.

Call `await engine.aclose()` when finished to return its TensorRT execution contexts,
CUDA graphs, streams, engines, and runtimes. When stopping a stream early, close the returned
iterator explicitly with `await stream.aclose()` or `contextlib.aclosing`. Caller
cancellation, iterator closure, and engine closure stop decoding before the next
token and wait for the current native operation to finish.

`capabilities.languages` is a `frozenset` of all 100 large-v3 language codes and
`capabilities.auto_language` is `True`. Language codes are case-sensitive;
unrecognized codes raise `UnsupportedLanguageError`, a `ConfigurationError`,
during construction, including an unrecognized code anywhere in a language list.
An empty language list, non-string list items, and language values other than
a string, list, or `None` raise listenflow's `ConfigurationError` before `prepare()`.
Other invalid settings also raise `ConfigurationError` before recognition.
Environment or model-load failures raise `EnvironmentUnavailableError`; native
recognition failures raise `InferenceError`.
Mapped failures preserve their original exception in `__cause__`.

Before the cached decoder and GPU feature changes, loading large-v3 took the
following times on an NVIDIA RTX PRO 4000 Blackwell
(sm120, 24 GB), driver 616.92, TensorRT 11.3.0.99, Python 3.13.15, and Linux x86_64
under WSL2. These are single measurements of `await engine.prepare()`, including
trial inference, with the Hugging Face checkpoint revision listed below. Each
precision used a separate cache; the saved-engine measurement reused the files
created by its first preparation.

| Precision | No saved TensorRT engines (build and warmup) | Saved engines (load and warmup) |
| --- | --- | --- |
| fp16 | about 60 seconds | about 5 seconds |
| fp32 | about 68 seconds | about 12 seconds |

Loading time depends on the checkpoint storage, host, GPU, and TensorRT version.

With an explicit language code, the model transcribes in that language without
language detection. A one-element list, such as `["ja"]`, behaves the same as
passing that code as a string. With two or more codes, such as `language=["ja", "en"]`,
the model detects each closed utterance's language only among the listed codes
and transcribes in the chosen language. With `language=None`, it detects the language
of each closed utterance and transcribes in the detected language, including non-English languages.
An utterance longer than 30 seconds uses the same timestamp windowing as
`transcribe_long_fp32` and yields one `Final` per text segment. Language detection
runs for each window of a long utterance, using the listed candidates when supplied.

tensoracle stores the mel, encoder, and decoder under `$XDG_CACHE_HOME/tensoracle/`, or
`~/.cache/tensoracle/` when `XDG_CACHE_HOME` is unset. The constructor has no cache
path argument. When a saved engine is corrupt or its build conditions no longer
match, `prepare()` automatically rebuilds that engine and continues. There is no
caller setting for rebuilding engines. If rebuilding fails or the rebuilt engine
cannot be opened, `prepare()` raises `EnvironmentUnavailableError` and preserves the
underlying error in `__cause__`.

### Rejected-window records

An exhausted rejected window produces one record in listenflow's `Final.metadata`.
A phrase-rejected window that is treated as silence also produces a record.
Each record is a tuple of `(key, value)` pairs; `dict(record)` provides named
access to its values.

| Key | Value | Meaning |
| --- | --- | --- |
| `type` | `"rejection"` | Identifies a rejected-window record from this engine. |
| `outcome` | `"passed"`, `"dropped"`, or `"no_speech"` | `"passed"`: rejected text was emitted. `"dropped"`: its output was suppressed after exhausting retries. `"no_speech"`: the phrase-rejected window was treated as silence. |
| `reason` | `"phrase"` or `"repetition"` | The reason the selected attempt was rejected. |
| `phrase` | `str` or `None` | The matching phrase exactly as supplied in `reject_phrases`, preserving its case and punctuation; `None` for repetition. |
| `text` | `str` | The last rejected attempt's text when generation stopped, retained even when its output is dropped or treated as silence. |
| `temperature` | `float` | The last rejected attempt's decoding temperature. |
| `window` | `int` | The window's zero-based index within its utterance. |
| `window_start_s`, `window_end_s` | `float` | The window's start and end in seconds from the utterance's start. |

`Final.utterance_id` identifies the utterance. Records follow window order.
When a rejected window produces text segments, only its first text `Final`
carries the record; passing a rejected window adds no `Final` events. A window
that produces no visible text, including dropped, silent, and whitespace-only
results, instead produces one `Final(text="", metadata=...)`. Its `ts_offset_s`
is the utterance's start plus `window_start_s`. Both listenflow's `transcribe()`
and `transcribe_utterance()` preserve these results; inspect `final.text` before
displaying recognized words.

A rejected attempt followed by a successful retry produces no record. A
repetition-rejected window treated as silence also produces no record. Windows
without text or a record produce no `Final`. With rejection disabled, all
`Final.metadata` values are `()` and transcript text and event counts retain their
existing behavior.

## Accuracy by language

The following measurements use the FLEURS test split at dataset revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd` and `openai/whisper-large-v3` at
checkpoint revision `06f233fe06e710322aca913c1bc4249a0d71fce1`, on an NVIDIA
RTX PRO 4000 Blackwell (sm120, 24 GB). All 6,085 test utterances across these
nine languages were decoded at each precision with an explicit language, no
initial prompt, the default decoding settings, and a fixed seed per utterance.

Runs kept the default temperature retry sequence. Each comparison below selects
the utterances for which both fp32 and the compared precision used only
temperature 0, across all windows. Each precision comparison has its own matched
subset, so its utterance count and fp32 baseline can differ from the other table.
Utterances with higher-temperature attempts are recorded separately because
sampling introduces variation.

Rates are percentages of total edit errors divided by total reference units.
CER counts characters; WER counts words. Text uses the official Whisper
`EnglishTextNormalizer` for English and `BasicTextNormalizer` for the other
languages; CER also removes whitespace after normalization. Deltas are percentage
points relative to fp32 on the same subset, calculated before rounding.

### fp16 compared with fp32

| Language | Unit | Utterances | fp32 | fp16 | Delta (points) |
| --- | --- | ---: | ---: | ---: | ---: |
| English (`en`) | WER | 647 | 4.032% | 4.046% | +0.014 |
| Japanese (`ja`) | CER | 649 | 4.579% | 4.579% | +0.000 |
| Chinese (`zh`) | CER | 945 | 8.004% | 7.996% | -0.008 |
| Cantonese (`yue`) | CER | 811 | 21.007% | 21.027% | +0.021 |
| Korean (`ko`) | CER | 382 | 4.904% | 4.904% | +0.000 |
| Russian (`ru`) | WER | 775 | 4.979% | 4.979% | +0.000 |
| Arabic (`ar`) | WER | 427 | 14.590% | 14.603% | +0.012 |
| Hindi (`hi`) | WER | 398 | 17.275% | 17.304% | +0.029 |
| Thai (`th`) | CER | 964 | 8.106% | 8.085% | -0.020 |

Each language stays within the fp16 limit of fp32 plus 0.05 percentage points
on these matched subsets.

### fp8 compared with fp32

| Language | Unit | Utterances | fp32 | fp8 | Delta (points) |
| --- | --- | ---: | ---: | ---: | ---: |
| English (`en`) | WER | 647 | 4.032% | 4.115% | +0.082 |
| Japanese (`ja`) | CER | 649 | 4.579% | 4.499% | -0.080 |
| Chinese (`zh`) | CER | 945 | 8.004% | 7.920% | -0.084 |
| Cantonese (`yue`) | CER | 807 | 21.035% | 21.255% | +0.221 |
| Korean (`ko`) | CER | 382 | 4.904% | 4.610% | -0.294 |
| Russian (`ru`) | WER | 775 | 4.979% | 4.986% | +0.007 |
| Arabic (`ar`) | WER | 427 | 14.590% | 14.676% | +0.085 |
| Hindi (`hi`) | WER | 400 | 17.475% | 17.212% | -0.264 |
| Thai (`th`) | CER | 957 | 8.099% | 8.151% | +0.052 |

FP8 rates are reported without a pass/fail threshold. The shipped activation
scales were calibrated on English speech and on FLEURS validation speech in the
nine languages above; the test split measured here was not used for calibration.
FP8 stays within 0.3 percentage points of fp32 for every listed language.

## Versioning

Release versions come from exact `vX.Y.Z` git tags through setuptools-scm, as configured
in `[tool.setuptools_scm]` in
[python/pyproject.toml](https://github.com/sync-dev-org/listenflow-fastest-whisper/blob/main/python/pyproject.toml).
A checkout without a matching release tag uses the fallback version `0.0.0.dev0`.
Release versions are published on PyPI.

## Development

Use Python `>=3.12,<3.14` and [uv](https://docs.astral.sh/uv/). The Python project,
tests, and development scripts live in `python/`, with one environment at
`python/.venv`. Run the following commands from the repository root:

```sh
cd python
uv sync --group dev --group evaluation
uv lock --check
uv run --group dev --group evaluation pytest
uv run --group dev ruff check
uv run --group dev ruff format --check
```

Run the remaining Python commands below from `python/`. The standalone C++ project
in `cpp/` uses the host's CMake and Ninja and does not discover Python or nanobind.
The Python project's README and license are symlinks to the repository's originals;
source distributions include their contents as regular files.

The evaluation group supplies the CPU libraries used by the reference and scoring
tests. The ordinary test suite runs without a sibling checkout, checkpoint, or GPU.
Distribution and native build tests need the verified build inputs described in
the C++ guide. Release wheels use the pinned container and ELF checks in
[the release build procedure](https://github.com/sync-dev-org/listenflow-fastest-whisper/blob/main/cpp/RELEASE.md).
The model's `gpu` tests require `LFFW_USE_GPU=1` and `LFFW_CHECKPOINT_DIR` pointing
to a usable local Hugging Face checkpoint. For cubin production and the standalone
native plugin build, see [the C++ development guide](https://github.com/sync-dev-org/listenflow-fastest-whisper/blob/main/cpp/README.md). Its small GPU
tests require a verified TensorRT SDK and `LFFW_USE_GPU=1`, without a Whisper
checkpoint. Before using a shared GPU, coordinate its use
with the host's shared-resource owner. Run GPU comparisons serially. Use a directory
on disk for the checkout; cache checks create several GiB of engine files. GPU
tests automatically put pytest temporary directories and `XDG_CACHE_HOME` in a
unique directory under the repository root's `.nf/tmp/pytest-gpu/`. An explicit
`--basetemp` places that directory under the supplied base instead. CPU tests retain their usual temporary
directories and cache environment.

```sh
LFFW_USE_GPU=1 LFFW_CHECKPOINT_DIR=/absolute/path/to/whisper-large-v3 \
  uv run --group dev --group evaluation pytest tests/test_fp32_gpu.py -q
```

Pytest removes the GPU run's directory, including its engine cache, when the run
finishes, including test failures and `KeyboardInterrupt`. A forced process kill
or host crash can prevent cleanup; remove that run's directory manually once its
processes have stopped. Concurrent runs use separate directories and leave each
other's files alone.

Plan disk and memory per GPU test file from the figures below, measured with one
file per process on an RTX PRO 4000 Blackwell (sm120) with TensorRT 11.4. Disk is
the peak size of the run's temporary directory. Memory is the cgroup peak, which
includes page cache; every file also passes with a 16 GiB memory limit and no swap.
`python/tests/test_engine_gpu.py` holds two engine caches at once: the module's shared
cache and the empty cache in which the cache reuse and corruption check starts.

| file | disk (GiB) | memory (GiB) | duration on sm120 (min) |
|---|---:|---:|---:|
| `python/tests/test_gpu_mel.py` | < 0.1 | 2.1 | < 1 |
| `python/tests/test_fp16_gpu.py` | 8.7 | 17.6 | 7 |
| `python/tests/test_bucket_graphs_gpu.py` | 2.9 | 9.6 | 3 |
| `python/tests/test_fp32_gpu.py` | 5.8 | 17.3 | 3 |
| `python/tests/test_engine_gpu.py` | 17.3 | 22.8 | 17 |

Engine builds are slower on older generations. On an RTX A6000 (sm86),
`python/tests/test_fp16_gpu.py` took 34 minutes, and the cache reuse and corruption check
alone took 38 minutes.

Outside pytest, the numerical comparison entries store encoder and decoder engines
under `$XDG_CACHE_HOME/tensoracle/` (or `~/.cache/tensoracle/` when unset) and rebuild
them on every model construction. Set `XDG_CACHE_HOME` to a directory on disk
for those comparisons and remove the engine files after the run. Both the
comparison entries and `FastestWhisperEngine` use tensoracle to build and execute
the same graph declarations.

`transcribe_fp32(checkpoint_dir, audio)` accepts a one-dimensional NumPy array of
16 kHz mono samples, normalized to floating-point values as PCM16 divided by
32768, with at most 30 seconds of audio. It returns a `Transcription` containing
`text` and `tokens`; the token list includes the final end-of-text token. The
function is an fp32 correctness entry point.

The numerical `transcribe_fp32`, `transcribe_fp16`, and `transcribe_long_fp32`
entries accept `initial_prompt`, `beam_size`, `temperature`, `best_of`,
`compression_ratio_threshold`, `logprob_threshold`, and
`no_speech_threshold`. `beam_size` defaults to 1 and `temperature` defaults to
`(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`. An omitted decoding setting starts with
greedy decoding and retries at higher temperatures only when a quality check
fails. A beam width above 1
selects candidates at temperature 0. `temperature` can be a single value or a
sequence of values tried in order when a window exceeds the compression ratio
limit or falls below the average log probability limit. Positive temperatures
sample candidates without beam search. `best_of` selects the number of sampled
candidates and defaults to 5. These retry defaults match the official
openai-whisper `transcribe` defaults: compression ratio 2.4, average log
probability -1.0, and no-speech probability 0.6. Pass `temperature=0.0` and
`best_of=1` to retain greedy decoding without retries.
The no-speech threshold can end retries and skip a long-form window. Each
threshold accepts `None` to disable its check. `FastestWhisperEngine` accepts
the same decoding arguments, as described in the [engine contract](#engine-contract).

### Changing the TensorRT graphs

The encoder also projects cross-attention keys and values once per audio window.
Those tensors and the encoder output remain on the GPU. The decoder accepts one
token, its position, and that candidate's self-attention keys and values, and
returns the current position's logits, a suppressed greedy token ID, quality
statistics, and updated keys
and values. Candidate
selection reuses the cache belonging to the selected prefix; retries reuse the
initial prefix, and a new audio window releases the previous window's tensors.
The self-attention cache begins with a dummy position that is always masked out,
because tensoracle's input profiles require positive dimensions. Encoder output
is copied to the host only when requested for numerical comparisons.

Executions reuse device allocations sized for the largest declared input profile.
Retained results keep their storage until released; the model returns all allocations
on closure. Decoder executions use CUDA graphs with a separate, unchanged tensoracle
execution context for each padded self-attention length interval. At most 16 captures
are retained. Padding is masked out, and device copies restore each result to its
compact cache layout before the next decoder step.
Greedy decoding keeps logits on the device, uploads a suppression mask, and transfers
only the selected int32 token ID when retries are disabled. The first greedy
attempt with retries enabled uses the same device selection and transfers three
fp32 scalars: the selected token ID, its log probability, and the unfiltered
no-speech probability. Equal maximum scores select the lowest eligible ID.
Beam search, sampling, and timestamp selection continue to read logits on the host.

The WER and long-form evaluation scripts keep their historical measurement
defaults (`--temperature 0 --best-of 1`) and output names. To measure the engine's
default retry policy, pass `--temperature 0,0.2,0.4,0.6,0.8,1 --best-of 5`; those
results receive separate names. The latency benchmark and profiler measure
greedy without retries for comparison with faster-whisper.

The mel engine computes a factored 400-point DFT, filter projection, logarithm, and
short-form clipping in fp32, including for fp16 recognition. Short-form features
pass directly to the encoder on the device. Long-form transforms use overlapping
blocks and return their log features to the existing host window selector; clipping
uses one threshold for the entire recording. All three engines follow the same
automatic recovery policy. The builder uses optimization level 5 with TF32 disabled.

The cache identity includes the checkpoint configuration, weight files, precision,
GPU, TensorRT version, input shapes, and a graph version for each of the mel, encoder,
and decoder. The mel identity also includes its filter asset and recognition precision;
its arithmetic remains fp32. The versions live in `_GRAPH_VERSION` in
`python/src/listenflow/fastest_whisper/_engines.py`. When a change to `_network.py` or `_mel.py`
alters a graph in a way that makes its serialized engine incompatible, raise that
graph's version. Raise all affected versions when build settings change.
Review the graph version in the same change as the graph,
then verify that a newly prepared engine uses new cache files and an unchanged graph
continues to reuse its old file. The version is a manual compatibility counter, not
a package version.

`transcribe_long_fp32(checkpoint_dir, audio)` reads a complete 16 kHz mono
waveform and returns a `LongTranscription` with the full text and timed
segments. It computes the mel features for the whole recording, then advances
30-second windows using decoded timestamp tokens and passes confirmed text and
timestamps to the next window. The existing `transcribe_fp32` and
`transcribe_fp16` entry points keep their single-window decode path. The
[long-form fp32 evaluation script](https://github.com/sync-dev-org/listenflow-fastest-whisper/blob/main/python/scripts/evaluate_longform.py)
compares complete transcripts, timed segments, and decoded token sequences.

## Fixed fp32 reference

The reference generator uses `openai-whisper==20250625` on CPU with `fp16=False`.
It loads the official [`large-v3.pt`](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt)
whose SHA-256 is
`e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb`.
The official model URL identifies the file by its content hash; it has no separate
model revision. Its weights were checked against
[`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1)
at revision `06f233fe06e710322aca913c1bc4249a0d71fce1`:

| Hugging Face file | SHA-256 |
| --- | --- |
| `model.safetensors` (fp16) | `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95` |
| `model.fp32-00001-of-00002.safetensors` | `08e0005225b3dbaf55dd13ac62926cc7e02c1025d66fa375e6fb305ff79cd4f9` |
| `model.fp32-00002-of-00002.safetensors` | `630ca774672856d2e0e39a702e590f635a1cfc5726a64b6578ab46dd367369a9` |

All 1,259 corresponding tensors have the same values after fp16-to-fp32 expansion.
The tokenizer IDs agree for all 51,866 tokens. The official checkpoint and the
Hugging Face files therefore represent the same weights and tokenizer for this
reference.

The checked-in `small_boat_16khz_mono_s16le.wav` is synthetic speech copied
from [listenflow-faster-whisper, commit d7c6087edd7b871bebda8d245a71a5f22b01cdc3](https://github.com/sync-dev-org/listenflow-faster-whisper/tree/d7c6087edd7b871bebda8d245a71a5f22b01cdc3/tests/fixtures).
It was generated with Qwen3-TTS-12Hz-1.7B-CustomVoice (Apache-2.0) using the
built-in Aiden speaker, English, greedy decoding, and seed 7, without a
reference recording. The three spoken sentences about a fishing boat were
written for the related repository. This repository distributes the generated
speech under MIT, as does its source repository; the external synthesis model
and weights retain Apache-2.0. The runtime, weights, settings, text, conversion,
and source hashes are recorded in `python/tests/fixtures/small_boat_speech_record.txt`.

The WAV contains 186,880 samples (11.68 seconds) of mono 16 kHz signed 16-bit
PCM. Its SHA-256 is
`f80e9a97d1294af40e1f6076454437825388d4ceff87542abd1127bf1ae587bb`.
To reproduce the conversion from the original 24 kHz source WAV with the
recorded FFmpeg version:

```sh
bash tests/fixtures/generate_synthetic_speech.sh /absolute/path/to/fixture-small-boat.wav
```

The script verifies the source hash before converting. The source WAV is an
external input; regenerating the recognition references uses the checked-in
16 kHz fixture directly. The second WAV contains one second of zero samples,
generated for this repository; its SHA-256 is
`643f8a8dc8bd9c19225afffad2becfec5426180b3749cb208abdf1a6c8354efc`.
Both fixtures are in `python/tests/fixtures/`, so reference regeneration needs no
sibling repository.
The Japanese language detection fixture is a 25.5-second excerpt of
[Marsian's Japanese pronunciation recording](https://commons.wikimedia.org/wiki/File:Ja-pronunciation-practice5.ogg).
The author dedicated the recording to the public domain. The 30.928299-second
Ogg original has SHA-256
`317ced938950ed677e2ed6063417a784c670e8e620ac642d9b9f55c4ca651c70`.
The excerpt ends during silence after a complete spoken phrase. Recreate the
16 kHz mono s16le WAV with:

```sh
ffmpeg -nostdin -hide_banner -loglevel error \
  -i Ja-pronunciation-practice5.ogg -t 25.5 -ac 1 -ar 16000 \
  -c:a pcm_s16le ja_pronunciation_practice5_25s_16khz_mono_s16le.wav
```

The WAV in `python/tests/fixtures/` has SHA-256
`8fc6dc380db5d2b7197c53926657441b7906eefd3217cfa909d0f640ddb8829a`.
With this exact WAV, openai-whisper 20250625 large-v3 on CPU detects token 50266,
`<|ja|>`, from the first 30-second padded mel window. No audio was played or recorded.
The mel filter coefficients in `python/src/listenflow/fastest_whisper/assets/mel_filters.npz`
come from [openai-whisper](https://github.com/openai/whisper), which is MIT licensed.

For each WAV, the generator divides PCM16 samples by 32768, right-pads or trims to
480,000 samples, calls `whisper.log_mel_spectrogram` with the model's mel bin count,
then calls `whisper.decode(model, mel, options)`. The options fix English
transcription, temperature 0 greedy decoding, no beam or sampling alternatives,
`without_timestamps=True`, `suppress_tokens="-1"`, `suppress_blank=True`, no
prompt or prefix, the official default sample length (224 tokens for large-v3),
and `fp16=False`. It records the log-mel array, encoder output, each decoder
step's raw logits before token suppression under the same oracle token prefix,
and the greedy token sequence including end-of-text. These options describe the
reference fixture, not a lasting decode policy for the module.

Regenerate the reference with the already downloaded official file:

```sh
uv run --group oracle python scripts/generate_oracle.py \
  --checkpoint /absolute/path/to/large-v3.pt
```

The default command also regenerates `initial_prompt.json`,
`initial_prompt_long_small_boat_repeat.json`, `beam_size_2.json`,
`beam_with_prompt.json`, and `beam_long_with_prompt.json`. Short prompted
cases cover both a long, unrelated hint and a one-word hint; long-form cases
repeat the synthetic waveform three times (35.04 seconds) to cross a
30-second window, and record segment boundaries and each window's tokens
and previous-text prompt. All outputs come from the official CPU fp32
implementation. The generated arrays and their hashes are in
`python/tests/fixtures/oracle/`.
The numerical tests allow absolute and relative errors of `3e-5` for log-mel,
an absolute error of `2e-2` and relative error of `5e-3` for encoder output,
and absolute and relative errors of `2e-2` for raw logits. CPU FFT rounding and
fp32 reduction orders across CPU and GPU architectures account for these margins.
With this speech input, the official fp32 implementation with TF32 disabled differs
between CPU and GPU by up to `0.076` in encoder output, with the largest differences
in the padded silence. The engine's measured maximum is below `0.018`. The encoder
bound leaves margin for different reduction orders while retaining the per-step
logit checks. Every greedy token ID must match the official CPU reference exactly.
