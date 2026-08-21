"""Speech in, speech out - the part that makes the demo feel like talking.

    python -m src.jarvis

LISTENING is Whisper (mlx-whisper on Apple Silicon). It transcribes; it does
NOT translate. "내 가방 어딨어?" comes back as Korean text, which is exactly
what we want - src/vocab.py knows Korean, and src/i18n.py answers in Korean.

SPEAKING is the `say` command that ships with macOS. No install, no download,
no network, and it has good Korean and Chinese voices. For a demo that has to
work in a hall with unknown wifi, "already on the machine" beats "better".

EVERYTHING DEGRADES. No microphone, no Whisper, no `say` - each one falls back
to text instead of failing. A demo that silently drops to the keyboard is
recoverable; a traceback on the projector is not.
"""

import importlib.util
import os
import queue
import shutil
import subprocess
import sys
import threading
import types
import wave

# macOS voices per language. `say -v '?'` lists what is installed; if the voice
# is missing, say() falls back to the system default rather than erroring.
VOICES = {
    "ko": "Yuna",
    "zh": "Tingting",
    "en": "Samantha",
}

SPOKEN_LIMIT = 200         # characters; past this it stops being an answer

SAMPLE_RATE = 16000        # what Whisper expects
SILENCE_SECONDS = 1.2      # how long a pause ends a phrase
SILENCE_LEVEL = 500        # amplitude below this counts as silence

# "small" is the smallest one that hears Korean and Chinese reliably, but it is
# a ~480MB download - painful on a slow connection and impossible to redo in a
# hall. WHISPER_MODEL lets a machine drop to "base" (~145MB) or "tiny" (~75MB)
# without editing code. Accuracy falls off fast below "small" for CJK, so treat
# the smaller ones as a way to get through a bad network, not as the default.
_WHISPER_REPO = os.environ.get("WHISPER_MODEL",
                               "mlx-community/whisper-small-mlx")
_model_loaded = False


# --- speaking ---------------------------------------------------------------
_talking = None            # the `say` we started last, so we can cut it off


def can_speak():
    return sys.platform == "darwin" and shutil.which("say") is not None


def spoken_form(text):
    """The sentence to say out loud, not the whole reply on screen.

    A reply carries two things: one sentence for the person, and a comparison
    table so they can SEE the difference between the candidates. Reading the
    table aloud - "Candidate A: white, metal, large, laptop..." - buries the
    question under fifteen seconds of attributes nobody can hold in their head.
    The table stays on screen, where it works; the voice gets the sentence.
    """
    first = str(text or "").strip().split("\n")[0].strip()
    if len(first) <= SPOKEN_LIMIT:
        return first
    cut = first[:SPOKEN_LIMIT]
    for stop_mark in (". ", "? ", "! ", "。", "？", "! "):
        at = cut.rfind(stop_mark)
        if at > SPOKEN_LIMIT // 2:
            return cut[:at + len(stop_mark)].strip()
    return cut.rstrip() + "..."


def stop():
    """Cut off whatever is being said. A new question means the old answer no
    longer matters - two voices over each other is worse than either alone."""
    global _talking
    process, _talking = _talking, None
    if process is not None and process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass


def say(text, lang="en", wait=False):
    """Speak `text`. Returns immediately unless `wait` - the photo should
    appear WHILE it talks, not after."""
    global _talking
    stop()
    spoken = spoken_form(text)
    if not spoken or not can_speak():
        return None
    command = ["say"]
    voice = VOICES.get(lang)
    if voice:
        command += ["-v", voice]
    command.append(spoken)
    try:
        process = subprocess.Popen(command,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
    except OSError:
        return None
    _talking = process
    if wait:
        process.wait()
    return process


def wait_until_quiet(timeout=12):
    """Let the answer finish before the microphone opens.

    Without this the mic records the laptop speaker and Whisper faithfully
    transcribes our own answer back to us.
    """
    process = _talking
    if process is None:
        return
    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        stop()


def installed_voices():
    """Which of our three voices this machine actually has."""
    if not can_speak():
        return {}
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True,
                                 text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return {lang: voice for lang, voice in VOICES.items()
            if any(line.startswith(voice) for line in listing.splitlines())}


# --- listening --------------------------------------------------------------
def _ensure_numba():
    """Let mlx-whisper import on a Python that has no numba wheel yet.

    mlx-whisper declares numba as a hard dependency, but it uses it in exactly
    one place: two @numba.jit functions in timing.py that implement dynamic time
    warping for WORD-level timestamps. We ask for a plain transcription, so
    those functions are never called - only the decorator is, at import time.

    On Python 3.14 there is no numba wheel, so `pip install mlx-whisper` fails
    outright and speech input dies for a reason that has nothing to do with
    speech. Installing with --no-deps and standing in a no-op decorator here
    costs us nothing we use, and the shim only appears when the real numba is
    genuinely absent - on a machine that has it, this function does nothing.
    """
    if "numba" in sys.modules or importlib.util.find_spec("numba") is not None:
        return

    def jit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function

    stub = types.ModuleType("numba")
    stub.jit = jit
    stub.njit = jit
    stub.vectorize = jit
    stub.prange = range
    stub.__shim__ = True
    sys.modules["numba"] = stub


def can_listen():
    _ensure_numba()
    try:
        import mlx_whisper          # noqa: F401
        import sounddevice          # noqa: F401
    except Exception:               # noqa: BLE001
        # Not only ImportError: sounddevice raises OSError when the PortAudio
        # library itself is missing, and that must fall back to typing too.
        return False
    return True


def missing_packages():
    """What to pip install, so the message can say something useful."""
    _ensure_numba()
    missing = []
    for name, package in (("mlx_whisper", "mlx-whisper --no-deps"),
                          ("sounddevice", "sounddevice")):
        try:
            __import__(name)
        except Exception:           # noqa: BLE001
            missing.append(package)
    return missing


def record_until_silence(max_seconds=15, on_level=None):
    """Record from the default microphone until the speaker stops.

    Returns raw 16-bit mono samples at SAMPLE_RATE, or None if there is no
    microphone. Stopping on silence rather than a fixed length is what makes it
    feel conversational - you finish your sentence and it answers.
    """
    import numpy as np
    import sounddevice as sd

    chunks = queue.Queue()

    def callback(indata, _frames, _time, status):        # noqa: ANN001
        if status:
            pass                    # over/underruns are not worth stopping for
        chunks.put(indata.copy())

    collected = []
    silent_for = 0.0
    heard_anything = False
    block = 0.1                                          # seconds per callback

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="int16", callback=callback,
                                blocksize=int(SAMPLE_RATE * block))
    except Exception:                                    # noqa: BLE001
        return None

    with stream:
        elapsed = 0.0
        while elapsed < max_seconds:
            try:
                data = chunks.get(timeout=1.0)
            except queue.Empty:
                break
            collected.append(data)
            elapsed += block

            level = float(np.abs(data).mean())
            if on_level:
                on_level(level)
            if level > SILENCE_LEVEL:
                heard_anything = True
                silent_for = 0.0
            elif heard_anything:
                silent_for += block
                if silent_for >= SILENCE_SECONDS:
                    break

    if not collected or not heard_anything:
        return None
    return np.concatenate(collected)


def write_wav(samples, path):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    return path


def transcribe(path):
    """Audio file -> (text, language). Whisper detects the language itself."""
    _ensure_numba()
    import mlx_whisper

    result = mlx_whisper.transcribe(str(path), path_or_hf_repo=_WHISPER_REPO)
    return (result.get("text") or "").strip(), result.get("language") or "en"


def is_talking():
    """Is the machine speaking right now? The browser microphone cannot tell
    our own answer from the person's next question."""
    return _talking is not None and _talking.poll() is None


def to_mono_float(samples):
    """Whatever the browser sent -> mono float32 in [-1, 1].

    Gradio hands over int16 from some browsers and float32 from others, mono
    or stereo depending on the device. Normalising once here keeps the silence
    threshold below meaningful instead of being 3000x off on half the laptops.
    """
    import numpy as np

    data = np.asarray(samples)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype.kind in "iu":
        data = data.astype("float32") / float(np.iinfo(data.dtype).max)
    return data.astype("float32")


def resample_to_whisper(samples, sample_rate):
    """Browsers record at 44.1 or 48kHz; Whisper wants 16kHz."""
    import numpy as np

    if sample_rate == SAMPLE_RATE:
        return samples
    try:
        from scipy.signal import resample_poly
        from math import gcd
        divisor = gcd(int(sample_rate), SAMPLE_RATE)
        return resample_poly(samples, SAMPLE_RATE // divisor,
                             int(sample_rate) // divisor).astype("float32")
    except ImportError:
        # Nearest-sample fallback. Audibly worse, still transcribable - better
        # than losing speech input because one library is missing.
        step = float(sample_rate) / SAMPLE_RATE
        wanted = int(len(samples) / step)
        picks = (np.arange(wanted) * step).astype("int64")
        return samples[picks].astype("float32")


def transcribe_samples(samples, sample_rate):
    """Audio already in memory -> (text, language).

    Passing the array straight to Whisper skips both the temporary file and
    ffmpeg, which transcribe(path) shells out to and which is not installed on
    every machine.
    """
    _ensure_numba()
    import mlx_whisper

    audio = resample_to_whisper(to_mono_float(samples), sample_rate)
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=_WHISPER_REPO)
    return (result.get("text") or "").strip(), result.get("language") or "en"


class PhraseCollector:
    """Turns a stream of microphone chunks into finished sentences.

    The browser sends audio continuously once recording starts. This decides
    where one utterance ends: sound, then SILENCE_SECONDS of quiet. That single
    decision is what removes the buttons - you finish your sentence and it
    answers, then it is listening again, with nothing to press in between.

    Kept here rather than in the page so the web UI and the terminal demo end
    a sentence the same way.
    """

    LEVEL = 0.004              # floor of the threshold, on a 0-1 scale
    OVER_ROOM = 4.0            # speech must be this many times the room noise
    PEAK_OVER_ROOM = 5.0       # ...and must peak this far above it somewhere
    MINIMUM_SECONDS = 0.4      # shorter than this is a cough, not a question
    MAXIMUM_SECONDS = 15.0

    def __init__(self, floor=None):
        self.chunks = []
        self.sample_rate = SAMPLE_RATE
        self.heard_speech = False
        self.quiet_seconds = 0.0
        self.seconds = 0.0
        self.level = 0.0
        self.peak = 0.0
        # How loud this room is when nobody is talking. Learned rather than
        # assumed: a browser's automatic gain, a built-in laptop microphone and
        # a hall full of people put "quiet" in wildly different places, and a
        # fixed threshold is either deaf in one of them or hallucinating in
        # another.
        self.floor = floor

    def reset(self):
        self.__init__(floor=self.floor)     # keep what we learned about the room

    def threshold(self):
        if self.floor is None:
            return self.LEVEL
        return max(self.LEVEL, self.floor * self.OVER_ROOM)

    def add(self, sample_rate, samples):
        """Feed one chunk. Returns the phrase when it is complete, else None."""
        import numpy as np

        data = to_mono_float(samples)
        if not len(data):
            return None
        self.sample_rate = sample_rate
        span = len(data) / float(sample_rate)
        self.seconds += span
        self.level = float(np.abs(data).mean())
        self.peak = max(self.peak, self.level)

        # Learn the room CONTINUOUSLY, not only before the first word. A laptop
        # microphone in a normal room idles around the level a quiet voice
        # reaches, so a fixed threshold either hears the room breathing as
        # speech - in which case the sentence never ends and nothing is ever
        # transcribed - or misses someone speaking softly. Falling fast and
        # rising slowly means a door slamming does not deafen it for a minute.
        if self.floor is None:
            # Start deaf-side-up rather than trusting the first chunk: if
            # recording began mid-sentence, that chunk is the person talking,
            # and treating it as the room's baseline makes us ignore them.
            self.floor = min(self.level, self.LEVEL)
        elif self.level < self.floor:
            self.floor = 0.7 * self.floor + 0.3 * self.level
        else:
            self.floor = 0.97 * self.floor + 0.03 * self.level

        if self.level > self.threshold():
            self.heard_speech = True
            self.quiet_seconds = 0.0
            self.chunks.append(data)
        elif self.heard_speech:
            self.quiet_seconds += span
            self.chunks.append(data)          # keep the tail of the word
        else:
            return None                        # room noise, before anything said

        finished = (self.quiet_seconds >= SILENCE_SECONDS
                    or self.seconds >= self.MAXIMUM_SECONDS)
        if not finished:
            return None

        phrase = np.concatenate(self.chunks) if self.chunks else None
        spoken = self.seconds - self.quiet_seconds
        peak, floor = self.peak, self.floor
        self.reset()
        if phrase is None or spoken < self.MINIMUM_SECONDS:
            return None
        # Nothing in it ever got properly loud, so it was the room, not a
        # person. This matters more than it sounds: handed a stretch of near
        # silence, Whisper does not return nothing - it confidently invents a
        # short phrase, and the demo answers a question no one asked.
        if peak < max(self.LEVEL, (floor or 0) * self.PEAK_OVER_ROOM):
            return None
        return sample_rate, phrase


# --- "hey lopa" -------------------------------------------------------------
# Whisper writes a made-up name a different way every time, and it has no idea
# this one is a name at all. Rather than one spelling we accept a family of
# them, in all three scripts plus the mishearings we can predict: Korean has no
# separate l/r, so 로파 comes back as ropa or lopa about equally, and the plain
# consonant is often written double or aspirated (로빠, 로퍼).
#
# The trade is deliberately one-sided. A false wake costs a shrug; a name that
# has to be said three times in front of an audience costs the demo.
WAKE_WORDS = (
    # Korean
    "헤이로파", "헤이로빠", "헤이로바", "헤이노파", "헤이로퍼", "헤이로파야",
    "헤이로화", "헤이롶아", "하이로파", "해이로파", "에이로파", "헤이라파",
    "로파야", "로파", "로빠", "로퍼",
    # English, and the Latin spellings Whisper reaches for when a Korean
    # speaker says it
    "heylopa", "heylofa", "heyropa", "heyrofa", "heylopah", "heylopaa",
    "hailopa", "helopa", "heylowpa", "heyloppa", "heyloafa", "hailofa",
    "lopa", "lofa", "ropa",
    # Mixed script: Whisper often writes the greeting in one alphabet and the
    # name in another, especially at the start of a Korean sentence.
    "hey로파", "헤이lopa", "헤이lofa",
    # Chinese
    "嘿罗帕", "嘿洛帕", "海罗帕", "你好罗帕", "罗帕", "洛帕",
)

SLEEP_AFTER_SECONDS = 45


def _squash(text):
    """Letters only, lowercased, with a map back to the original positions.

    Spacing is the least reliable thing Whisper produces - "헤이 로파",
    "헤이로파" and "Hey, Lopa!" are the same word - so the match happens with
    every space and comma removed. The map is what lets us hand back the REST
    of the sentence afterwards, with its spacing intact.
    """
    kept, positions = [], []
    for index, character in enumerate(str(text)):
        if character.isalnum():
            kept.append(character.lower())
            positions.append(index)
    return "".join(kept), positions


def hears_wake_word(text):
    """(heard, rest of the sentence).

    "헤이 로파, 내 가방 어딨어?" said in one breath returns (True, "내 가방
    어딨어?") - waking up and then asking people to repeat themselves is what
    makes voice assistants tiring.
    """
    squashed, positions = _squash(text)
    if not squashed:
        return False, ""
    best = None
    for word in WAKE_WORDS:
        target, _ = _squash(word)
        at = squashed.find(target)
        if at >= 0 and (best is None or len(target) > best[1] - best[0]):
            best = (at, at + len(target))
    if best is None:
        return False, ""
    end = best[1]
    if end >= len(positions):
        return True, ""
    remainder = str(text)[positions[end - 1] + 1:]
    return True, remainder.lstrip(" ,.!?~-·、，。！？").strip()


class WakeGate:
    """Decides whether the microphone should be acted on at all.

    The stream is always open once recording starts, which is what removes the
    buttons - but "always open" and "always acting" are different things. This
    keeps it quiet until it hears its name, then stays awake through the whole
    exchange (nobody should have to say "hey lopa" again to answer "blue"), and
    goes back to sleep once the conversation stops.
    """

    IGNORE, WOKE, SPEAK = "ignore", "woke", "speak"

    def __init__(self, enabled=True, timeout=SLEEP_AFTER_SECONDS):
        self.enabled = enabled
        self.timeout = timeout
        self.awake_until = 0.0

    @property
    def awake(self):
        import time
        return not self.enabled or time.monotonic() < self.awake_until

    def touch(self):
        import time
        self.awake_until = time.monotonic() + self.timeout

    def check(self, text):
        """(what to do, what was actually said)."""
        heard, rest = hears_wake_word(text)
        if heard:
            self.touch()
            return (self.SPEAK, rest) if rest else (self.WOKE, "")
        if self.awake:
            self.touch()
            return self.SPEAK, text
        return self.IGNORE, ""


def warm_up(on_progress=None):
    """Load Whisper before the demo starts.

    The first transcribe() downloads and loads the model - twenty seconds of
    silence at exactly the wrong moment. Call this while the audience is still
    reading the title slide.
    """
    global _model_loaded
    if _model_loaded or not can_listen():
        return _model_loaded

    import tempfile

    import numpy as np

    if on_progress:
        on_progress("Loading the speech model...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        write_wav(np.zeros(SAMPLE_RATE // 2, dtype="int16"), handle.name)
        try:
            transcribe(handle.name)
            _model_loaded = True
        except Exception as exc:                          # noqa: BLE001
            if on_progress:
                on_progress(f"   speech model unavailable: {exc}")
    return _model_loaded


def listen(on_progress=None):
    """One spoken turn: record, transcribe. Returns (text, language) or None."""
    import tempfile

    samples = record_until_silence()
    if samples is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        write_wav(samples, handle.name)
        try:
            return transcribe(handle.name)
        except Exception as exc:                          # noqa: BLE001
            if on_progress:
                on_progress(f"   could not transcribe: {exc}")
            return None


# --- typing while it listens ------------------------------------------------
def read_line_async():
    """A keyboard line, on a background thread.

    The escape hatch that makes a live voice demo survivable: a noisy hall, an
    accent the model does not like, a phrase it mishears three times running.
    You just type it instead, and nobody in the audience has to watch you fight
    a microphone.
    """
    result = {}

    def worker():
        try:
            result["line"] = input()
        except (EOFError, KeyboardInterrupt):
            result["line"] = None

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread, result


# --- preflight --------------------------------------------------------------
def _preflight():
    """`python -m src.voice` - check the speech setup before it matters.

    Every part of this fails quietly at demo time in a way that looks like the
    system is broken: a missing voice reads Korean with an English accent, an
    undownloaded model stalls for a minute, a muted microphone hears nothing.
    Run it the day before, not five minutes before.
    """
    print(f"speaking: {'yes' if can_speak() else 'no (not macOS?)'}")
    if can_speak():
        have = installed_voices()
        for lang, name in VOICES.items():
            mark = "ok" if lang in have else "MISSING - System Settings > "
            print(f"   {lang}: {name} - {mark}")
        for lang, text in (("en", "Ready."), ("ko", "준비됐습니다."),
                           ("zh", "准备好了。")):
            say(text, lang, wait=True)

    print(f"listening: {'yes' if can_listen() else 'no'}")
    if not can_listen():
        print(f"   pip install {' '.join(missing_packages())}")
        return
    print(f"   model: {_WHISPER_REPO}")
    if not warm_up(on_progress=lambda line: print(f"   {line}")):
        return
    print("   model ready")
    print("\nSay something (any language), or ctrl-c to stop:")
    heard = listen(on_progress=print)
    if heard:
        print(f"   heard ({heard[1]}): {heard[0]}")
    else:
        print("   heard nothing - check the microphone input in System Settings")


if __name__ == "__main__":
    _preflight()
