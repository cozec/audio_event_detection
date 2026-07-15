"""Step 7: Live macOS inference demo.

Runs the streaming detector (step 3) on the exported TFLite models
(step 4/5): YAMNet f32 backbone + INT8 head. Audio comes from the
Mac microphone by default, or from a wav file with --file (replayed
in real time, or as fast as possible with --fast).

    python mic_demo.py                       # live mic (grant mic access)
    python mic_demo.py --file some.wav       # replay a file in real time
    python mic_demo.py --file some.wav --fast

Prints a live top-3 probability readout each hop and an EVENT line
when the decision layer fires.
"""

import argparse
import queue
import sys
import time

import numpy as np

from config import RESULTS_DIR, SAMPLE_RATE, SELECTED_CLASSES
from streaming_inference import HOP_S, StreamingDetector, WINDOW
from benchmark_tflite import make_interpreter, run_tflite, yamnet_embed

CHUNK = 1024


class _Arr(np.ndarray):
    """ndarray that also answers .numpy(), mimicking a tf tensor."""

    def numpy(self):
        return np.asarray(self)


class TFLiteYamnet:
    """Adapter: TFLite backbone with the hub-model call signature."""

    def __init__(self):
        self.interp = make_interpreter("yamnet_backbone_f32.tflite", [WINDOW])

    def __call__(self, window):
        emb = yamnet_embed(self.interp, np.asarray(window, dtype=np.float32))
        return None, emb.view(_Arr), None


class TFLiteHead:
    """Adapter: TFLite INT8 head with the Keras call signature."""

    def __init__(self):
        self.interp = make_interpreter("head_int8.tflite")

    def __call__(self, emb, training=False):
        return np.stack([run_tflite(self.interp, e[np.newaxis])[0] for e in emb])


def render(detector, events_total):
    """One-line live readout of the top-3 smoothed class probabilities."""
    t, smoothed = detector.posteriors[-1]
    top = np.argsort(smoothed)[::-1][:3]
    parts = [
        f"{detector.class_names[c]:>15s} {'█' * int(smoothed[c] * 20):<20s}"
        f"{smoothed[c]:4.2f}"
        for c in top
    ]
    sys.stdout.write(
        f"\r{t:7.1f}s | {' | '.join(parts)} | events: {events_total} "
    )
    sys.stdout.flush()


def stream_mic(detector):
    """Feed the detector from the default input device until Ctrl-C."""
    import sounddevice as sd

    q = queue.Queue()

    def callback(indata, frames, t, status):
        if status:
            print(f"\n[audio status] {status}", file=sys.stderr)
        q.put(indata[:, 0].copy())

    events_total = 0
    print(f"Listening on: {sd.query_devices(kind='input')['name']} "
          f"(Ctrl-C to stop)")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=CHUNK, dtype="float32", callback=callback):
        while True:
            for ev in detector.feed(q.get()):
                events_total += 1
                print(f"\n*** EVENT  {ev['class']:<18s} p={ev['prob']:.2f} "
                      f"at {ev['time_s']:.1f}s ***\a")
            if detector.posteriors:
                render(detector, events_total)


def stream_file(detector, path, fast):
    """Replay a wav through the detector, real-time paced unless fast."""
    import librosa

    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    print(f"Replaying {path} ({len(wav) / SAMPLE_RATE:.1f}s)"
          f"{' as fast as possible' if fast else ''}")
    events_total = 0
    t_start = time.time()
    for i in range(0, len(wav), CHUNK):
        if not fast:
            target = i / SAMPLE_RATE
            lag = target - (time.time() - t_start)
            if lag > 0:
                time.sleep(lag)
        for ev in detector.feed(wav[i:i + CHUNK]):
            events_total += 1
            print(f"\n*** EVENT  {ev['class']:<18s} p={ev['prob']:.2f} "
                  f"at {ev['time_s']:.1f}s ***")
        if detector.posteriors:
            render(detector, events_total)
    print(f"\nDone: {events_total} event(s), "
          f"median inference {np.median(detector.infer_ms):.1f} ms/hop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="replay a wav instead of using the mic")
    parser.add_argument("--fast", action="store_true",
                        help="with --file: no real-time pacing")
    args = parser.parse_args()

    print("Loading TFLite models (YAMNet f32 backbone + INT8 head) ...")
    detector = StreamingDetector(
        TFLiteYamnet(), TFLiteHead(), sorted(SELECTED_CLASSES)
    )

    try:
        if args.file:
            stream_file(detector, args.file, args.fast)
        else:
            stream_mic(detector)
    except KeyboardInterrupt:
        print(f"\nStopped. {len([1 for t, _ in detector.posteriors])} hops, "
              f"median inference "
              f"{np.median(detector.infer_ms):.1f} ms/hop")


if __name__ == "__main__":
    main()
