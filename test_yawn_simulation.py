"""
Automated Simulation Test Suite for 4-Stage Yawn Detection.
Validates the 4-phase lifecycle, peak hold dwell time, and 2.5s-10.5s duration constraints.
"""

import sys
import numpy as np
from config import DetectorConfig
from yawn_tracker import YawnDetector, YawnStage


def generate_mar_curve(duration: float, peak_mar: float, baseline_mar: float = 0.10,
                       peak_hold_ratio: float = 0.30, fps: int = 30):
    """
    Generate synthetic MAR values following a natural yawn curve:
    - Progressive Opening (Rise)
    - Fully Opening (Plateau / Peak hold)
    - Progressive Closing (Fall)
    """
    total_frames = int(duration * fps)
    timestamps = [i / fps for i in range(total_frames)]
    rise_pct = (1.0 - peak_hold_ratio) / 2.0
    fall_pct = rise_pct

    mar_values = []
    for i in range(total_frames):
        p = i / max(1, total_frames - 1)
        if p < rise_pct:
            w = np.sin((p / rise_pct) * (np.pi / 2))
            mar = baseline_mar + (peak_mar - baseline_mar) * w
        elif p <= (1.0 - fall_pct):
            mar = peak_mar
        else:
            fall_p = (p - (1.0 - fall_pct)) / fall_pct
            w = np.cos(fall_p * (np.pi / 2))
            mar = baseline_mar + (peak_mar - baseline_mar) * w
        mar_values.append(mar)
    return timestamps, mar_values


def run_simulation(name: str, duration: float, peak_mar: float, expected_result: bool,
                   peak_hold_ratio: float = 0.30, inject_reopening: bool = False,
                   custom_mar: list = None) -> bool:
    print(f"\n--- Testing Scenario: {name} ---")
    print(f"Target Duration: {duration:.2f}s | Peak MAR: {peak_mar:.2f} | Expected Yawn: {expected_result}")

    config = DetectorConfig(enable_audio_alert=False)
    detector = YawnDetector(config)

    fps = 30
    if custom_mar is not None:
        mar_values = list(custom_mar)
        timestamps = [i / fps for i in range(len(mar_values))]
    else:
        timestamps, mar_values = generate_mar_curve(
            duration=duration, peak_mar=peak_mar, peak_hold_ratio=peak_hold_ratio, fps=fps
        )

    detected_yawn = False
    stages_visited = set()

    # If test requires simulating speech re-opening during closing
    if inject_reopening:
        closing_idx = int(len(mar_values) * 0.90)
        mar_values[closing_idx:closing_idx + 8] = [0.62] * 8

    for t, mar in zip(timestamps, mar_values):
        stage, is_new = detector.update(mar, current_time=t)
        if stage != YawnStage.IDLE:
            stages_visited.add(stage)
        if is_new:
            detected_yawn = True
            print(f"  [EVENT] Yawn Confirmed at t={t:.2f}s! Total duration: {detector.last_yawn_duration:.2f}s")

    # Feed baseline frames at the end to allow full closing confirmation
    end_t = timestamps[-1]
    for i in range(15):
        t = end_t + (i + 1) / fps
        stage, is_new = detector.update(0.10, current_time=t)
        if stage != YawnStage.IDLE:
            stages_visited.add(stage)
        if is_new:
            detected_yawn = True
            print(f"  [EVENT] Yawn Confirmed at t={t:.2f}s! Total duration: {detector.last_yawn_duration:.2f}s")

    print(f"Stages visited during cycle: {[s.name for s in sorted(stages_visited, key=lambda x: x.value)]}")
    print(f"Total Yawn Count: {detector.yawn_count}")
    print(f"Final Status: {detector.last_status_message}")

    passed = (detected_yawn == expected_result)
    status_str = "PASS [OK]" if passed else "FAIL [X]"
    print(f"Result: {status_str}")
    return passed


def main():
    print("=" * 68)
    print("  AUTOMATED TEST SUITE: 4-STAGE YAWN DETECTION (2.5s-10.5s WINDOW)")
    print("=" * 68)

    results = []

    # Scenario 1: Standard Natural Yawn (5.2s, peak MAR 0.78) -> MUST PASS
    r1 = run_simulation(
        name="Standard Natural Yawn (5.2s, 4-stage lifecycle)",
        duration=5.2,
        peak_mar=0.78,
        expected_result=True
    )
    results.append(("Standard Natural Yawn (5.2s)", r1))

    # Scenario 2: Valid Quick Yawn (2.6s, peak MAR 0.74, Case 1 model) -> MUST PASS
    r2 = run_simulation(
        name="Valid Quick Yawn (2.6s >= 2.4s cutoff - Case 1)",
        duration=2.6,
        peak_mar=0.74,
        peak_hold_ratio=0.35, # 0.9s peak hold
        expected_result=True
    )
    results.append(("Valid Quick Yawn (2.6s)", r2))

    # Scenario 3: Valid Deep/Extended Yawn (9.5s, peak MAR 0.85 - Case 2 model) -> MUST PASS
    r3 = run_simulation(
        name="Valid Deep Yawn (9.5s <= 10.5s cutoff - Case 2)",
        duration=9.5,
        peak_mar=0.85,
        peak_hold_ratio=0.45, # 4.2s peak hold
        expected_result=True
    )
    results.append(("Valid Deep Yawn (9.5s)", r3))

    # Scenario 4: Fast Movement (< 2.0s: laugh/cough/quick speech) -> MUST REJECT
    r4 = run_simulation(
        name="Too Fast Movement (1.5s < 2.4s minimum)",
        duration=1.5,
        peak_mar=0.75,
        expected_result=False
    )
    results.append(("Fast Movement (< 2.4s rejected)", r4))

    # Scenario 5: Speech Exclamation Spike (0.20s momentary peak < 0.50s) -> MUST REJECT
    frames_pre = [0.10] * 15 # 0.5s baseline
    frames_rise = [0.25, 0.45, 0.68]
    frames_peak = [0.72] * 6 # 6 frames at 30fps = 0.20s (< 0.50s min hold)
    frames_fall = [0.45, 0.25]
    frames_post = [0.10] * 30
    speech_frames = frames_pre + frames_rise + frames_peak + frames_fall + frames_post
    r5 = run_simulation(
        name="Speech Exclamation Spike (Peak held 0.20s < 0.50s)",
        duration=len(speech_frames) / 30.0,
        peak_mar=0.72,
        expected_result=False,
        custom_mar=speech_frames
    )
    results.append(("Speech Exclamation Spike (< 0.5s hold rejected)", r5))

    # Scenario 6: Speech Re-Opening During Closing (Phone call speech) -> MUST REJECT
    r6 = run_simulation(
        name="Speech Re-opening During Closing Phase (Phone conversation)",
        duration=5.0,
        peak_mar=0.75,
        expected_result=False,
        inject_reopening=True
    )
    results.append(("Speech Re-opening During Closing (Rejected)", r6))

    # Scenario 7: Shallow Opening (Conversational speech peak MAR 0.38 < 0.65) -> MUST REJECT
    r7 = run_simulation(
        name="Shallow Opening (Speech peak MAR 0.38)",
        duration=4.5,
        peak_mar=0.38,
        expected_result=False
    )
    results.append(("Shallow Opening (Peak never reached)", r7))

    print("\n" + "=" * 68)
    print("  TEST SUMMARY")
    print("=" * 68)
    all_passed = True
    for test_name, passed in results:
        mark = "PASS" if passed else "FAIL"
        print(f"  {mark.ljust(6)} | {test_name}")
        if not passed:
            all_passed = False

    print("-" * 68)
    if all_passed:
        print("  ALL 7 SCENARIOS PASSED PERFECTLY!")
        sys.exit(0)
    else:
        print("  SOME TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
