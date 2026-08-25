from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from concordia.errors import ValidationError


@dataclass(frozen=True)
class DetectorObservation:
    time: float
    position: float
    density: float
    speed: float

    def __post_init__(self) -> None:
        if self.time < 0 or self.density < 0 or self.speed < 0:
            raise ValidationError("wave observations cannot contain negative time/density/speed")


@dataclass(frozen=True)
class PhantomJamEvent:
    start_time: float
    end_time: float
    minimum_speed: float
    oscillation_amplitude: float
    backward_wave_speed: float
    detector_count: int


def detect_phantom_jam(
    observations: Iterable[DetectorObservation],
    critical_density: float,
    low_speed_threshold: float,
    minimum_duration: float,
    minimum_amplitude: float,
) -> List[PhantomJamEvent]:
    """Detect sustained, backward-propagating speed collapses across fixed detectors.

    Position must increase in the downstream travel direction. A valid event needs at least
    two detector onsets, sustained high density/low speed, sufficient oscillation amplitude,
    and a fitted negative propagation speed.
    """
    if min(critical_density, low_speed_threshold, minimum_duration, minimum_amplitude) <= 0:
        raise ValidationError("phantom-jam detector thresholds must be positive")
    grouped = {}
    for observation in observations:
        grouped.setdefault(observation.position, []).append(observation)
    onsets = []
    all_event_observations: List[DetectorObservation] = []
    for position, samples in sorted(grouped.items()):
        samples.sort(key=lambda sample: sample.time)
        active = [
            sample
            for sample in samples
            if sample.density >= critical_density and sample.speed <= low_speed_threshold
        ]
        if not active:
            continue
        # Find a contiguous qualifying episode using the median sampling interval as the gap.
        times = [sample.time for sample in samples]
        intervals = np.diff(times)
        allowed_gap = float(np.median(intervals) * 1.5) if len(intervals) else minimum_duration
        episodes: List[List[DetectorObservation]] = [[active[0]]]
        for sample in active[1:]:
            if sample.time - episodes[-1][-1].time <= allowed_gap + 1e-12:
                episodes[-1].append(sample)
            else:
                episodes.append([sample])
        qualifying = next(
            (
                episode
                for episode in episodes
                if episode[-1].time - episode[0].time >= minimum_duration
            ),
            None,
        )
        if qualifying:
            onsets.append((position, qualifying[0].time))
            all_event_observations.extend(qualifying)
    if len(onsets) < 2:
        return []
    positions = np.asarray([value[0] for value in onsets], dtype=float)
    onset_times = np.asarray([value[1] for value in onsets], dtype=float)
    if np.ptp(positions) <= 0:
        return []
    slope, _ = np.polyfit(positions, onset_times, 1)
    if abs(slope) < 1e-12:
        return []
    wave_speed = float(1.0 / slope)
    speeds = [sample.speed for sample in all_event_observations]
    amplitude = max(speeds) - min(speeds)
    if wave_speed >= 0 or amplitude < minimum_amplitude:
        return []
    return [
        PhantomJamEvent(
            start_time=min(sample.time for sample in all_event_observations),
            end_time=max(sample.time for sample in all_event_observations),
            minimum_speed=min(speeds),
            oscillation_amplitude=amplitude,
            backward_wave_speed=wave_speed,
            detector_count=len(onsets),
        )
    ]
