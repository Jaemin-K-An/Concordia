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
    origin_position: float
    affected_length: float
    confidence: float


@dataclass(frozen=True)
class PhantomJamEventDetector:
    critical_density: float
    low_speed_threshold: float
    minimum_duration: float
    minimum_amplitude: float

    def detect(self, observations: Iterable[DetectorObservation]) -> List[PhantomJamEvent]:
        return detect_phantom_jam(
            observations,
            self.critical_density,
            self.low_speed_threshold,
            self.minimum_duration,
            self.minimum_amplitude,
        )


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
    qualified_episodes = []
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
        for episode in episodes:
            if episode[-1].time - episode[0].time >= minimum_duration:
                qualified_episodes.append(
                    {
                        "position": position,
                        "onset": episode[0].time,
                        "end": episode[-1].time,
                        "samples": episode,
                    }
                )
    if len(qualified_episodes) < 2:
        return []
    association_window = max(minimum_duration * 3.0, 1.0)
    clusters = []
    for episode in sorted(qualified_episodes, key=lambda item: item["onset"]):
        compatible = next(
            (
                cluster
                for cluster in reversed(clusters)
                if episode["onset"] - max(item["onset"] for item in cluster)
                <= association_window
                and episode["position"] not in {item["position"] for item in cluster}
            ),
            None,
        )
        if compatible is None:
            clusters.append([episode])
        else:
            compatible.append(episode)
    events = []
    for cluster in clusters:
        if len({item["position"] for item in cluster}) < 2:
            continue
        positions = np.asarray([item["position"] for item in cluster], dtype=float)
        onset_times = np.asarray([item["onset"] for item in cluster], dtype=float)
        if np.ptp(positions) <= 0:
            continue
        slope, _ = np.polyfit(positions, onset_times, 1)
        if abs(slope) < 1e-12:
            continue
        wave_speed = float(1.0 / slope)
        samples = [sample for item in cluster for sample in item["samples"]]
        speeds = [sample.speed for sample in samples]
        amplitude = max(speeds) - min(speeds)
        if wave_speed >= 0 or amplitude < minimum_amplitude:
            continue
        earliest = min(cluster, key=lambda item: item["onset"])
        duration = max(item["end"] for item in cluster) - min(
            item["onset"] for item in cluster
        )
        confidence = min(
            1.0,
            (len(cluster) / 3.0)
            * min(1.0, duration / minimum_duration)
            * min(1.0, amplitude / minimum_amplitude),
        )
        events.append(
            PhantomJamEvent(
                start_time=min(item["onset"] for item in cluster),
                end_time=max(item["end"] for item in cluster),
                minimum_speed=min(speeds),
                oscillation_amplitude=amplitude,
                backward_wave_speed=wave_speed,
                detector_count=len(cluster),
                origin_position=float(earliest["position"]),
                affected_length=float(np.ptp(positions)),
                confidence=confidence,
            )
        )
    return events
