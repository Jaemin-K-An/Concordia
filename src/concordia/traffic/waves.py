from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Sequence

import numpy as np

from concordia.errors import ValidationError


class PhantomJamValidationStatus(str, Enum):
    VALID = "VALID"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    PHYSICALLY_IMPLAUSIBLE = "PHYSICALLY_IMPLAUSIBLE"
    INSUFFICIENT_DETECTORS = "INSUFFICIENT_DETECTORS"


@dataclass(frozen=True)
class DetectorObservation:
    """A fixed-detector observation in SI units.

    ``time`` is seconds from simulation start and ``position`` is metres increasing in
    the downstream travel direction. Density is vehicles/km/lane and speed is m/s.
    """

    time: float
    position: float
    density: float
    speed: float

    def __post_init__(self) -> None:
        if self.time < 0 or self.density < 0 or self.speed < 0:
            raise ValidationError("wave observations cannot contain negative time/density/speed")


@dataclass(frozen=True)
class PhantomJamEventValidation:
    status: PhantomJamValidationStatus
    detector_count: int
    regression_r_squared: float
    propagation_speed_meters_per_second: float
    propagation_speed_kilometers_per_hour: float
    propagation_speed_plausible: bool
    duration_seconds: float
    affected_length_meters: float
    oscillation_amplitude_meters_per_second: float
    minimum_speed_meters_per_second: float
    density_elevation_vehicles_per_km_per_lane: float
    queue_evidence: bool
    onset_uncertainty_seconds: float


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
    validation: PhantomJamEventValidation

    @property
    def is_valid(self) -> bool:
        return self.validation.status == PhantomJamValidationStatus.VALID


@dataclass(frozen=True)
class PhantomJamEventDetector:
    critical_density: float
    low_speed_threshold: float
    minimum_duration: float
    minimum_amplitude: float
    minimum_absolute_wave_speed_meters_per_second: float = 3.0 / 3.6
    maximum_absolute_wave_speed_meters_per_second: float = 25.0 / 3.6
    minimum_detectors: int = 3
    minimum_regression_r_squared: float = 0.60
    ewma_alpha: float = 0.80
    sustained_samples: int = 2

    def detect(self, observations: Iterable[DetectorObservation]) -> List[PhantomJamEvent]:
        return detect_phantom_jam(
            observations,
            self.critical_density,
            self.low_speed_threshold,
            self.minimum_duration,
            self.minimum_amplitude,
            minimum_absolute_wave_speed_meters_per_second=(
                self.minimum_absolute_wave_speed_meters_per_second
            ),
            maximum_absolute_wave_speed_meters_per_second=(
                self.maximum_absolute_wave_speed_meters_per_second
            ),
            minimum_detectors=self.minimum_detectors,
            minimum_regression_r_squared=self.minimum_regression_r_squared,
            ewma_alpha=self.ewma_alpha,
            sustained_samples=self.sustained_samples,
        )

    def detect_valid(self, observations: Iterable[DetectorObservation]) -> List[PhantomJamEvent]:
        return [event for event in self.detect(observations) if event.is_valid]


def _smooth(samples: Sequence[DetectorObservation], alpha: float) -> list[dict[str, float]]:
    smoothed = []
    density = samples[0].density
    speed = samples[0].speed
    for sample in samples:
        density = alpha * sample.density + (1.0 - alpha) * density
        speed = alpha * sample.speed + (1.0 - alpha) * speed
        smoothed.append(
            {
                "time": sample.time,
                "position": sample.position,
                "density": density,
                "speed": speed,
            }
        )
    return smoothed


def _qualified_episodes(
    samples: Sequence[DetectorObservation],
    critical_density: float,
    low_speed_threshold: float,
    minimum_duration: float,
    ewma_alpha: float,
    sustained_samples: int,
) -> list[dict[str, object]]:
    smoothed = _smooth(samples, ewma_alpha)
    times = np.asarray([sample.time for sample in samples], dtype=float)
    intervals = np.diff(times)
    sampling_interval = float(np.median(intervals)) if len(intervals) else minimum_duration
    allowed_gap = max(sampling_interval * 1.5, 1e-12)
    qualifies = [
        sample["density"] >= critical_density and sample["speed"] <= low_speed_threshold
        for sample in smoothed
    ]
    runs: list[list[int]] = []
    current: list[int] = []
    for index, qualified in enumerate(qualifies):
        if qualified and (
            not current or samples[index].time - samples[current[-1]].time <= allowed_gap
        ):
            current.append(index)
        elif qualified:
            if current:
                runs.append(current)
            current = [index]
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    result = []
    for run in runs:
        if len(run) < sustained_samples:
            continue
        onset_index = run[sustained_samples - 1]
        onset_time = samples[onset_index].time
        end_time = samples[run[-1]].time
        if end_time - onset_time < minimum_duration:
            continue
        result.append(
            {
                "position": samples[0].position,
                "onset": onset_time,
                "end": end_time,
                "samples": samples[run[0] : run[-1] + 1],
                "onset_uncertainty": sampling_interval * sustained_samples / 2.0,
            }
        )
    return result


def _status(
    detector_count: int,
    minimum_detectors: int,
    wave_speed: float,
    plausible: bool,
    r_squared: float,
    minimum_r_squared: float,
    queue_evidence: bool,
) -> PhantomJamValidationStatus:
    if detector_count < minimum_detectors:
        return PhantomJamValidationStatus.INSUFFICIENT_DETECTORS
    if wave_speed >= 0 or not plausible:
        return PhantomJamValidationStatus.PHYSICALLY_IMPLAUSIBLE
    if r_squared < minimum_r_squared or not queue_evidence:
        return PhantomJamValidationStatus.LOW_CONFIDENCE
    return PhantomJamValidationStatus.VALID


def detect_phantom_jam(
    observations: Iterable[DetectorObservation],
    critical_density: float,
    low_speed_threshold: float,
    minimum_duration: float,
    minimum_amplitude: float,
    *,
    minimum_absolute_wave_speed_meters_per_second: float = 3.0 / 3.6,
    maximum_absolute_wave_speed_meters_per_second: float = 25.0 / 3.6,
    minimum_detectors: int = 3,
    minimum_regression_r_squared: float = 0.60,
    ewma_alpha: float = 0.80,
    sustained_samples: int = 2,
) -> List[PhantomJamEvent]:
    """Detect and physically classify sustained stop-and-go waves.

    Onsets are estimated after EWMA smoothing and a sustained-threshold requirement. The
    propagation model is ``position_m = intercept + wave_speed_mps * onset_seconds``. Every
    multi-detector candidate is returned with an explicit validation status; scientific H3
    analyses must count only ``VALID`` events.
    """
    thresholds = (
        critical_density,
        low_speed_threshold,
        minimum_duration,
        minimum_amplitude,
        minimum_absolute_wave_speed_meters_per_second,
        maximum_absolute_wave_speed_meters_per_second,
    )
    if min(thresholds) <= 0:
        raise ValidationError("phantom-jam detector thresholds must be positive")
    if (
        minimum_absolute_wave_speed_meters_per_second
        >= maximum_absolute_wave_speed_meters_per_second
    ):
        raise ValidationError("phantom-jam wave-speed bounds are invalid")
    if minimum_detectors < 2 or not 0 <= minimum_regression_r_squared <= 1:
        raise ValidationError("phantom-jam detector-count/R-squared thresholds are invalid")
    if not 0 < ewma_alpha <= 1 or sustained_samples < 1:
        raise ValidationError("phantom-jam smoothing configuration is invalid")

    grouped: dict[float, list[DetectorObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.position, []).append(observation)
    qualified_episodes = []
    for _, samples in sorted(grouped.items()):
        samples.sort(key=lambda sample: sample.time)
        if samples:
            qualified_episodes.extend(
                _qualified_episodes(
                    samples,
                    critical_density,
                    low_speed_threshold,
                    minimum_duration,
                    ewma_alpha,
                    sustained_samples,
                )
            )
    if len(qualified_episodes) < 2:
        return []

    clusters: list[list[dict[str, object]]] = []
    for episode in sorted(qualified_episodes, key=lambda item: float(item["onset"])):
        compatible = None
        for cluster in reversed(clusters):
            if float(episode["position"]) in {
                float(item["position"]) for item in cluster
            }:
                continue
            latest = max(cluster, key=lambda item: float(item["onset"]))
            position_difference = float(latest["position"]) - float(episode["position"])
            time_difference = float(episode["onset"]) - float(latest["onset"])
            maximum_physical_delay = (
                position_difference / minimum_absolute_wave_speed_meters_per_second
                if position_difference > 0
                else -1.0
            )
            if 0 <= time_difference <= maximum_physical_delay + float(
                episode["onset_uncertainty"]
            ):
                compatible = cluster
                break
        if compatible is None:
            clusters.append([episode])
        else:
            compatible.append(episode)

    events = []
    for cluster in clusters:
        detector_count = len({float(item["position"]) for item in cluster})
        if detector_count < 2:
            continue
        positions = np.asarray([float(item["position"]) for item in cluster], dtype=float)
        onset_times = np.asarray([float(item["onset"]) for item in cluster], dtype=float)
        if np.ptp(positions) <= 0 or np.ptp(onset_times) <= 0:
            continue
        design = np.column_stack([np.ones(len(onset_times)), onset_times])
        coefficients, *_ = np.linalg.lstsq(design, positions, rcond=None)
        fitted = design @ coefficients
        residual_sum = float(np.sum((positions - fitted) ** 2))
        total_sum = float(np.sum((positions - positions.mean()) ** 2))
        r_squared = 1.0 if total_sum <= 1e-12 else max(0.0, 1.0 - residual_sum / total_sum)
        wave_speed = float(coefficients[1])
        samples = [sample for item in cluster for sample in item["samples"]]  # type: ignore[index]
        speeds = [sample.speed for sample in samples]
        densities = [sample.density for sample in samples]
        amplitude = max(speeds) - min(speeds)
        if amplitude < minimum_amplitude:
            continue
        duration = max(float(item["end"]) for item in cluster) - min(
            float(item["onset"]) for item in cluster
        )
        density_elevation = max(0.0, float(np.mean(densities)) - critical_density)
        plausible = (
            wave_speed < 0
            and minimum_absolute_wave_speed_meters_per_second
            <= abs(wave_speed)
            <= maximum_absolute_wave_speed_meters_per_second
        )
        queue_evidence = (
            min(speeds) <= low_speed_threshold
            and float(np.mean(densities)) >= critical_density
            and duration >= minimum_duration
        )
        validation_status = _status(
            detector_count,
            minimum_detectors,
            wave_speed,
            plausible,
            r_squared,
            minimum_regression_r_squared,
            queue_evidence,
        )
        onset_uncertainty = max(float(item["onset_uncertainty"]) for item in cluster)
        confidence = min(
            1.0,
            (detector_count / max(3.0, float(minimum_detectors)))
            * r_squared
            * min(1.0, duration / minimum_duration)
            * min(1.0, amplitude / minimum_amplitude),
        )
        earliest = min(cluster, key=lambda item: float(item["onset"]))
        validation = PhantomJamEventValidation(
            status=validation_status,
            detector_count=detector_count,
            regression_r_squared=r_squared,
            propagation_speed_meters_per_second=wave_speed,
            propagation_speed_kilometers_per_hour=wave_speed * 3.6,
            propagation_speed_plausible=plausible,
            duration_seconds=duration,
            affected_length_meters=float(np.ptp(positions)),
            oscillation_amplitude_meters_per_second=amplitude,
            minimum_speed_meters_per_second=min(speeds),
            density_elevation_vehicles_per_km_per_lane=density_elevation,
            queue_evidence=queue_evidence,
            onset_uncertainty_seconds=onset_uncertainty,
        )
        events.append(
            PhantomJamEvent(
                start_time=min(float(item["onset"]) for item in cluster),
                end_time=max(float(item["end"]) for item in cluster),
                minimum_speed=min(speeds),
                oscillation_amplitude=amplitude,
                backward_wave_speed=wave_speed,
                detector_count=detector_count,
                origin_position=float(earliest["position"]),
                affected_length=float(np.ptp(positions)),
                confidence=confidence,
                validation=validation,
            )
        )
    return events
