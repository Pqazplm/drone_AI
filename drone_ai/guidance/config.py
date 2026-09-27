"""
Guidance configuration dataclasses and helper functions.
"""

import math
from dataclasses import dataclass


def _clamp(value, low, high):
    return max(low, min(high, value))


def _deadband(value, width):
    if abs(value) <= width:
        return 0.0
    return math.copysign((abs(value) - width) / (1.0 - width), value)


@dataclass(frozen=True)
class GuidanceConfig:
    """Configuration for the marker guidance system."""

    port: str
    marker_class_id: int
    raw_order: str
    hover_throttle: int
    throttle_min: int
    throttle_max: int

    dry_run: bool = True
    primary_override_verified: bool = False
    baudrate: int = 115200  # Usually nominal for USB CDC.
    frequency_hz: float = 25.0
    serial_timeout: float = 0.08
    max_loop_gap: float = 0.20

    aux_low: int = 1300
    aux_high: int = 1700
    max_observation_age: float = 0.35
    acquire_timeout: float = 3.0
    minimum_confidence: float = 0.4
    minimum_track_age: int = 1

    desired_box_height: float = 0.25
    center_deadband: float = 0.02

    # Microseconds of channel correction per unit normalized image error.
    roll_gain: float = 25.0
    yaw_gain: float = 70.0
    pitch_gain: float = 55.0
    throttle_gain: float = 30.0

    roll_limit: float = 30.0
    yaw_limit: float = 80.0
    pitch_limit: float = 60.0

    roll_sign: int = 1
    pitch_sign: int = 1
    yaw_sign: int = 1
    throttle_sign: int = -1

    # Slew limiting applies to commanded channels while control is active.
    slew_us_per_second: float = 400.0

    # How many consecutive missed inference cycles a LOCKED marker may
    # coast (using its last known box) before guidance declares FAULT.
    lock_grace_misses: int = 6

    # When a locked track is lost beyond lock_grace_misses, guidance
    # will look for a fresh detection of the same class within
    # reacquire_radius (normalized frame units) of the last known
    # position before declaring FAULT. 0 disables re-acquisition.
    reacquire_radius: float = 0.25

    # Score threshold for the re-acquisition candidate.
    reacquire_min_confidence: float = 0.5

    def validate(self):
        """Validate configuration values. Raises ValueError on invalid config."""
        if sorted(self.raw_order) != sorted("AETR"):
            raise ValueError("raw_order must be a permutation of AETR")
        if self.marker_class_id < 0:
            raise ValueError("marker_class_id must be nonnegative")
        if not 1000 <= self.throttle_min <= self.hover_throttle <= self.throttle_max <= 2000:
            raise ValueError("Invalid throttle bounds/calibration")
        if not 1 <= self.frequency_hz <= 50:
            raise ValueError("frequency_hz must be between 1 and 50")
        if not 0 < self.serial_timeout < self.max_loop_gap:
            raise ValueError("Invalid serial timeout / loop-gap limit")
        if self.max_loop_gap <= 1 / self.frequency_hz:
            raise ValueError("max_loop_gap must exceed the command period")
        if not 0 < self.desired_box_height < 1:
            raise ValueError("desired_box_height must be normalized to (0,1)")
        if not 0 <= self.center_deadband < 1:
            raise ValueError("Invalid center_deadband")
        if not 1000 <= self.aux_low < self.aux_high <= 2000:
            raise ValueError("Invalid AUX thresholds")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("Invalid minimum_confidence")
        if self.minimum_track_age < 1:
            raise ValueError("minimum_track_age must be positive")
        if self.reacquire_radius < 0:
            raise ValueError("reacquire_radius must be non-negative")
        if not 0 <= self.reacquire_min_confidence <= 1:
            raise ValueError("Invalid reacquire_min_confidence")
        if min(
            self.max_observation_age,
            self.acquire_timeout,
            self.slew_us_per_second,
        ) <= 0:
            raise ValueError("Age, acquisition and slew limits must be positive")
        if min(
            self.roll_gain, self.pitch_gain, self.yaw_gain, self.throttle_gain,
            self.roll_limit, self.pitch_limit, self.yaw_limit,
        ) < 0:
            raise ValueError("Gains and limits must be nonnegative")
        for direction in (
            self.roll_sign, self.pitch_sign, self.yaw_sign, self.throttle_sign
        ):
            if direction not in (-1, 1):
                raise ValueError("Channel signs must be +1 or -1")
        if not self.dry_run and not self.primary_override_verified:
            raise ValueError(
                "Live operation requires verified primary-only MSP override"
            )


@dataclass(frozen=True)
class Observation:
    """A single marker observation from the tracker."""
    track_id: int
    box: tuple
    score: float
    age: int
    misses: int = 0
