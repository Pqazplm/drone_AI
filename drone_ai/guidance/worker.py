"""
Marker-guidance worker. Runs in its own thread, consumes snapshots
submitted via submit_tracks() from the detection loop, and (only when
explicitly enabled) writes RC channels over MSP.

Integration contract with the detection loop:
  * submit_tracks() MUST be called exactly once per INFERENCE cycle
    (never on skipped display frames).
  * captured_at MUST be a conservative timestamp taken BEFORE
    capture_array(), so capture-call duration is included in the age.
  * Only tracks with misses == 0 are considered; coasting tracks are
    deliberately ignored.
"""

import logging
import math
import threading
import time

from drone_ai.guidance.config import GuidanceConfig, Observation
from drone_ai.guidance.msp import _MSP
from drone_ai.guidance.msp import MSP_FC_VARIANT, MSP_FC_VERSION, MSP_API_VERSION

GUIDANCE_LOG = logging.getLogger("marker_guidance")


class Guidance:
    """
    Marker-guidance worker.

    Runs in its own thread, consumes snapshots submitted via submit_tracks()
    from the detection loop, and (only when explicitly enabled) writes RC
    channels over MSP.
    """

    def __init__(self, config: GuidanceConfig):
        config.validate()
        self.config = config
        self._lock = threading.Lock()
        self._snapshot = None
        self._stop = threading.Event()
        self._thread = None
        self._state = None
        self._target_id = None
        self._last_report = 0.0
        self._previous_output = None
        # Channels (r,p,y,t) as the pilot held them at the moment AUX1 went
        # high. Used as the initial seed for guidance.
        self._base_channels = None
        # Rate-limit for the "no eligible marker" diagnostic in _select.
        self._last_elig_log = 0.0
        # Last box of the currently locked target. Used by re-acquisition.
        self._last_locked_box = None

    def submit_tracks(self, tracks, captured_at):
        """
        Copy ONLY the current inference result; never retain mutable tracks.

        Must be called exactly once per INFERENCE cycle.
        """
        captured_at = float(captured_at)
        now = time.monotonic()
        if not math.isfinite(captured_at) or captured_at > now:
            GUIDANCE_LOG.error("Rejected invalid/future observation timestamp")
            return

        observations = []
        for track in tracks:
            # Carry coasting tracks through with their `misses` count so
            # _select can decide: fresh-only for initial acquisition, but
            # allow a locked target to ride out brief detection dropouts
            # up to lock_grace_misses before declaring the lock lost.
            if track.misses > self.config.lock_grace_misses:
                continue
            if int(track.class_id) != self.config.marker_class_id:
                continue
            box = tuple(float(value) for value in track.box)
            score = float(track.score)
            if (
                len(box) != 4
                or not all(math.isfinite(value) for value in box)
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                GUIDANCE_LOG.warning("Rejected malformed marker observation")
                continue

            x1, y1, x2, y2 = box
            # YOLO boxes routinely come back 1-3 px outside [0,1] when the
            # object is clipped by the frame edge. Clamping is necessary.
            cx1 = min(1.0, max(0.0, x1))
            cy1 = min(1.0, max(0.0, y1))
            cx2 = min(1.0, max(0.0, x2))
            cy2 = min(1.0, max(0.0, y2))
            if not (0.0 <= cx1 < cx2 <= 1.0 and 0.0 <= cy1 < cy2 <= 1.0):
                GUIDANCE_LOG.warning(
                    "Rejected marker box %s: no area inside frame after clamping",
                    box,
                )
                continue
            observations.append(
                Observation(
                    int(track.track_id),
                    (cx1, cy1, cx2, cy2),
                    score,
                    int(track.age),
                    int(track.misses),
                )
            )

        with self._lock:
            if self._snapshot is not None and captured_at <= self._snapshot[0]:
                return
            self._snapshot = (captured_at, tuple(observations))

    def start(self):
        """Start the guidance worker thread."""
        if self._thread is not None:
            raise RuntimeError("Guidance instances cannot be restarted")
        self._thread = threading.Thread(
            target=self._run, name="marker-guidance", daemon=False
        )
        self._thread.start()

    def close(self):
        """Stop the guidance worker thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                GUIDANCE_LOG.critical(
                    "Guidance thread has not exited; use independent pilot override"
                )

    def _transition(self, state: str, reason: str):
        """Log and record a state transition."""
        if state != self._state:
            GUIDANCE_LOG.info(
                "state=%s -> %s reason=%s", self._state, state, reason
            )
            self._state = state

    def _select(self, now: float):
        """
        Select the best eligible marker observation.

        Returns (observation, age, reason) tuple.
        """
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            return None, None, "no inference observation"
        captured_at, observations = snapshot
        age = now - captured_at
        if age > self.config.max_observation_age:
            return None, age, "stale observation"

        if self._target_id is None:
            # Initial acquisition: fresh tracks only.
            eligible = [
                item
                for item in observations
                if item.score >= self.config.minimum_confidence
                and item.age >= self.config.minimum_track_age
                and item.misses == 0
            ]
        else:
            # Already locked: allow coasting up to lock_grace_misses.
            eligible = [
                item
                for item in observations
                if item.score >= self.config.minimum_confidence
                and item.age >= self.config.minimum_track_age
                and item.misses <= self.config.lock_grace_misses
            ]

        # Re-acquisition: original track is gone beyond the grace period.
        if not eligible and self._target_id is not None and self._last_locked_box is not None:
            cfg = self.config
            if cfg.reacquire_radius > 0:
                lx1, ly1, lx2, ly2 = self._last_locked_box
                lcx = (lx1 + lx2) / 2.0
                lcy = (ly1 + ly2) / 2.0
                best = None
                best_dist = cfg.reacquire_radius ** 2
                for item in observations:
                    if item.misses != 0:
                        continue
                    if item.score < cfg.reacquire_min_confidence:
                        continue
                    bx1, by1, bx2, by2 = item.box
                    bcx = (bx1 + bx2) / 2.0
                    bcy = (by1 + by2) / 2.0
                    d2 = (bcx - lcx) ** 2 + (bcy - lcy) ** 2
                    if d2 < best_dist:
                        best_dist = d2
                        best = item
                if best is not None:
                    GUIDANCE_LOG.info(
                        "Re-acquired marker as track_id=%d score=%.3f "
                        "(was track_id=%d, dist=%.3f)",
                        best.track_id, best.score, self._target_id,
                        best_dist ** 0.5,
                    )
                    self._target_id = best.track_id
                    self._last_locked_box = best.box
                    self._previous_output = None  # re-seed from receiver
                    return best, age, ""

        if self._target_id is not None:
            for item in eligible:
                if item.track_id == self._target_id:
                    return item, age, ""
            return None, age, "locked marker missing or insufficient confidence"

        if not eligible:
            # Rate-limit the diagnostic to ~2 Hz.
            now_mono = time.monotonic()
            if now_mono - self._last_elig_log >= 0.5:
                scores = [round(o.score, 3) for o in observations]
                ages = [o.age for o in observations]
                GUIDANCE_LOG.info(
                    "no eligible marker: n_obs=%d scores=%s ages=%s "
                    "min_conf=%.2f min_age=%d",
                    len(observations), scores, ages,
                    self.config.minimum_confidence,
                    self.config.minimum_track_age,
                )
                self._last_elig_log = now_mono
            return None, age, "no confirmed marker"

        # Initial acquisition only: prefer the marker nearest the image center.
        target = min(
            eligible,
            key=lambda item: (
                ((item.box[0] + item.box[2]) / 2 - 0.5) ** 2
                + ((item.box[1] + item.box[3]) / 2 - 0.5) ** 2,
                -item.score,
            ),
        )
        self._target_id = target.track_id
        self._last_locked_box = target.box
        GUIDANCE_LOG.info(
            "Locked marker track_id=%d score=%.3f", target.track_id, target.score
        )
        return target, age, ""

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    def _command(self, target, receiver, dt):
        """Compute desired RC channels from the current target observation."""
        cfg = self.config
        x1, y1, x2, y2 = target.box

        def _deadband(value, width):
            if abs(value) <= width:
                return 0.0
            return math.copysign((abs(value) - width) / (1.0 - width), value)

        error_x = _deadband(x1 + x2 - 1.0, cfg.center_deadband)
        error_y = _deadband(y1 + y2 - 1.0, cfg.center_deadband)

        # Relative image-scale cue, NOT meters.
        approach_error = self._clamp(
            (cfg.desired_box_height - (y2 - y1)) / cfg.desired_box_height,
            0.0,
            1.0,
        )

        # Reduce forward input while the marker is off-center.
        alignment = self._clamp(
            1.0 - 2.0 * abs(error_x), 0.0, 1.0
        )
        alignment *= self._clamp(
            1.0 - 2.0 * abs(error_y), 0.0, 1.0
        )

        roll = 1500 + cfg.roll_sign * self._clamp(
            cfg.roll_gain * error_x, -cfg.roll_limit, cfg.roll_limit
        )
        pitch = 1500 + cfg.pitch_sign * min(
            cfg.pitch_gain * approach_error * alignment, cfg.pitch_limit
        )
        yaw = 1500 + cfg.yaw_sign * self._clamp(
            cfg.yaw_gain * error_x, -cfg.yaw_limit, cfg.yaw_limit
        )

        # Base throttle: prefer the value captured at AUX1 rising edge.
        base_throttle = (
            self._base_channels[3]
            if self._base_channels is not None
            else float(cfg.hover_throttle)
        )

        span_dn = cfg.hover_throttle - cfg.throttle_min
        span_up = cfg.throttle_max - cfg.hover_throttle

        throttle = self._clamp(
            base_throttle
            + cfg.throttle_sign * cfg.throttle_gain * error_y,
            max(1000.0, base_throttle - span_dn),
            min(2000.0, base_throttle + span_up),
        )

        # Remember the last locked box for re-acquisition.
        self._last_locked_box = (x1, y1, x2, y2)

        desired = [roll, pitch, yaw, throttle]
        if self._previous_output is None:
            # Seed from the captured channels so the very first
            # MSP_SET_RAW_RC matches what the pilot was already doing.
            if self._base_channels is not None:
                self._previous_output = list(self._base_channels)
            else:
                self._previous_output = [
                    1500.0, 1500.0, 1500.0, float(cfg.hover_throttle)
                ]

        step = cfg.slew_us_per_second * dt
        output = [
            previous + self._clamp(goal - previous, -step, step)
            for previous, goal in zip(self._previous_output, desired)
        ]
        self._previous_output = output

        # AUX values are carried through, never generated as mode/arm commands.
        channels = [int(round(self._clamp(value, 1000, 2000))) for value in output]
        channels.extend(receiver[4:])
        return channels, error_x, error_y, approach_error

    def _session(self, link: _MSP):
        """Main guidance session loop; runs while AUX1 is active."""
        cfg = self.config
        # MSP protocol initialization (MSPv1 command numbers):
        #   MSP_FC_VARIANT (2) → returns "BTFL" (1 byte)
        #   MSP_FC_VERSION (3) → returns version string
        #   MSP_API_VERSION (1) → returns api level (1 byte)
        variant_payload = link.request(MSP_FC_VARIANT)
        variant = variant_payload.decode("ascii", errors="replace")
        version_payload = link.request(MSP_FC_VERSION)
        api_payload = link.request(MSP_API_VERSION)
        GUIDANCE_LOG.info(
            "Connected variant=%r firmware=%r api=%r",
            variant, version_payload, api_payload,
        )
        if variant != "BTFL":
            raise RuntimeError(
                f"Expected Betaflight FC variant 'BTFL', got {variant!r}. "
                "Check MSP baud rate and Betaflight MSP mode."
            )

        # Every connection requires an observed LOW before allowing HIGH.
        ready = False
        fault = False
        aux_enabled = False
        acquire_started = None
        self._target_id = None
        self._previous_output = None
        self._base_channels = None
        self._last_locked_box = None
        self._last_valid_receiver = None
        self._transition("WAIT_AUX_LOW", "startup/reconnection interlock")

        previous_tick = time.monotonic()
        next_tick = previous_tick
        period = 1.0 / cfg.frequency_hz

        while not self._stop.is_set():
            if self._stop.wait(max(0.0, next_tick - time.monotonic())):
                return
            tick = time.monotonic()
            dt = tick - previous_tick
            previous_tick = tick

            receiver = link.read_rc()
            now = time.monotonic()

            # Covers scheduling delay AND delay obtaining receiver telemetry.
            delayed = dt > cfg.max_loop_gap or now - tick > cfg.max_loop_gap
            aux1 = receiver[4]

            if aux1 <= cfg.aux_low:
                ready = True
                fault = False
                aux_enabled = False
                acquire_started = None
                self._target_id = None
                self._previous_output = None
                self._base_channels = None
                self._last_locked_box = None
                # Cache the pilot's real channel values while AUX1 is low
                # and MSP-override is inactive. These are what we will use
                # as the base if AUX1 rises and the FC starts reporting
                # its MSP-override defaults instead of real sticks.
                if min(receiver[:4]) >= 1000:
                    self._last_valid_receiver = tuple(float(v) for v in receiver[:4])
                self._transition("STANDBY", "physical AUX1 low; no RC transmission")
            else:
                if aux1 >= cfg.aux_high:
                    if not aux_enabled:
                        # Rising edge: freeze the pilot's current channels.
                        #
                        # Caveat: the moment AUX1 rises, the FC switches
                        # into MSP-override mode and -- until it receives
                        # its FIRST MSP_SET_RAW_RC -- may report its
                        # internal defaults (885 in Betaflight) instead
                        # of the receiver values. We must break that
                        # deadlock by falling back to the last known
                        # valid MSP_RC sample cached while AUX1 was low.
                        captured = tuple(float(v) for v in receiver[:4])
                        if min(captured) < 1000.0:
                            if self._last_valid_receiver is not None:
                                GUIDANCE_LOG.warning(
                                    "AUX1 rising: FC reports defaults "
                                    "(roll=%.0f pitch=%.0f yaw=%.0f throttle=%.0f); "
                                    "using last cached valid channels "
                                    "roll=%.0f pitch=%.0f yaw=%.0f throttle=%.0f",
                                    captured[0], captured[1], captured[2], captured[3],
                                    self._last_valid_receiver[0],
                                    self._last_valid_receiver[1],
                                    self._last_valid_receiver[2],
                                    self._last_valid_receiver[3],
                                )
                                captured = self._last_valid_receiver
                            else:
                                GUIDANCE_LOG.warning(
                                    "AUX1 rising: FC reports defaults "
                                    "(roll=%.0f pitch=%.0f yaw=%.0f throttle=%.0f) "
                                    "and no cached valid sample is available; "
                                    "falling back to neutral",
                                    captured[0], captured[1], captured[2], captured[3],
                                )
                                captured = (1500.0, 1500.0, 1500.0,
                                            float(cfg.hover_throttle))
                        self._base_channels = captured
                        aux_enabled = True
                        GUIDANCE_LOG.info(
                            "AUX1 rising: captured base channels "
                            "roll=%.0f pitch=%.0f yaw=%.0f throttle=%.0f "
                            "(envelope +%d/-%d us around captured throttle)",
                            self._base_channels[0],
                            self._base_channels[1],
                            self._base_channels[2],
                            self._base_channels[3],
                            cfg.throttle_max - cfg.hover_throttle,
                            cfg.hover_throttle - cfg.throttle_min,
                        )
                    aux_enabled = True

                if not ready:
                    self._transition("WAIT_AUX_LOW", "must observe AUX1 low first")
                elif delayed:
                    fault = True
                    self._transition("FAULT", "control/telemetry deadline exceeded")
                elif fault:
                    self._transition("FAULT", "toggle AUX1 low to reset")
                elif aux_enabled:
                    if acquire_started is None:
                        acquire_started = now
                    target, observation_age, reason = self._select(now)

                    if target is None:
                        if (
                            self._target_id is not None
                            or now - acquire_started > cfg.acquire_timeout
                        ):
                            fault = True
                            self._transition("FAULT", reason)
                        else:
                            self._transition("ACQUIRING", reason)
                    else:
                        channels, ex, ey, approach = self._command(
                            target, receiver, self._clamp(dt, 0.0, cfg.max_loop_gap)
                        )

                        # Recheck observation age immediately before transmission.
                        with self._lock:
                            latest = self._snapshot
                        if (
                            latest is None
                            or observation_age + time.monotonic() - now
                            > cfg.max_observation_age
                        ):
                            fault = True
                            self._transition(
                                "FAULT", "observation expired before send"
                            )
                        elif not self._stop.is_set():
                            self._transition(
                                "DRY_RUN" if cfg.dry_run else "ACTIVE",
                                "fresh locked marker and AUX1 enabled",
                            )
                            if not cfg.dry_run:
                                link.write_rc(channels, cfg)

                            if now - self._last_report >= 0.5:
                                coast_suffix = (
                                    f" coast={target.misses}"
                                    if target.misses
                                    else ""
                                )
                                GUIDANCE_LOG.info(
                                    "state=%s target=%d age_ms=%.1f AUX1=%d "
                                    "ex=%.3f ey=%.3f approach=%.3f "
                                    "roll=%d pitch=%d yaw=%d throttle=%d%s",
                                    self._state, target.track_id,
                                    observation_age * 1000, aux1,
                                    ex, ey, approach, *channels[:4],
                                    coast_suffix,
                                )
                                self._last_report = now

            # Never send catch-up bursts.
            next_tick += period
            if next_tick < time.monotonic():
                next_tick = time.monotonic()

    def _run(self):
        """Main loop: connect, run session, reconnect on failure."""
        GUIDANCE_LOG.warning(
            "Starting dry_run=%s; target class=%d; no automatic arming. "
            "Firmware must own takeover/failsafe.",
            self.config.dry_run, self.config.marker_class_id,
        )
        while not self._stop.is_set():
            link = None
            try:
                link = _MSP(self.config)
                self._session(link)
            except Exception:
                self._transition("LINK_FAULT", "RC transmission stopped")
                GUIDANCE_LOG.exception(
                    "Guidance/MSP failure; reconnect requires AUX1-low interlock"
                )
            finally:
                if link is not None:
                    try:
                        link.close()
                    except Exception:
                        GUIDANCE_LOG.exception("Error closing MSP connection")
            self._stop.wait(1.0)

        self._transition("STOPPED", "no further RC transmissions")
        GUIDANCE_LOG.warning(
            "Stopped. No neutral/throttle/disarm packet was sent; "
            "receiver takeover/failsafe remains the flight controller's responsibility."
        )

