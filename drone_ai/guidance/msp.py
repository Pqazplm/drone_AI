"""
Synchronous MSPv1 transport; accessed only by the guidance thread.
"""

import logging
import struct
import time

from drone_ai.guidance.config import GuidanceConfig

GUIDANCE_LOG = logging.getLogger("marker_guidance")

# MSP command IDs (Betaflight MSPv1 protocol)
MSP_API_VERSION = 1
MSP_FC_VARIANT = 2
MSP_FC_VERSION = 3
MSP_RC = 105
MSP_SET_RAW_RC = 200


class _MSP:
    """Synchronous MSPv1 serial transport."""

    def __init__(self, config: GuidanceConfig):
        import serial

        if serial is None:
            raise RuntimeError(
                "pyserial not installed (sudo apt install python3-serial)"
            )
        self.timeout = config.serial_timeout
        self.serial = serial.Serial(
            config.port,
            baudrate=config.baudrate,
            timeout=0.005,
            write_timeout=config.serial_timeout,
        )
        self.buffer = bytearray()
        self.serial.reset_input_buffer()

    @staticmethod
    def packet(command: int, payload: bytes = b"") -> bytes:
        """Build an MSPv1 packet."""
        if len(payload) > 255:
            raise ValueError("MSPv1 payload too large")
        body = bytes((len(payload), command)) + payload
        checksum = 0
        for value in body:
            checksum ^= value
        return b"$M<" + body + bytes((checksum,))

    def _extract(self):
        """Extract one complete MSP frame from the serial buffer."""
        while True:
            position = self.buffer.find(b"$M")
            if position < 0:
                # Keep a possible partial header.
                self.buffer[:] = b"$" if self.buffer.endswith(b"$") else b""
                return None
            if position:
                del self.buffer[:position]
            if len(self.buffer) < 6:
                return None
            if self.buffer[2] not in (ord(">"), ord("!")):
                del self.buffer[0]
                continue
            length = self.buffer[3]
            total = length + 6
            if len(self.buffer) < total:
                return None
            frame = bytes(self.buffer[:total])
            checksum = 0
            for value in frame[3:-1]:
                checksum ^= value
            if checksum != frame[-1]:
                GUIDANCE_LOG.warning("MSP checksum mismatch; resynchronizing")
                del self.buffer[0]
                continue
            del self.buffer[:total]
            return frame[2], frame[4], frame[5:-1]

    def request(self, command: int, payload: bytes = b""):
        """Send an MSP request and wait for the response."""
        data = self.packet(command, payload)
        started = time.monotonic()
        written = self.serial.write(data)
        if written != len(data):
            raise IOError("Short MSP serial write")
        deadline = started + self.timeout
        while time.monotonic() < deadline:
            frame = self._extract()
            if frame is None:
                chunk = self.serial.read(
                    max(1, min(512, self.serial.in_waiting))
                )
                self.buffer.extend(chunk)
                continue
            direction, received_command, response = frame
            if received_command != command:
                GUIDANCE_LOG.debug(
                    "Ignoring unrelated MSP response %d", received_command
                )
                continue
            if direction == ord("!"):
                raise RuntimeError(f"Flight controller rejected MSP {command}")
            return response
        raise TimeoutError(f"MSP command {command} response timeout")

    def read_rc(self) -> tuple:
        """Read current RC channel values from the flight controller."""
        payload = self.request(MSP_RC)
        if len(payload) < 10 or len(payload) % 2:
            raise ValueError(f"Invalid MSP_RC payload length {len(payload)}")
        channels = struct.unpack("<" + "H" * (len(payload) // 2), payload)
        if any(not 750 <= value <= 2250 for value in channels):
            raise ValueError(f"Invalid RC channel values: {channels}")
        return channels

    def write_rc(self, logical: list, config: GuidanceConfig):
        """
        Write RC channel values to the flight controller.

        Args:
            logical: [roll, pitch, yaw, throttle, aux1, ...] in logical order.
            config: GuidanceConfig with raw_order specification.
        """
        values = dict(zip("AERT", logical[:4]))
        raw = [values[letter] for letter in config.raw_order]
        raw.extend(logical[4:])
        if len(raw) > 18:
            raise ValueError(
                "More than 18 RC channels; verify firmware support"
            )
        self.request(MSP_SET_RAW_RC, struct.pack("<" + "H" * len(raw), *raw))

    def close(self):
        """Close the serial connection."""
        self.serial.close()
