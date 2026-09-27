"""
Marker guidance over MSP (Betaflight) protocol.

MSP connectivity alone does not establish safe RC arbitration. See the
safety warning at the top of the main file and in the --help for --guidance.

Dependencies:
    sudo apt install python3-serial

OPERATION
---------
* AUX1 must first be LOW after startup.
* Raising AUX1 selects a fresh, confirmed marker-class track.
* The selected track ID remains locked until AUX1 goes LOW.
* A missing/stale selected marker, serial error or excessive scheduling
  delay latches a fault. Toggle AUX1 LOW before another attempt.
* A fault stops RC transmissions. It does NOT send minimum throttle or
  claim to land the aircraft. Actual takeover/failsafe is firmware-owned.
* Marker class IDs refer to YOUR model. The supplied person/car classes
  are not marker classes. Use a marker-trained model and update --classes.

The default age limit may reject the Zero 2W's slow detections. That is
intentional. Do not increase it merely to hide inference latency.

Logs include state transitions, observation age, target ID, desired
channels, MSP timeouts, protocol errors and exception tracebacks.
"""

from drone_ai.guidance.config import GuidanceConfig, Observation
from drone_ai.guidance.msp import _MSP, MSP_FC_VARIANT, MSP_FC_VERSION, MSP_API_VERSION
from drone_ai.guidance.worker import Guidance
from drone_ai.guidance.logging_setup import configure_logging

__all__ = [
    "Guidance",
    "GuidanceConfig",
    "Observation",
    "configure_logging",
]
