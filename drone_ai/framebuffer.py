"""
Framebuffer writer for composite (PAL) output on Raspberry Pi.

Writes BGR frames into /dev/fb0, converting to the panel's native depth.

Critically, the fb's actual (width, height, bits-per-pixel, row stride)
are read from /sys/class/graphics/fb0/*, NOT assumed to equal the camera
capture resolution. If you write more bytes per line than the real row
stride, or more lines than the real height, the write() syscall fails
with ENOSPC ("No space left on device").
"""

import cv2
import numpy as np


class FramebufferWriter:
    """Write BGR frames to a Linux framebuffer device."""

    def __init__(self, device: str, requested_width: int, requested_height: int):
        self.device = device
        self.bpp = self._read_int("bits_per_pixel", default=16)
        self.width, self.height = self._read_virtual_size(
            requested_width, requested_height
        )
        self.bytes_per_pixel = max(1, self.bpp // 8)
        self.stride = self._read_stride(
            default=self.width * self.bytes_per_pixel
        )

        if (self.width, self.height) != (requested_width, requested_height):
            print(
                f"[fb] note: panel is {self.width}x{self.height}@{self.bpp}bpp "
                f"(stride={self.stride}B), resizing frames from "
                f"{requested_width}x{requested_height} to fit",
                file=__import__("sys").stderr,
            )

        self.fb = open(device, "r+b")
        self._rgb_scratch = None     # converted pixel buffer, no stride padding
        self._line_buffer = None     # full stride-padded buffer actually written

    @staticmethod
    def _read_int(name: str, default: int) -> int:
        """Read an integer value from /sys/class/graphics/fb0/<name>."""
        try:
            with open(f"/sys/class/graphics/fb0/{name}") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return default

    def _read_virtual_size(self, fallback_w: int, fallback_h: int):
        """Read the virtual screen size from sysfs, with fallback."""
        try:
            with open("/sys/class/graphics/fb0/virtual_size") as f:
                w_str, h_str = f.read().strip().split(",")
                return int(w_str), int(h_str)
        except (OSError, ValueError):
            return fallback_w, fallback_h

    def _read_stride(self, default: int) -> int:
        """Read the pixel stride from sysfs, with fallback."""
        return self._read_int("stride", default=default)

    def write(self, frame_bgr: np.ndarray):
        """
        Write a BGR frame to the framebuffer.

        Resizes if necessary, converts to the panel's native pixel format,
        and handles row stride padding.
        """
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(
                frame_bgr,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )

        if self.bpp == 16:
            if self._rgb_scratch is None:
                self._rgb_scratch = np.empty(
                    (self.height, self.width), dtype=np.uint16
                )
            b = (frame_bgr[:, :, 0] >> 3).astype(np.uint16)
            g = (frame_bgr[:, :, 1] >> 2).astype(np.uint16)
            r = (frame_bgr[:, :, 2] >> 3).astype(np.uint16)
            self._rgb_scratch[:] = (r << 11) | (g << 5) | b
            row_bytes = self._rgb_scratch.view(np.uint8).reshape(self.height, -1)
        elif self.bpp == 32:
            if self._rgb_scratch is None:
                self._rgb_scratch = np.empty(
                    (self.height, self.width, 4), dtype=np.uint8
                )
            self._rgb_scratch[:, :, :3] = frame_bgr
            self._rgb_scratch[:, :, 3] = 0
            row_bytes = self._rgb_scratch.reshape(self.height, -1)
        else:  # 24 bpp
            row_bytes = np.ascontiguousarray(frame_bgr).reshape(self.height, -1)

        row_len = row_bytes.shape[1]

        if self.stride == row_len:
            payload = row_bytes.tobytes()
        else:
            # Real panel has row padding (stride > pixel bytes per row) --
            # write into a stride-sized buffer so lines land at the right
            # offsets instead of drifting/wrapping.
            if (
                self._line_buffer is None
                or self._line_buffer.shape != (self.height, self.stride)
            ):
                self._line_buffer = np.zeros(
                    (self.height, self.stride), dtype=np.uint8
                )
            self._line_buffer[:, :row_len] = row_bytes
            payload = self._line_buffer.tobytes()

        try:
            self.fb.seek(0)
            self.fb.write(payload)
            self.fb.flush()
        except OSError as e:
            print(f"[fb] write error: {e}", file=__import__("sys").stderr)

    def close(self):
        """Close the framebuffer device."""
        try:
            self.fb.close()
        except OSError:
            pass
