"""
drawlib/camera.py
OrbitCamera -- yaw/pitch/distance spherical camera with an optional elliptical
auto-orbit mode.  Decoupled from any windowing framework: the GUI calls the
input helpers (on_drag, on_scroll) and tick(dt), then reads mvp(window_size).

Usage::

    from drawlib.camera import OrbitCamera

    cam = OrbitCamera()

    # each frame
    cam.tick(frame_time)
    mvp = cam.mvp(window_size)     # (width, height) tuple

    # input forwarding
    cam.on_drag(dx, dy)
    cam.on_scroll(y_offset)

    # toggle auto-orbit
    cam.orbit_enabled = not cam.orbit_enabled
"""

import numpy as np


class OrbitCamera:
    """Spherical orbit camera with optional Lissajous auto-orbit.

    Parameters
    ----------
    yaw, pitch:
        Initial orientation in degrees.
    distance:
        Initial distance from the origin.
    fov:
        Vertical field-of-view in degrees.
    near, far:
        Clip plane distances.
    orbit_enabled:
        Start in auto-orbit mode.
    orbit_a:
        XZ semi-axis of the elliptical orbit path.
    orbit_b:
        Y semi-axis (vertical amplitude).
    orbit_speed:
        Horizontal angular speed in rad/s.
    orbit_phi:
        Vertical frequency multiplier (golden-ratio fraction by default → slow drift).
    orbit_resume_delay:
        Seconds of idle time before auto-orbit reclaims the camera after user input.
    drag_sensitivity:
        Degrees per pixel for mouse drag.
    scroll_sensitivity:
        Distance units per scroll tick.
    dist_min, dist_max:
        Clamping range for distance.
    """

    def __init__(
        self,
        yaw:                float = 35.0,
        pitch:              float = -25.0,
        distance:           float = 1.0,
        fov:                float = 55.0,
        near:               float = 0.1,
        far:                float = 100.0,
        orbit_enabled:      bool  = True,
        orbit_a:            float = 1.5,
        orbit_b:            float = 0.6,
        orbit_speed:        float = 0.22,
        orbit_phi:          float = (1 + 5 ** 0.5) / 2 * 0.5,  # ≈ 0.809
        orbit_resume_delay: float = 2.0,
        drag_sensitivity:   float = 0.4,
        scroll_sensitivity: float = 0.2,
        dist_min:           float = 1.0,
        dist_max:           float = 12.0,
    ):
        self.yaw   = yaw
        self.pitch = pitch
        self.dist  = distance

        self._fov  = np.radians(fov)
        self._near = near
        self._far  = far

        # Auto-orbit
        self.orbit_enabled      = orbit_enabled
        self._orbit_t           = 0.0
        self._orbit_a           = orbit_a
        self._orbit_b           = orbit_b
        self._orbit_speed       = orbit_speed
        self._orbit_phi         = orbit_phi
        self._orbit_resume_delay = orbit_resume_delay
        self._user_idle         = 0.0   # countdown; orbit resumes when ≤ 0

        # Input
        self._drag_sens   = drag_sensitivity
        self._scroll_sens = scroll_sensitivity
        self._dist_min    = dist_min
        self._dist_max    = dist_max

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, dt: float) -> None:
        """Advance auto-orbit by *dt* seconds (no-op when disabled or user is active)."""
        if not self.orbit_enabled:
            return
        if self._user_idle > 0.0:
            self._user_idle -= dt
            return

        self._orbit_t += dt
        ox = self._orbit_a * np.cos(self._orbit_t * self._orbit_speed)
        oy = self._orbit_b * np.sin(self._orbit_t * self._orbit_speed * self._orbit_phi)
        oz = self._orbit_a * np.sin(self._orbit_t * self._orbit_speed)
        d  = float(np.sqrt(ox * ox + oy * oy + oz * oz))

        self.dist  = d
        self.yaw   = float(np.degrees(np.arctan2(ox, oz)))
        self.pitch = float(np.degrees(np.arcsin(np.clip(oy / (d + 1e-9), -1.0, 1.0))))

    def on_drag(self, dx: float, dy: float) -> None:
        """Update yaw/pitch from a mouse drag delta."""
        if self.orbit_enabled:
            self._user_idle = self._orbit_resume_delay
        self.yaw   += dx * self._drag_sens
        self.pitch  = float(np.clip(self.pitch - dy * self._drag_sens, -89.0, 89.0))

    def on_scroll(self, y_offset: float) -> None:
        """Zoom in/out from a scroll event."""
        if self.orbit_enabled:
            self._user_idle = self._orbit_resume_delay
        self.dist = float(np.clip(
            self.dist - y_offset * self._scroll_sens,
            self._dist_min,
            self._dist_max,
        ))

    def mvp(self, window_size: tuple[int, int]) -> np.ndarray:
        """Return the column-major MVP matrix for the current frame.

        Parameters
        ----------
        window_size:
            ``(width, height)`` in pixels; used to compute the aspect ratio.
        """
        yaw   = np.radians(self.yaw)
        pitch = np.radians(self.pitch)
        d     = self.dist

        eye = np.array([
            d * np.cos(pitch) * np.sin(yaw),
            d * np.sin(pitch),
            d * np.cos(pitch) * np.cos(yaw),
        ], dtype=np.float32)

        center = np.zeros(3, dtype=np.float32)
        up     = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        f = center - eye;  f /= np.linalg.norm(f)
        r = np.cross(f, up); r /= np.linalg.norm(r)
        u = np.cross(r, f)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = r;  view[0, 3] = -r.dot(eye)
        view[1, :3] = u;  view[1, 3] = -u.dot(eye)
        view[2, :3] = -f; view[2, 3] =  f.dot(eye)

        w, h = window_size
        asp  = w / max(h, 1)
        t    = np.tan(self._fov / 2)
        near, far = self._near, self._far

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] =  1.0 / (asp * t)
        proj[1, 1] =  1.0 / t
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0

        return (proj @ view).T
