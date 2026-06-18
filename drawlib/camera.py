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
    # or via the mode string property
    cam.mode = "static"   # "auto_orbit" | "static"

    # lerp to a pose
    cam.lerp_to(yaw=90.0, pitch=-15.0, dist=3.0, duration=2.0)
"""

import numpy as np


def _short_angle(a: float, b: float) -> float:
    """Signed shortest delta from angle a to b (both in degrees)."""
    return (b - a + 180.0) % 360.0 - 180.0


class OrbitCamera:
    """Spherical orbit camera with optional Lissajous auto-orbit and pose lerp.

    Modes
    -----
    "auto_orbit"  (orbit_enabled=True, default)
        tick() drives yaw/pitch/dist from a Lissajous path controlled by
        orbit_speed, orbit_a, orbit_b, and orbit_phi.
    "static"  (orbit_enabled=False)
        yaw/pitch/dist are held at whatever value is set externally.

    lerp_to(yaw, pitch, dist, duration)
        Overrides both modes for the duration of the transition; on completion
        the camera stays at the target and the previous mode resumes.

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
        XZ semi-axis of the elliptical orbit path (left-right distance amplitude).
    orbit_b:
        Y semi-axis (vertical / up-down amplitude).
    orbit_speed:
        Horizontal angular speed in rad/s (left-right orbit frequency).
    orbit_phi:
        Up-down frequency multiplier — effective vertical frequency is
        orbit_speed * orbit_phi (default ≈ 0.809, golden-ratio drift).
    orbit_resume_delay:
        Seconds of idle time before auto-orbit reclaims the camera after user input.
    lerp_duration:
        Default lerp duration in seconds used by lerp_to() when no duration is given.
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
        orbit_phi:          float = (1 + 5 ** 0.5) / 2 * 0.5,
        orbit_resume_delay: float = 2.0,
        lerp_duration:      float = 1.0,
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
        self.orbit_enabled       = orbit_enabled
        self._orbit_t            = 0.0
        self.orbit_speed         = orbit_speed
        self.orbit_a             = orbit_a
        self.orbit_b             = orbit_b
        self.orbit_phi           = orbit_phi
        self._orbit_resume_delay = orbit_resume_delay
        self._user_idle          = 0.0

        # Lerp
        self.lerp_duration = lerp_duration
        self._lerp_active  = False
        self._lerp_t       = 0.0
        self._lerp_src     = (yaw, pitch, distance)
        self._lerp_dst     = (yaw, pitch, distance)
        self._lerp_dur     = lerp_duration

        # Input
        self._drag_sens   = drag_sensitivity
        self._scroll_sens = scroll_sensitivity
        self._dist_min    = dist_min
        self._dist_max    = dist_max

    # ── Mode ──────────────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Camera drive mode: "auto_orbit" or "static"."""
        return "auto_orbit" if self.orbit_enabled else "static"

    @mode.setter
    def mode(self, value: str) -> None:
        self.orbit_enabled = (value == "auto_orbit")

    # ── Lerp ──────────────────────────────────────────────────────────────────

    def lerp_to(
        self,
        yaw:      float,
        pitch:    float,
        dist:     float,
        duration: float | None = None,
    ) -> None:
        """Smoothly transition to (yaw, pitch, dist) over *duration* seconds.

        While the lerp is active it overrides both static and auto-orbit modes.
        On completion the camera holds the target pose and the previous mode resumes.
        """
        self._lerp_src = (self.yaw, self.pitch, self.dist)
        self._lerp_dst = (yaw, pitch, dist)
        self._lerp_dur = duration if duration is not None else self.lerp_duration
        self._lerp_t   = 0.0
        self._lerp_active = True

    # ── Per-frame update ──────────────────────────────────────────────────────

    def tick(self, dt: float) -> None:
        """Advance camera state by dt seconds."""
        if self._lerp_active:
            self._lerp_t += dt
            t   = min(self._lerp_t / max(self._lerp_dur, 1e-6), 1.0)
            t_s = t * t * (3.0 - 2.0 * t)  # smoothstep ease

            sy, sp, sd = self._lerp_src
            dy, dp, dd = self._lerp_dst
            self.yaw   = sy + _short_angle(sy, dy) * t_s
            self.pitch = sp + (dp - sp) * t_s
            self.dist  = sd + (dd - sd) * t_s

            if t >= 1.0:
                self._lerp_active = False
            return

        if not self.orbit_enabled:
            return  # static mode — hold current yaw/pitch/dist

        if self._user_idle > 0.0:
            self._user_idle -= dt
            return

        self._orbit_t += dt
        ox = self.orbit_a * np.cos(self._orbit_t * self.orbit_speed)
        oy = self.orbit_b * np.sin(self._orbit_t * self.orbit_speed * self.orbit_phi)
        oz = self.orbit_a * np.sin(self._orbit_t * self.orbit_speed)
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

    def position_and_axes(self) -> tuple:
        """Return (eye, forward, right, up) as float32 unit vectors.

        eye     -- camera position in world space
        forward -- unit vector pointing into the scene (eye toward origin)
        right   -- unit vector to the camera's right
        up      -- unit vector upward in camera space
        """
        yaw   = np.radians(self.yaw)
        pitch = np.radians(self.pitch)
        d     = self.dist

        eye = np.array([
            d * np.cos(pitch) * np.sin(yaw),
            d * np.sin(pitch),
            d * np.cos(pitch) * np.cos(yaw),
        ], dtype=np.float32)

        fwd = -eye / (np.linalg.norm(eye) + 1e-9)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(fwd, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = (right / rn).astype(np.float32)

        up = np.cross(right, fwd).astype(np.float32)
        return eye, fwd, right, up

    def mvp(self, window_size: tuple) -> np.ndarray:
        """Return the column-major MVP matrix for the current frame."""
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
