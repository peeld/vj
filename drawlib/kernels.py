"""
kernels.py
Warp GPU kernels -- compiled once at import time.
"""

import warp as wp


@wp.kernel
def color_by_interaction(
    positions:   wp.array(dtype=wp.vec3),
    colors:      wp.array(dtype=wp.vec4),
    ball_pos:    wp.array(dtype=wp.vec3),
    ball_colors: wp.array(dtype=wp.vec4),
    radius:      float,
):
    """Blend each point's color toward any ball within radius. Color persists."""
    i = wp.tid()
    p = positions[i]
    c = colors[i]

    for b in range(ball_pos.shape[0]):
        bp = ball_pos[b]
        dx = p[0] - bp[0]
        dy = p[1] - bp[1]
        dz = p[2] - bp[2]
        dist2 = dx*dx + dy*dy + dz*dz
        r2    = radius * radius

        if dist2 < r2 and dist2 > 0.00001:
            dist = wp.sqrt(dist2)
            t    = 1.0 - dist / radius   # 0 at edge, 1 at center
            bc   = ball_colors[b]
            c    = wp.vec4(
                c[0] * (1.0 - t) + bc[0] * t,
                c[1] * (1.0 - t) + bc[1] * t,
                c[2] * (1.0 - t) + bc[2] * t,
                0.3,
            )

    colors[i] = c


@wp.kernel
def influence_points(
    positions:  wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    ball_pos:   wp.array(dtype=wp.vec3),
    radius:     float,
    strength:   float,
    dt:         float,
):
    """Repel each point away from any ball within radius."""
    i = wp.tid()
    p = positions[i]
    v = velocities[i]

    vx = v[0]; vy = v[1]; vz = v[2]

    for b in range(ball_pos.shape[0]):
        bp = ball_pos[b]
        dx = p[0] - bp[0]
        dy = p[1] - bp[1]
        dz = p[2] - bp[2]
        dist2 = dx*dx + dy*dy + dz*dz
        r2    = radius * radius

        if dist2 < r2 and dist2 > 0.00001:
            dist   = wp.sqrt(dist2)
            factor = strength * (1.0 - dist / radius) * dt / dist
            vx = vx + dx * factor
            vy = vy + dy * factor
            vz = vz + dz * factor

    velocities[i] = wp.vec3(vx, vy, vz)


@wp.kernel
def step_points(
    positions:  wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt:         float,
    damping:    float,
):
    """Integrate position and apply velocity damping."""
    i = wp.tid()
    p = positions[i]
    v = velocities[i]

    vx = v[0] * damping
    vy = v[1] * damping
    vz = v[2] * damping

    positions[i]  = wp.vec3(p[0] + vx*dt, p[1] + vy*dt, p[2] + vz*dt)
    velocities[i] = wp.vec3(vx, vy, vz)


@wp.kernel
def respawn_escaped(
    positions:  wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    colors:     wp.array(dtype=wp.vec4),
    half:       float,
    vel_thresh: float,
    seed:       int,
):
    """Respawn points that escaped the cube and have near-zero velocity."""
    i = wp.tid()
    p = positions[i]
    v = velocities[i]

    outside = (wp.abs(p[0]) > half) or (wp.abs(p[1]) > half) or (wp.abs(p[2]) > half)
    slow    = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) < vel_thresh * vel_thresh

    if outside and slow:
        r0 = wp.rand_init(seed, i * 3 + 0)
        r1 = wp.rand_init(seed, i * 3 + 1)
        r2 = wp.rand_init(seed, i * 3 + 2)
        nx = (wp.randf(r0) * 2.0 - 1.0) * half
        ny = (wp.randf(r1) * 2.0 - 1.0) * half
        nz = (wp.randf(r2) * 2.0 - 1.0) * half
        positions[i]  = wp.vec3(nx, ny, nz)
        velocities[i] = wp.vec3(0.0, 0.0, 0.0)
        colors[i]     = wp.vec4(0.5, 0.5, 0.5, 0.1)


@wp.kernel
def bounce_balls(
    positions:  wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt:         float,
    half:       float,
):
    i = wp.tid()
    p = positions[i]
    v = velocities[i]

    px = p[0] + v[0] * dt
    py = p[1] + v[1] * dt
    pz = p[2] + v[2] * dt

    vx = v[0]; vy = v[1]; vz = v[2]

    if px >= half:
        px = half;  vx = -vx
    if px <= -half:
        px = -half; vx = -vx
    if py >= half:
        py = half;  vy = -vy
    if py <= -half:
        py = -half; vy = -vy
    if pz >= half:
        pz = half;  vz = -vz
    if pz <= -half:
        pz = -half; vz = -vz

    positions[i]  = wp.vec3(px, py, pz)
    velocities[i] = wp.vec3(vx, vy, vz)
