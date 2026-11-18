"""Manim scenes for the CIET free-energy surfaces.

Render (from this directory, with the `manim` conda env):
    manim -ql ciet_scenes.py CIETStatic          # quick preview, 480p15
    manim -qm ciet_scenes.py CIETSweep           # homepage clip, 720p30

All physics lives in ciet_model.py; this file only draws.
"""

import numpy as np
from manim import (
    BLUE_D, DEGREES, ORANGE, RED_D, YELLOW,
    Create, Dot3D, FadeIn, MathTex, ParametricFunction,
    Surface, ThreeDAxes, ThreeDScene, ValueTracker, VGroup,
    always_redraw, there_and_back,
)

from ciet_model import intersection_x, mu_O, mu_R, params_from_s, saddle_point

# Plot window (model units)
X_RANGE = (-2.0, 2.0)
XI_RANGE = (-0.5, 2.5)
Z_MAX = 4.2  # heights above this are clipped (flattens the steep outer corners only)
RESOLUTION = (28, 28)

COLOR_O = BLUE_D
COLOR_R = RED_D
OPACITY = 0.65
PHI, THETA = 70, -60  # camera angles in degrees

DOWN = np.array([0, -1, 0])
LEFT_EDGE = np.array([-1, 0, 0])
UP_LEFT = np.array([-1, 1, 0])


def make_axes():
    axes = ThreeDAxes(
        x_range=[X_RANGE[0], X_RANGE[1], 1],
        y_range=[XI_RANGE[0], XI_RANGE[1], 1],
        z_range=[0, Z_MAX, 1],
        x_length=6, y_length=5, z_length=3.5,
        tips=False,
    )
    labels = axes.get_axis_labels(
        x_label=MathTex("x"), y_label=MathTex(r"\xi"), z_label=MathTex(r"\mu"),
    )
    return axes, labels


def make_surface(axes, mu, p, color):
    def point(u, v):
        return axes.c2p(u, v, min(mu(u, v, p), Z_MAX))

    return Surface(
        point,
        u_range=X_RANGE, v_range=XI_RANGE,
        resolution=RESOLUTION,
        checkerboard_colors=False,
        fill_color=color, fill_opacity=OPACITY,
        stroke_color=color, stroke_width=0.3, stroke_opacity=0.6,
    )


def make_intersection(axes, p):
    """Orange line where mu_O = mu_R (iso-energetic electron transfer)."""
    def point(xi):
        x = intersection_x(xi, p)
        return axes.c2p(x, xi, min(mu_O(x, xi, p), Z_MAX))

    # Keep only the part of the line inside the x window
    xi_vals = np.linspace(*XI_RANGE, 400)
    x_vals = intersection_x(xi_vals, p)
    inside = xi_vals[(x_vals >= X_RANGE[0]) & (x_vals <= X_RANGE[1])]
    if inside.size < 2:
        raise ValueError(
            f"Intersection line leaves the plot window for {p}; "
            "widen X_RANGE or reduce the driving force mu_R0 - mu_O0."
        )
    return ParametricFunction(
        point, t_range=[inside.min(), inside.max()], color=ORANGE, stroke_width=5,
    )


def make_saddle(axes, p):
    x, xi, m = saddle_point(p)
    return Dot3D(axes.c2p(x, xi, min(m, Z_MAX)), radius=0.13, color=YELLOW)


def setup_camera(scene, axes):
    """Point the camera at the middle of the plot box instead of the axes origin."""
    center = axes.c2p(0.0, 0.5 * (XI_RANGE[0] + XI_RANGE[1]), 0.35 * Z_MAX)
    scene.set_camera_orientation(
        phi=PHI * DEGREES, theta=THETA * DEGREES, zoom=0.95, frame_center=center,
    )


def build_landscape(axes, p):
    return VGroup(
        make_surface(axes, mu_O, p, COLOR_O),
        make_surface(axes, mu_R, p, COLOR_R),
        make_intersection(axes, p),
        make_saddle(axes, p),
    )


class CIETStatic(ThreeDScene):
    """Both surfaces at s = 1 with the intersection line and saddle point."""

    def construct(self):
        axes, labels = make_axes()
        setup_camera(self, axes)
        p = params_from_s(1.0)

        land = build_landscape(axes, p)
        self.add(axes, labels)
        self.play(Create(land[0]), Create(land[1]), run_time=2)
        self.play(Create(land[2]), FadeIn(land[3]))

        legend = VGroup(
            MathTex(r"\mu_O(x,\xi)", color=COLOR_O),
            MathTex(r"\mu_R(x,\xi)", color=COLOR_R),
        ).arrange(DOWN, aligned_edge=LEFT_EDGE).to_corner(UP_LEFT)
        self.add_fixed_in_frame_mobjects(legend)

        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(6)


class CIETSweep(ThreeDScene):
    """Sweep s = dE_IT / lam on a log scale at fixed lam + dE_IT.

    The tracker holds t = log10(s); t goes -1 -> +1 -> -1 so the clip loops.
    The labels name only the energy-scale limits (which coordinate carries the
    barrier). This toy model has the same barrier C/4 for every s, so it does
    not show the different rate laws of the ECIT and ICET limits.
    """

    T_MIN, T_MAX = -1.0, 1.0
    SWEEP_TIME = 12.0       # seconds for one full there-and-back cycle
    CAMERA_SWING = 35       # degrees the camera turns out (and back) during the cycle
    S_LOW, S_HIGH = 0.2, 5.0  # limit labels for s <= 0.2 and s >= 5
    HUD_SCALE = 0.8

    LABEL_SOLVENT = r"\lambda \gg \Delta E_{IT}:\ \text{barrier along } x \text{ (ECIT limit)}"
    LABEL_ION = r"\Delta E_{IT} \gg \lambda:\ \text{barrier along } \xi \text{ (ICET limit)}"
    LABEL_MIXED = r"\lambda \sim \Delta E_{IT}:\ \text{barrier along both}"

    def regime_label(self, tv):
        # Decide on the value as displayed (2 decimals) so label and number agree
        s_shown = round(10.0 ** tv, 2)
        if s_shown <= self.S_LOW:
            return self.LABEL_SOLVENT
        if s_shown >= self.S_HIGH:
            return self.LABEL_ION
        return self.LABEL_MIXED

    def construct(self):
        axes, labels = make_axes()
        setup_camera(self, axes)
        t = ValueTracker(self.T_MIN)

        land = always_redraw(
            lambda: build_landscape(axes, params_from_s(10.0 ** t.get_value()))
        )

        def make_hud():
            tv = t.get_value()
            return VGroup(
                MathTex(rf"s = \Delta E_{{IT}} / \lambda = {10.0 ** tv:.2f}"),
                MathTex(self.regime_label(tv)).scale(self.HUD_SCALE),
            ).arrange(DOWN, aligned_edge=LEFT_EDGE).to_corner(UP_LEFT)

        # The HUD keeps one object identity (play() caches the moving mobjects,
        # so swapping in a new object leaves an unfixed ghost of the old one).
        # become() may add new glyph submobjects that the 3D camera does not
        # know are fixed-in-frame, so the family is re-registered every frame.
        hud = make_hud()

        def refresh_hud(m):
            m.become(make_hud())
            self.renderer.camera.add_fixed_in_frame_mobjects(m)

        self.add_fixed_in_frame_mobjects(hud)
        hud.add_updater(refresh_hud)
        self.add(axes, labels, land)
        # The camera swings out and back with the same rate function as the sweep,
        # so the last frame equals the first and the clip loops without a jump.
        theta = self.renderer.camera.theta_tracker
        self.play(
            t.animate.set_value(self.T_MAX),
            theta.animate.increment_value(self.CAMERA_SWING * DEGREES),
            run_time=self.SWEEP_TIME,
            rate_func=there_and_back,
        )
        hud.clear_updaters()
