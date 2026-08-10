"""
visualizer.py
=============
Real-time 3D matplotlib visualization for air combat.

Layout:
  Left  (large):  Overview of entire battlefield
  Right (small):  Chase camera for each red aircraft (close-up)

Uses Poly3DCollection for aircraft + missile rendering.

Coordinate display: ENU (X=East, Y=North, Z=Up)
Internal coords:    NED (x=East, y=North, z=Down)
Conversion:         disp_X = x, disp_Y = y, disp_Z = -z  (no swap)
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from typing import List
from env import MISSILES_PER_AIRCRAFT




# ============================================================
#  Missile 3D model (simple elongated octahedron + fins)
# ============================================================
class MissileModel3D:
    """
    Minimal 3D missile model for rendering.

    Body frame (NED convention):
      +x = forward (nose),  +y = starboard,  +z = down (belly)
      Dorsal fin at -z (up).

    Shape: pointed nose, octagonal body, tail cone, 4 small fins.
    """

    def __init__(self, scale=30.0, color='#FFAA00', edge_color='#CC7700'):
        self.scale = scale
        self.color = color
        self.edge_color = edge_color

        # Vertices in body frame (before scaling)
        #   0  nose tip
        #   1  tail tip
        #   2  body +y (starboard)
        #   3  body -y (port)
        #   4  body -z (dorsal / up)
        #   5  body +z (ventral / down)
        #   6  fin +y tip
        #   7  fin -y tip
        #   8  fin -z tip  (dorsal fin)
        #   9  fin +z tip  (ventral fin)
        self.vertices = np.array([
            [ 0.50,  0.00,  0.00],   # 0  nose tip
            [-0.35,  0.00,  0.00],   # 1  tail tip
            [ 0.05,  0.06,  0.00],   # 2  body starboard
            [ 0.05, -0.06,  0.00],   # 3  body port
            [ 0.05,  0.00, -0.06],   # 4  body dorsal (up)
            [ 0.05,  0.00,  0.06],   # 5  body ventral (down)
            [-0.15,  0.18,  0.00],   # 6  fin starboard tip
            [-0.15, -0.18,  0.00],   # 7  fin port tip
            [-0.15,  0.00, -0.18],   # 8  fin dorsal tip (up)
            [-0.15,  0.00,  0.18],   # 9  fin ventral tip (down)
        ], dtype=float) * scale

        self.faces = [
            # Nose cone (4 triangles)
            [0, 2, 4],   # nose -> starboard -> dorsal
            [0, 4, 3],   # nose -> dorsal -> port
            [0, 3, 5],   # nose -> port -> ventral
            [0, 5, 2],   # nose -> ventral -> starboard
            # Body (4 quads connecting body to fin roots)
            [2, 6, 8],   # starboard -> finR -> dorsal
            [3, 7, 6],   # port -> finL -> finR
            [5, 9, 7],   # ventral -> finD -> finL
            [4, 8, 9],   # dorsal -> finU -> finD
            # Tail cone (4 triangles)
            [1, 8, 6],   # tail -> dorsal -> finR
            [1, 7, 8],   # tail -> finL -> dorsal
            [1, 9, 7],   # tail -> finD -> finL
            [1, 6, 9],   # tail -> finR -> finD
            # Fins (8 triangles, 2 per fin)
            [6, 2, 1],   # right fin upper
            [6, 1, 8],   # right fin lower
            [7, 3, 1],   # left fin upper
            [7, 1, 6],   # left fin lower
            [8, 4, 1],   # dorsal fin left
            [8, 1, 9],   # dorsal fin right
            [9, 5, 1],   # ventral fin left
            [9, 1, 7],   # ventral fin right
        ]

    def get_transformed(self, position, velocity):
        """
        Transform missile vertices from body frame to ENU display coords.
        Orients the missile along its velocity vector (yaw + pitch, no roll).
        """
        verts = self.vertices.copy()

        v_norm = np.linalg.norm(velocity)
        if v_norm > 1e-6:
            # Compute heading (psi) and pitch (theta) from velocity
            psi = np.arctan2(velocity[1], velocity[0])
            horizontal = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
            theta = np.arctan2(-velocity[2], horizontal)  # nose up positive

            # Pitch (theta) around Y axis
            if theta != 0:
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                verts = verts @ R.T

            # Yaw (psi) around Z axis
            if psi != 0:
                c, s = np.cos(psi), np.sin(psi)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                verts = verts @ R.T

        # NED -> ENU: flip z
        verts[:, 2] = -verts[:, 2]

        # Translate to world position (NED -> ENU)
        pos_enu = np.array([position[0], position[1], -position[2]])
        verts += pos_enu

        return verts

    def get_faces_3d(self, transformed_verts):
        """Return list of face vertex arrays for Poly3DCollection."""
        return [transformed_verts[face] for face in self.faces]


# ============================================================
#  3D Visualizer with chase cameras
# ============================================================
class Visualizer3D:
    """Real-time 3D visualization with overview + red chase cameras."""

    def __init__(self, n_red: int = 3, n_blue: int = 3):
        self.n_red = n_red
        self.n_blue = n_blue

        # Import models here to avoid circular imports
        from aircraft_models import (
            create_red_model, create_blue_model,
            get_red_model_name, get_blue_model_name,
        )

        # Create aircraft models
        self.red_models = [create_red_model(i) for i in range(n_red)]
        self.blue_models = [create_blue_model(i) for i in range(n_blue)]

        self.red_names = [get_red_model_name(i) for i in range(n_red)]
        self.blue_names = [get_blue_model_name(i) for i in range(n_blue)]

        # Team colors for missiles (derived from aircraft model colors)
        self.red_color = self.red_models[0].color
        self.red_edge = self.red_models[0].edge_color
        self.blue_color = self.blue_models[0].color
        self.blue_edge = self.blue_models[0].edge_color

        # Create missile models (one per team for color distinction)
        self.missile_red = MissileModel3D(
            scale=30.0, color=self.red_color, edge_color=self.red_edge
        )
        self.missile_blue = MissileModel3D(
            scale=30.0, color=self.blue_color, edge_color=self.blue_edge
        )

        # Trail history
        self.red_trails = [[] for _ in range(n_red)]
        self.blue_trails = [[] for _ in range(n_blue)]
        self.max_trail = 150

        # Missile data: list of (position, target_id, shooter_team, velocity)
        self.missile_data = []
        # Bullet data: list of (position, team)
        self.bullet_data = []

        # HUD element tracking (to avoid clearing axis labels on 3D axes)
        self._hud_texts: List = []
        self._chase_texts: List[List] = [[] for _ in range(n_red)]  # per chase camera

        # Time counter
        self.time = 0.0

        # Episode info
        self.episode = 0
        self.total_episodes = 0

        # ---- Layout: left=overview, right=n_red chase cameras ----
        n_closeups = n_red
        n_rows = max(n_closeups, 1)
        n_cols = 2   # left overview, right close-ups

        self.fig = plt.figure(facecolor='white')
        # Remove all figure margins so gridspec fills the entire window
        self.fig.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.0)

        gs = gridspec.GridSpec(
            n_rows, n_cols,
            width_ratios=[2.2, 1],   # overview ~70%, chase column ~30%
            hspace=0.08, wspace=0.05,
            left=0.01, right=0.99, top=0.96, bottom=0.01,
        )

        # ---- Main overview (left, spans all rows) ----
        # Initial range is just a placeholder; update_frame() adjusts dynamically
        self.ax_main = self.fig.add_subplot(gs[:, 0], projection='3d')
        self._setup_axis(self.ax_main, 'Battlefield Overview')
        self.ax_main.set_xlim(-5000, 5000)
        self.ax_main.set_ylim(-5000, 5000)
        self.ax_main.set_zlim(0, 8000)
        self.ax_main.view_init(elev=25, azim=-60)

        # ---- Chase camera axes (right column, one per red) ----
        self.ax_chase = []
        for i in range(n_closeups):
            ax = self.fig.add_subplot(gs[i, 1], projection='3d')
            name = self.red_names[i]
            self._setup_axis(ax, f'Red {name} - Chase Cam')
            ax.set_xlim(-500, 500)
            ax.set_ylim(-500, 500)
            ax.set_zlim(-500, 500)
            ax.view_init(elev=15, azim=-45)
            ax.set_facecolor('#f5f5f5')
            self.ax_chase.append(ax)

        self.fig.suptitle(
            'Air Combat Simulation',
            color='black', fontsize=11, y=0.99
        )

        # Maximize window on startup (works for TkAgg backend)
        try:
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')  # Windows maximize
        except Exception:
            pass  # non-Windows or non-TkAgg backends: just use default size

    @staticmethod
    def _safe_info_val(info, key, idx, default):
        """Safely get per-agent value from info dict (handles both list and scalar)."""
        val = info.get(key) if info else None
        if val is None:
            return default
        if isinstance(val, (list, tuple)):
            return val[idx] if idx < len(val) else default
        return val  # scalar (e.g. total count from env.step info)

    def _setup_axis(self, ax, title):
        """Configure axis appearance (white theme)."""
        ax.set_facecolor('white')
        ax.set_xlabel('East (m)', color='black', fontsize=7, labelpad=1)
        ax.set_ylabel('North (m)', color='black', fontsize=7, labelpad=1)
        ax.set_zlabel('Alt (m)', color='black', fontsize=7, labelpad=1)
        ax.tick_params(colors='black', labelsize=5, pad=0)
        ax.set_title(title, color='black', fontsize=9, pad=2)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#cccccc')
        ax.yaxis.pane.set_edgecolor('#cccccc')
        ax.zaxis.pane.set_edgecolor('#cccccc')

    # ----------------------------------------------------------
    #  Drawing helpers
    # ----------------------------------------------------------
    def _draw_aircraft_on(self, ax, model, position, phi, theta, psi):
        """Draw a single aircraft on a given axis."""
        tv = model.get_transformed_vertices(position, phi, theta, psi)
        faces_3d = model.get_faces_3d(tv)
        poly = Poly3DCollection(
            faces_3d,
            facecolors=model.color,
            edgecolors=model.edge_color,
            alpha=0.85,
            linewidth=0.5
        )
        ax.add_collection3d(poly)

    def _draw_missile_on(self, ax, missile_model, position, velocity):
        """Draw a single 3D missile on a given axis."""
        tv = missile_model.get_transformed(position, velocity)
        faces_3d = missile_model.get_faces_3d(tv)
        poly = Poly3DCollection(
            faces_3d,
            facecolors=missile_model.color,
            edgecolors=missile_model.edge_color,
            alpha=0.75,
            linewidth=0.3
        )
        ax.add_collection3d(poly)

    def _draw_trail_on(self, ax, trail, color):
        """Draw a flight trail on a given axis."""
        if len(trail) < 2:
            return
        trail_arr = np.array(trail)
        ax.plot(
            trail_arr[:, 0], trail_arr[:, 1], -trail_arr[:, 2],
            color=color, alpha=0.3, linewidth=1
        )

    def _draw_heading_line_on(self, ax, ac, color, hdg_len=600):
        """
        Draw a 3D heading indicator line from aircraft position
        along the actual velocity vector direction.
        NED velocity -> ENU display: flip z only.
        """
        vel = ac.velocity
        vel_norm = np.linalg.norm(vel)
        if vel_norm < 1e-6:
            return
        vel_dir = vel / vel_norm  # NED unit direction
        cx = ac.position[0]
        cy = ac.position[1]
        cz = -ac.position[2]
        hx = cx + hdg_len * vel_dir[0]
        hy = cy + hdg_len * vel_dir[1]
        hz = cz + hdg_len * (-vel_dir[2])
        ax.plot([cx, hx], [cy, hy], [cz, hz],
                color=color, linewidth=1.2, alpha=0.5,
                linestyle='--')

    def _draw_missiles_on(self, ax):
        """Draw all active missiles as 3D models with team colors."""
        if len(self.missile_data) == 0:
            return
        for pos, tid, team, vel, _ in self.missile_data:
            m_model = self.missile_red if team == 'red' else self.missile_blue
            self._draw_missile_on(ax, m_model, pos, vel)

    def _draw_hud(self, red_aircraft, blue_aircraft, info):
        """Draw HUD overlay on main view with colored HP bars + weapon counts.
        
        Uses per-line text2D with bbox instead of a standalone Rectangle patch
        because Axes3D.draw() sorts patches together with 3D collections and
        calls do_3d_projection() on them, which Rectangle lacks.
        
        Weapon counts (M:x/2 B:xxx) are embedded directly in the text line
        after the HP bar for compact, gap-free display.
        """
        # ---- Remove only previous HUD elements ----
        for txt in self._hud_texts:
            try:
                txt.remove()
            except ValueError:
                pass
        self._hud_texts.clear()

        lines = []   # [(text, color, killed), ...]
        episode_str = f"Ep {self.episode}/{self.total_episodes}" if self.total_episodes > 0 else ""
        lines.append((f"T={self.time:.1f}s {episode_str}", 'black', False))

        # Red team
        lines.append(("--- RED (RL) ---", 'black', False))
        for i, ac in enumerate(red_aircraft):
            hp = self._safe_info_val(info, 'red_hp', i, 100.0) if info else 100.0
            m_left = self._safe_info_val(info, 'red_missiles', i, MISSILES_PER_AIRCRAFT) if info else MISSILES_PER_AIRCRAFT
            b_left = self._safe_info_val(info, 'red_bullets', i, 200) if info else 200
            alt = -ac.position[2]
            spd = ac.speed
            alive = "ALIVE" if ac.alive else "KILLED"
            hp_bar = self._hp_bar(hp)
            lines.append((f"  {self.red_names[i]:5}: {alt:.0f}m {spd:.0f}m/s [{alive}]  "
                         f"HP:{hp_bar}  M:{int(m_left)}/{MISSILES_PER_AIRCRAFT} B:{b_left}",
                         self.red_models[i].color, not ac.alive))

        # Blue team
        lines.append(("--- BLUE (Rule) ---", 'black', False))
        for i, ac in enumerate(blue_aircraft):
            hp = self._safe_info_val(info, 'blue_hp', i, 100.0) if info else 100.0
            m_left = self._safe_info_val(info, 'blue_missiles', i, MISSILES_PER_AIRCRAFT) if info else MISSILES_PER_AIRCRAFT
            b_left = self._safe_info_val(info, 'blue_bullets', i, 200) if info else 200
            alt = -ac.position[2]
            spd = ac.speed
            alive = "ALIVE" if ac.alive else "KILLED"
            hp_bar = self._hp_bar(hp)
            lines.append((f"  {self.blue_names[i]:5}: {alt:.0f}m {spd:.0f}m/s [{alive}]  "
                         f"HP:{hp_bar}  M:{int(m_left)}/{MISSILES_PER_AIRCRAFT} B:{b_left}",
                         self.blue_models[i].color, not ac.alive))

        if info:
            a_m = info.get('active_missiles', 0)
            a_b = info.get('active_bullets', 0)
            lines.append((f"Active: {a_m} missiles  {a_b} bullets", 'black', False))

        # Per-line bbox properties (white background, no border)
        bbox_props = dict(facecolor='white', alpha=0.88, edgecolor='none',
                          boxstyle='square,pad=0.01')

        for entry in lines:
            text, color, killed = entry

            y = 0.98 - 0.01 - lines.index(entry) * 0.025

            if killed:
                t = self.ax_main.text2D(
                    0.018, y, text, transform=self.ax_main.transAxes,
                    color='#999999', verticalalignment='top', fontsize=6.5,
                    fontfamily='monospace', bbox=bbox_props)
                self._hud_texts.append(t)
                # Dim strikethrough overlay
                dash_count = max(len(text), 10)
                dash_line = '\u2500' * dash_count
                t2 = self.ax_main.text2D(
                    0.018, y, dash_line, transform=self.ax_main.transAxes,
                    color='#CC0000', verticalalignment='top', fontsize=6.5,
                    fontfamily='monospace',
                    bbox=dict(facecolor='none', alpha=0.6, edgecolor='none'))
                self._hud_texts.append(t2)
            else:
                t = self.ax_main.text2D(
                    0.018, y, text, transform=self.ax_main.transAxes,
                    color=color, verticalalignment='top', fontsize=6.5,
                    fontfamily='monospace', bbox=bbox_props)
                self._hud_texts.append(t)

    def _draw_chase_info(self, ax, idx, ac, info=None):
        """Draw aircraft info overlay on chase camera (with HP + weapon counts)."""
        # ---- Remove only previous chase info texts (NOT axis labels) ----
        for txt in self._chase_texts[idx]:
            try:
                txt.remove()
            except ValueError:
                pass
        self._chase_texts[idx].clear()

        heading_deg = np.degrees(ac.psi) % 360
        pitch_deg = np.degrees(ac.theta)
        roll_deg = np.degrees(ac.phi)
        alt = -ac.position[2]
        spd = ac.speed

        text = (
            f"HDG: {heading_deg:06.2f} deg\n"
            f"PIT: {pitch_deg:+06.2f} deg\n"
            f"ROL: {roll_deg:+06.2f} deg\n"
            f"ALT: {alt:.0f} m\n"
            f"SPD: {spd:.0f} m/s"
        )

        # HP + weapon counts appended inline (inside the same text bbox)
        if info and ac.alive:
            hp = self._safe_info_val(info, 'red_hp', idx, 100.0) if info else 100.0
            m_left = self._safe_info_val(info, 'red_missiles', idx, MISSILES_PER_AIRCRAFT) if info else MISSILES_PER_AIRCRAFT
            b_left = self._safe_info_val(info, 'red_bullets', idx, 200) if info else 200
            hp_bar = self._hp_bar(hp)
            text += (f"\nHP:{hp_bar}\n"
                     f"M:{int(m_left)}/{MISSILES_PER_AIRCRAFT}  "
                     f"B:{b_left}")

        if not ac.alive:
            text = "** DESTROYED **"

        t = ax.text2D(
            0.02, 0.98, text,
            transform=ax.transAxes,
            color='#006600', verticalalignment='top', fontsize=7,
            fontfamily='monospace',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#006600',
                      linewidth=0.5)
        )
        self._chase_texts[idx].append(t)

    def _hp_bar(self, hp: float, max_hp: float = 100.0, width: int = 10) -> str:
        """Return a Unicode HP bar string: '\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2591\u2591' """
        filled = int(round(max(0.0, hp) / max_hp * width))
        filled = max(0, min(width, filled))
        empty = width - filled
        return '\u2588' * filled + '\u2591' * empty

    # ----------------------------------------------------------
    #  Frame update
    # ----------------------------------------------------------
    def update_frame(self, red_aircraft, blue_aircraft, info=None):
        """
        Update one frame: overview + chase cameras.
        """
        self.time += 0.1

        # ---- Clear all axes ----
        all_axes = [self.ax_main] + self.ax_chase
        for ax in all_axes:
            for artist in list(ax.collections):
                artist.remove()
            for line in list(ax.lines):
                line.remove()

        # ---- Collect all positions (ENU) for dynamic bounding box ----
        all_pos_enu = []  # [(x, y, z) in ENU]
        for ac in red_aircraft:
            if ac.alive:
                all_pos_enu.append((ac.position[0], ac.position[1], -ac.position[2]))
        for ac in blue_aircraft:
            if ac.alive:
                all_pos_enu.append((ac.position[0], ac.position[1], -ac.position[2]))
        for pos, tid, team, vel, _ in self.missile_data:
            p = np.array(pos)
            all_pos_enu.append((p[0], p[1], -p[2]))
        # Bullet positions for dynamic bounding box
        for pos, team, _ in self.bullet_data:
            p = np.array(pos)
            all_pos_enu.append((p[0], p[1], -p[2]))

        # ---- Main overview: draw + dynamic scene range ----
        for i, ac in enumerate(red_aircraft):
            if ac.alive:
                self._draw_aircraft_on(
                    self.ax_main, self.red_models[i],
                    ac.position, ac.phi, ac.theta, ac.psi
                )
                self.red_trails[i].append(ac.position.copy())
                if len(self.red_trails[i]) > self.max_trail:
                    self.red_trails[i].pop(0)
                self._draw_heading_line_on(self.ax_main, ac, self.red_models[i].color)
            self._draw_trail_on(self.ax_main, self.red_trails[i], self.red_models[i].color)

        for i, ac in enumerate(blue_aircraft):
            if ac.alive:
                self._draw_aircraft_on(
                    self.ax_main, self.blue_models[i],
                    ac.position, ac.phi, ac.theta, ac.psi
                )
                self.blue_trails[i].append(ac.position.copy())
                if len(self.blue_trails[i]) > self.max_trail:
                    self.blue_trails[i].pop(0)
                self._draw_heading_line_on(self.ax_main, ac, self.blue_models[i].color)
            self._draw_trail_on(self.ax_main, self.blue_trails[i], self.blue_models[i].color)

        # Draw missiles
        self._draw_missiles_on(self.ax_main)

        # Draw bullets as tiny scatter points (realistic size)
        if len(self.bullet_data) > 0:
            pts = np.array([(p[0], p[1], -p[2]) for p, t, _ in self.bullet_data])
            self.ax_main.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c='#FF3333', s=1, alpha=0.9, marker='o', edgecolors='none')

        self._draw_hud(red_aircraft, blue_aircraft, info)

        # Dynamic overview range: fit all objects with margin
        if len(all_pos_enu) > 0:
            pts = np.array(all_pos_enu)
            x_min, y_min, z_min = pts.min(axis=0)
            x_max, y_max, z_max = pts.max(axis=0)
            # Ensure minimum span so scene doesn't collapse when objects are close
            min_span = 600.0
            span_x = max(x_max - x_min, min_span)
            span_y = max(y_max - y_min, min_span)
            span_z = max(z_max - z_min, min_span)
            # Use the largest span to keep aspect ratio equal
            max_span = max(span_x, span_y, span_z)
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            cz = (z_min + z_max) / 2.0
            half = max_span / 2.0 * 1.3  # 30% margin
            self.ax_main.set_xlim(cx - half, cx + half)
            self.ax_main.set_ylim(cy - half, cy + half)
            self.ax_main.set_zlim(cz - half, cz + half)

        # ---- Chase cameras (one per red) ----
        CHASE_RANGE = 500  # meters

        for i in range(self.n_red):
            ax = self.ax_chase[i]
            ac = red_aircraft[i]

            cx = ac.position[0]   # East
            cy = ac.position[1]   # North
            cz = -ac.position[2]  # Up (ENU)

            ax.set_xlim(cx - CHASE_RANGE, cx + CHASE_RANGE)
            ax.set_ylim(cy - CHASE_RANGE, cy + CHASE_RANGE)
            ax.set_zlim(cz - CHASE_RANGE, cz + CHASE_RANGE)

            # Draw this red aircraft (always visible)
            if ac.alive:
                self._draw_aircraft_on(
                    ax, self.red_models[i],
                    ac.position, ac.phi, ac.theta, ac.psi
                )

            # Draw nearby aircraft (within 2x range)
            all_others = list(enumerate(red_aircraft)) + [
                (j + self.n_red, bac) for j, bac in enumerate(blue_aircraft)
            ]
            for j, other in all_others:
                if j == i:
                    continue
                dist = np.linalg.norm(ac.position - other.position)
                if dist < CHASE_RANGE * 2 and other.alive:
                    if j < self.n_red:
                        model = self.red_models[j]
                    else:
                        model = self.blue_models[j - self.n_red]
                    self._draw_aircraft_on(
                        ax, model,
                        other.position, other.phi, other.theta, other.psi
                    )

            # Draw heading indicator line (3D velocity direction)
            if ac.alive:
                hdg_len = 300
                vel = ac.velocity  # NED: [vx, vy, vz]
                vel_norm = np.linalg.norm(vel)
                if vel_norm > 1e-6:
                    vel_dir = vel / vel_norm  # NED direction
                    hx = cx + hdg_len * vel_dir[0]
                    hy = cy + hdg_len * vel_dir[1]
                    hz = cz + hdg_len * (-vel_dir[2])
                    ax.plot([cx, hx], [cy, hy], [cz, hz],
                           color='#006600', linewidth=1.5, alpha=0.5)

            # Draw nearby bullets on chase camera
            for pos, team, _ in self.bullet_data:
                p = np.array(pos)
                dist = np.linalg.norm(ac.position - p)
                if dist < CHASE_RANGE * 2:
                    ax.scatter(p[0], p[1], -p[2],
                              c='#FF3333', s=1, alpha=0.9, marker='o', edgecolors='none')

            # Chase info overlay (with HP + weapons)
            self._draw_chase_info(ax, i, ac, info)

        plt.pause(0.05)

    def set_episode_info(self, episode, total_episodes):
        """Set current episode number for display."""
        self.episode = episode
        self.total_episodes = total_episodes

    def reset_trails(self):
        """Reset all flight trails for a new episode."""
        self.red_trails = [[] for _ in range(self.n_red)]
        self.blue_trails = [[] for _ in range(self.n_blue)]
        self.time = 0.0

    def update(self, env):
        """
        Update visualization from environment state.
        
        Args:
            env: CombatEnv object with red_aircraft, blue_aircraft, missile_mgr, bullet_mgr
        """
        try:
            # Get missile and bullet data from environment
            self.missile_data = env.missile_mgr.get_all_positions()
            self.bullet_data = env.bullet_mgr.get_all_positions()
            
            # Per-aircraft info for HUD and chase cameras
            info = {
                'active_missiles': env.missile_mgr.get_active_count(),
                'active_bullets':  env.bullet_mgr.get_active_count(),
                'red_hp':          list(env.red_hp),
                'blue_hp':         list(env.blue_hp),
                'red_missiles':    list(env.red_missiles_left),
                'blue_missiles':   list(env.blue_missiles_left),
                'red_bullets':     list(env.red_bullets),
                'blue_bullets':    list(env.blue_bullets),
            }
            self.update_frame(env.red_aircraft, env.blue_aircraft, info)
        except Exception as e:
            # Log the error but don't crash training - fall back to basic rendering
            import traceback
            print(f"[visualizer] ERROR in update(): {e}")
            traceback.print_exc()
            # Attempt basic rendering without HUD/bullets as fallback
            try:
                self.missile_data = env.missile_mgr.get_all_positions()
                self.bullet_data = []
                self.update_frame(env.red_aircraft, env.blue_aircraft, info=None)
            except Exception:
                pass  # last resort: just skip this frame

    def set_missile_data(self, missile_data):
        """
        Update missile data for rendering.
        Args:
            missile_data: list of (position, target_id, shooter_team, velocity, shooter_idx)
        """
        self.missile_data = missile_data

    def set_missile_positions(self, missile_positions):
        """
        Legacy API compatibility.
        Args:
            missile_positions: list of 3D position arrays (team inferred as 'red').
        """
        self.missile_data = [
            (pos, 0, 'red', np.array([1, 0, 0]), 0)
            for pos in missile_positions
        ]

    def close(self):
        """Close the figure."""
        plt.close(self.fig)

    @staticmethod
    def replay_from_json(filepath: str, fps: int = 20):
        """Replay a saved episode from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        n_red = data['n_red']
        n_blue = data['n_blue']
        frames = data['frames']

        print(f"Replaying: {n_red}v{n_blue}, {len(frames)} frames")

        viz = Visualizer3D(n_red=n_red, n_blue=n_blue)

        for frame in frames:
            from aircraft_models import AircraftState

            red_ac_list = []
            for rd in frame['red']:
                ac = AircraftState(rd['pos'], speed=rd['speed'])
                ac.phi = rd['phi']
                ac.theta = rd['theta']
                ac.psi = rd['psi']
                ac.alive = rd['alive']
                ac.velocity = np.array(rd['vel'])
                red_ac_list.append(ac)

            blue_ac_list = []
            for bd in frame['blue']:
                ac = AircraftState(bd['pos'], speed=bd['speed'])
                ac.phi = bd['phi']
                ac.theta = bd['theta']
                ac.psi = bd['psi']
                ac.alive = bd['alive']
                ac.velocity = np.array(bd['vel'])
                blue_ac_list.append(ac)

            # Legacy missile position format (no team/velocity info)
            miss_pos = [m['pos'] for m in frame.get('missiles', [])]
            viz.set_missile_positions(miss_pos)
            viz.update_frame(red_ac_list, blue_ac_list)
            plt.pause(1.0 / fps)

        print("Replay finished.")
        plt.show()
