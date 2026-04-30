"""
terrain_3d_map.py
=================
Universal animated 3-D terrain mapper.

  • Accepts ANY CSV with longitude, latitude & altitude columns
  • Uses Shapely to compute the true dataset boundary
  • Animates:
      1. Camera rising and doing a full 360° orbit
      2. A glowing red marker falling from the sky to the target
      3. Shapely-built ripple rings pulsing outward on landing

Usage
-----
    python terrain_3d_map.py                   # interactive prompts
    python terrain_3d_map.py data.csv          # skip CSV prompt
    python terrain_3d_map.py data.csv --save   # force-save to terrain_map.gif

Requirements
------------
    pip install pandas numpy matplotlib scipy shapely pillow
"""

import sys
import os

# ── Backend detection MUST happen before any pyplot import ─────────
import matplotlib

_INTERACTIVE_BACKENDS = [
    "TkAgg", "Qt5Agg", "QtAgg", "Qt6Agg",
    "Qt4Agg", "WXAgg", "GTK3Agg", "GTK4Agg", "MacOSX",
]

def _pick_backend(force_save: bool = False) -> bool:
    """
    Try each interactive backend. Returns True if one works,
    False if we must fall back to Agg (save-to-GIF mode).
    """
    if force_save:
        matplotlib.use("Agg")
        return False

    for b in _INTERACTIVE_BACKENDS:
        try:
            matplotlib.use(b)
            import matplotlib.pyplot as _plt
            fig = _plt.figure()
            _plt.close(fig)
            return True
        except Exception:
            continue

    matplotlib.use("Agg")
    return False


# Parse --save flag before pyplot is ever imported
_FORCE_SAVE   = "--save" in sys.argv
if _FORCE_SAVE:
    sys.argv.remove("--save")

_INTERACTIVE = _pick_backend(force_save=_FORCE_SAVE)

# Safe to import now
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import MultiPoint, Point


# ══════════════════════════════════════════════════════════════════
# 1. CSV LOADING
# ══════════════════════════════════════════════════════════════════

def smart_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Map any lon/lat/alt column variant to standard names."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    mapping = {}
    for col in df.columns:
        lc = col.replace("_", "").replace(" ", "")
        if lc in ("longitude","lon","long","lng","x") and "lon" not in mapping.values():
            mapping[col] = "lon"
        elif lc in ("latitude","lat","y") and "lat" not in mapping.values():
            mapping[col] = "lat"
        elif lc in ("altitudem","altitude","alt","elevation","elev",
                    "height","z","dem","dtm") and "alt" not in mapping.values():
            mapping[col] = "alt"
    return df.rename(columns=mapping)


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        sys.exit(f"\n[ERROR] File not found: {csv_path}\n")
    df = smart_rename(pd.read_csv(csv_path))
    missing = [c for c in ("lon", "lat", "alt") if c not in df.columns]
    if missing:
        print(f"\n[ERROR] Could not auto-detect columns: {missing}")
        print(f"  Found: {list(df.columns)}")
        print("  Rename your columns to: longitude, latitude, altitude\n")
        sys.exit(1)
    df = df[["lon", "lat", "alt"]].dropna()
    print(f"  ✓  {len(df):,} valid data points loaded.")
    return df


# ══════════════════════════════════════════════════════════════════
# 2. USER INPUT
# ══════════════════════════════════════════════════════════════════

def ask_csv() -> str:
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║          Universal 3-D Terrain Animator              ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    return input("  CSV file path : ").strip().strip('"').strip("'")


def ask_coordinate(df: pd.DataFrame) -> tuple:
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    print()
    print("  Dataset extent")
    print(f"    Longitude : {lon_min:.6f}  →  {lon_max:.6f}")
    print(f"    Latitude  : {lat_min:.6f}  →  {lat_max:.6f}")
    print()
    while True:
        try:
            lon_in = float(input("  Target Longitude : "))
            lat_in = float(input("  Target Latitude  : "))
            if not (lon_min <= lon_in <= lon_max and lat_min <= lat_in <= lat_max):
                print("\n  ⚠  Outside extent — will still be plotted.\n")
            return lon_in, lat_in
        except ValueError:
            print("  [!] Enter numeric values.\n")


def interpolate_altitude(df: pd.DataFrame, lon: float, lat: float) -> float:
    alt = griddata(
        (df["lon"].values, df["lat"].values), df["alt"].values,
        (lon, lat), method="linear"
    )
    val = float(np.atleast_1d(alt)[0])
    if np.isnan(val):
        dists = np.hypot(df["lon"] - lon, df["lat"] - lat)
        val   = float(df.loc[dists.idxmin(), "alt"])
    return val


# ══════════════════════════════════════════════════════════════════
# 3. SHAPELY GEOMETRY
# ══════════════════════════════════════════════════════════════════

def compute_boundary(df: pd.DataFrame, buffer_deg: float = 0.002):
    pts    = MultiPoint(list(zip(df["lon"].values, df["lat"].values)))
    hull   = pts.convex_hull.buffer(buffer_deg)
    coords = np.array(hull.exterior.coords)
    return hull, coords[:, 0], coords[:, 1]


def mask_outside_boundary(glon, glat, galt, hull):
    inside = np.array([
        hull.contains(Point(lo, la))
        for lo, la in zip(glon.ravel(), glat.ravel())
    ]).reshape(glon.shape)
    out = galt.copy().astype(float)
    out[~inside] = np.nan
    return out


# ══════════════════════════════════════════════════════════════════
# 4. GRID INTERPOLATION
# ══════════════════════════════════════════════════════════════════

def build_grid(df: pd.DataFrame, hull, resolution: int = 130):
    lon_lin = np.linspace(df["lon"].min(), df["lon"].max(), resolution)
    lat_lin = np.linspace(df["lat"].min(), df["lat"].max(), resolution)
    glon, glat = np.meshgrid(lon_lin, lat_lin)

    galt = griddata(
        (df["lon"].values, df["lat"].values), df["alt"].values,
        (glon, glat), method="linear"
    )
    galt_nn = griddata(
        (df["lon"].values, df["lat"].values), df["alt"].values,
        (glon, glat), method="nearest"
    )
    galt[np.isnan(galt)] = galt_nn[np.isnan(galt)]
    return glon, glat, mask_outside_boundary(glon, glat, galt, hull)


# ══════════════════════════════════════════════════════════════════
# 5. ANIMATION
# ══════════════════════════════════════════════════════════════════

TOTAL_FRAMES = 300
ORBIT_START  = 60
DROP_START   = 40
DROP_END     = 110
RIPPLE_START = 110
FPS          = 30


def animate(df, glon, glat, galt, bx, by,
            user_lon, user_lat, user_alt, dataset_name: str,
            interactive: bool = True):

    alt_valid = galt[~np.isnan(galt)]
    alt_min, alt_max = alt_valid.min(), alt_valid.max()
    alt_range = alt_max - alt_min
    z_floor   = alt_min - alt_range * 0.10

    norm_alt  = (galt - alt_min) / (alt_range + 1e-9)
    face_cols = cm.terrain(norm_alt)
    face_cols[np.isnan(galt)] = [0.04, 0.06, 0.10, 0.0]

    # ── Figure ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor="#060a10")
    ax  = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor("#060a10")

    ax.plot_surface(glon, glat, galt,
                    facecolors=face_cols, linewidth=0,
                    antialiased=True, alpha=0.95, shade=True, zorder=1)

    ax.plot(bx, by, z_floor,
            color="#00e5ff", linewidth=1.0, alpha=0.55, zorder=2)

    ax.contourf(glon, glat, galt,
                zdir="z", offset=z_floor,
                levels=14, cmap="terrain", alpha=0.25, zorder=0)

    sm = cm.ScalarMappable(cmap="terrain")
    sm.set_array(galt)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.40, pad=0.01, aspect=18)
    cbar.set_label("Altitude (m)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cbar.outline.set_edgecolor("#444444")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#060a10")
        axis.pane.set_edgecolor("#1a2535")
        axis._axinfo["grid"]["color"] = "#1a2535"
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xlabel("Longitude",    color="white", labelpad=10, fontsize=8)
    ax.set_ylabel("Latitude",     color="white", labelpad=10, fontsize=8)
    ax.set_zlabel("Altitude (m)", color="white", labelpad=10, fontsize=8)
    ax.set_zlim(z_floor, alt_max + alt_range * 0.18)

    title = fig.suptitle(
        f"3-D Terrain  ·  {dataset_name}",
        color="white", fontsize=12, y=0.97, fontfamily="monospace"
    )

    # ── Dynamic artists ────────────────────────────────────────────
    sky_z = alt_max + alt_range * 0.90

    marker_plt = ax.scatter([], [], [], s=0,
                            color="#ff3030", depthshade=False,
                            edgecolors="white", linewidths=1.2, zorder=10)
    pole_line, = ax.plot([], [], [],
                         color="#ff3030", linewidth=1.8, alpha=0.0, zorder=9)

    # 2-D overlay avoids matplotlib Text3D compatibility issues
    label_box = ax.text2D(
        0.72, 0.88, "",
        transform=ax.transAxes,
        color="white", fontsize=8.5, fontfamily="monospace",
        alpha=0.0, linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="#160000", edgecolor="#ff4040", alpha=0.80),
    )

    ripple_lines = [
        ax.plot([], [], [], color="#ff5050", linewidth=0.9, alpha=0.0, zorder=8)[0]
        for _ in range(4)
    ]

    ax.text2D(
        0.02, 0.03,
        f"Target  Lon {user_lon:.6f}   Lat {user_lat:.6f}   Alt {user_alt:.1f} m",
        transform=ax.transAxes,
        color="#777777", fontsize=7.5, fontfamily="monospace",
    )

    def ripple_ring(cx, cy, cz, radius):
        coords = np.array(Point(cx, cy).buffer(radius).exterior.coords)
        return coords[:, 0], coords[:, 1], np.full(len(coords), cz)

    def update(frame):
        # Camera orbit
        if frame < ORBIT_START:
            t    = frame / ORBIT_START
            elev = 15 + 20 * t
            azim = -65
        else:
            t    = (frame - ORBIT_START) / (TOTAL_FRAMES - ORBIT_START)
            elev = 35 + 5 * np.sin(t * 2 * np.pi)
            azim = -65 + 360 * t
        ax.view_init(elev=elev, azim=azim)

        # Marker drop
        if frame < DROP_START:
            marker_plt._sizes = np.array([0])
            pole_line.set_alpha(0.0)
            label_box.set_alpha(0.0)

        elif frame <= DROP_END:
            t_drop = (frame - DROP_START) / (DROP_END - DROP_START)
            t_ease = t_drop ** 2
            cur_z  = sky_z - (sky_z - user_alt) * t_ease
            glow   = min(1.0, t_drop * 3)

            marker_plt._offsets3d = (
                np.array([user_lon]),
                np.array([user_lat]),
                np.array([cur_z]),
            )
            marker_plt._sizes = np.array([80 + 160 * glow])
            pole_line.set_data_3d(
                [user_lon, user_lon],
                [user_lat, user_lat],
                [user_alt, cur_z],
            )
            pole_line.set_alpha(0.55 * glow)
            label_box.set_alpha(0.0)

        else:
            marker_plt._offsets3d = (
                np.array([user_lon]),
                np.array([user_lat]),
                np.array([user_alt + alt_range * 0.018]),
            )
            marker_plt._sizes = np.array([240])
            pole_line.set_data_3d(
                [user_lon, user_lon],
                [user_lat, user_lat],
                [user_alt, user_alt + alt_range * 0.018],
            )
            pole_line.set_alpha(0.90)
            label_box.set_text(
                f"Lon : {user_lon:.6f}\n"
                f"Lat : {user_lat:.6f}\n"
                f"Alt : {user_alt:.1f} m"
            )
            label_box.set_alpha(1.0)

        # Ripple rings
        if frame >= RIPPLE_START:
            t_rip    = (frame - RIPPLE_START) / (TOTAL_FRAMES - RIPPLE_START + 1)
            lon_span = df["lon"].max() - df["lon"].min()
            for i, line in enumerate(ripple_lines):
                phase  = i / len(ripple_lines)
                age    = (t_rip - phase) % 1.0
                radius = age * lon_span * 0.25
                alpha  = max(0.0, 0.70 * (1.0 - age * 1.65))
                if alpha > 0.01:
                    rx, ry, rz = ripple_ring(user_lon, user_lat, user_alt, radius)
                    line.set_data_3d(rx, ry, rz)
                    line.set_alpha(alpha)
                else:
                    line.set_data_3d([], [], [])
                    line.set_alpha(0.0)
        else:
            for line in ripple_lines:
                line.set_data_3d([], [], [])
                line.set_alpha(0.0)

        return [marker_plt, pole_line, label_box, title, *ripple_lines]

    # Keep reference — prevents garbage collection before rendering
    ani = animation.FuncAnimation(
        fig, update,
        frames=TOTAL_FRAMES,
        interval=1000 // FPS,
        blit=False,
    )

    plt.tight_layout()

    if interactive:
        print("\n  ▶  Animation running — close the window to exit.")
        print("     (Drag to rotate manually once the animation loops)\n")
        plt.show()          # blocks until window is closed
    else:
        out = "terrain_map.gif"
        print(f"\n  Saving animation →  {out}  …")
        ani.save(out, writer=animation.PillowWriter(fps=FPS))
        print(f"  ✓  Done: {out}\n")

    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print()
        print("╔══════════════════════════════════════════════════════╗")
        print("║          Universal 3-D Terrain Animator              ║")
        print("╚══════════════════════════════════════════════════════╝")
    else:
        csv_path = ask_csv()

    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    print(f"\n  Loading  {csv_path} …")
    df = load_data(csv_path)

    user_lon, user_lat = ask_coordinate(df)
    user_alt = interpolate_altitude(df, user_lon, user_lat)

    print()
    print("  📍 Target coordinate")
    print(f"     Longitude : {user_lon}")
    print(f"     Latitude  : {user_lat}")
    print(f"     Altitude  : {user_alt:.1f} m  (interpolated)")

    print("\n  Building Shapely boundary …")
    hull, bx, by = compute_boundary(df)

    print("  Interpolating terrain grid …")
    glon, glat, galt = build_grid(df, hull, resolution=130)

    if not _INTERACTIVE:
        print("\n  [INFO] No interactive display found.")
        print("         Animation will be saved as  terrain_map.gif")
        print("         (Re-run with a desktop session or pass --save to suppress this message)")

    print("\n  Starting animation …")
    animate(df, glon, glat, galt, bx, by,
            user_lon, user_lat, user_alt, dataset_name,
            interactive=_INTERACTIVE)


if __name__ == "__main__":
    main()
