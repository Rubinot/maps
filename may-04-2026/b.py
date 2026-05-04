import pandas as pd
import json
import numpy as np
from shapely.geometry import shape, Point
import plotly.graph_objects as go
from scipy.interpolate import griddata

def run_targeted_terrain_viz():
    # 1. SETUP & DATA LOADING
    csv_name = 'Khairahani_data.csv' 
    json_name = 'localboundries.json'

    try:
        df = pd.read_csv(csv_name)
        with open(json_name, 'r') as f:
            geojson_data = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. USER INPUT FOR TARGET COORDINATES (Source 1 Logic)
    lon_min, lon_max = df['Longitude'].min(), df['Longitude'].max()
    lat_min, lat_max = df['Latitude'].min(), df['Latitude'].max()
    
    print("\n" + "="*45)
    print("      3D TERRAIN TARGET PLOTTER")
    print("="*45)
    print(f"Data Longitude Range: {lon_min:.6f} to {lon_max:.6f}")
    print(f"Data Latitude Range:  {lat_min:.6f} to {lat_max:.6f}")
    
    try:
        user_lon = float(84.73107874631451)#input("\nEnter Target Longitude: "))
        user_lat = float(27.67947622094321)#input("Enter Target Latitude:  "))
    except ValueError:
        print("Invalid input. Please enter numeric coordinates.")
        return

    # 3. BOUNDARY IDENTIFICATION (Source 2 Logic)
    detected_muni = "Selected Area"
    max_count = -1
    final_df = pd.DataFrame()
    
    df_sample = df.sample(min(len(df), 500))
    for feature in geojson_data['features']:
        poly = shape(feature['geometry'])
        mask = df_sample.apply(lambda row: poly.contains(Point(row['Longitude'], row['Latitude'])), axis=1)
        if mask.sum() > max_count:
            max_count = mask.sum()
            detected_muni = feature['properties'].get('GaPa_NaPa') or feature['properties'].get('NAME_3') or "Area"
            full_mask = df.apply(lambda row: poly.contains(Point(row['Longitude'], row['Latitude'])), axis=1)
            final_df = df[full_mask]

    if final_df.empty:
        final_df = df
    
    # 4. INTERPOLATION & TARGET HEIGHT
    x, y, z = final_df['Longitude'].values, final_df['Latitude'].values, final_df['Altitude_m'].values
    grid_x, grid_y = np.mgrid[x.min():x.max():120j, y.min():y.max():120j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # Interpolate altitude for your specific coordinates
    user_alt = griddata((x, y), z, (user_lon, user_lat), method='linear')
    if np.isnan(user_alt):
        # Fallback to nearest point if coordinate is slightly outside triangulation
        dists = np.hypot(x - user_lon, y - user_lat)
        user_alt = z[np.argmin(dists)]

    # 5. PLOTLY 3D CONSTRUCTION
    fig = go.Figure()

    # Add the Terrain Surface
    fig.add_trace(go.Surface(
        x=grid_x[:,0], y=grid_y[0,:], z=grid_z.T, 
        colorscale='Earth', 
        name='Terrain'
    ))

    # Add the RED USER TARGET Marker (Source 1 Style)
    fig.add_trace(go.Scatter3d(
        x=[user_lon], y=[user_lat], z=[user_alt + 5], # Elevated slightly for visibility
        mode='markers+text',
        marker=dict(size=10, color='red', symbol='diamond', line=dict(width=2, color='white')),
        text=[f"Target: {user_alt:.1f}m"],
        textposition="top center",
        name='Your Target'
    ))

    # 6. ANIMATION & LAYOUT
    frames = [go.Frame(layout=dict(scene_camera=dict(eye=dict(
        x=2.0 * np.cos(np.radians(a)), 
        y=2.0 * np.sin(np.radians(a)), 
        z=0.7)))) for a in range(0, 360, 5)]

    fig.update_layout(
        title=f"3D Analysis for {detected_muni}<br>Target: {user_lon}, {user_lat}",
        template="plotly_dark",
        scene=dict(
            xaxis_title='Longitude', yaxis_title='Latitude', zaxis_title='Altitude (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4) 
        ),
        updatemenus=[dict(
            type='buttons',
            buttons=[dict(label='Play 360° View',
                          method='animate',
                          args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])]
        )]
    )

    fig.frames = frames
    fig.show()

if __name__ == "__main__":
    run_targeted_terrain_viz()
