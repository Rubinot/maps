import pandas as pd
import json
import numpy as np
from shapely.geometry import shape, Point
import plotly.graph_objects as go
from scipy.interpolate import griddata

def run_topography_surface_viz():
    # 1. Inputs
    csv_name = 'Rapti_data.csv' #input("Enter the name of your CSV file (e.g., Rapti_data.csv): ")
    json_name = 'localboundries.json'

    try:
        df = pd.read_csv(csv_name)
        with open(json_name, 'r') as f:
            geojson_data = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Municipality Identification
    detected_muni = "Unknown"
    max_count = -1
    final_df = pd.DataFrame()
    
    # Quick sample check to find the boundary
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
        print("No matching boundary found. Using full dataset.")
        final_df = df
    
    print(f"Generating smooth terrain for: {detected_muni}...")

    # 3. INTERPOLATION (Turning points into a surface)
    x = final_df['Longitude'].values
    y = final_df['Latitude'].values
    z = final_df['Altitude_m'].values

    # Create a dense grid (100x100)
    grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]

    # Interpolate the Z values (altitude) across the grid
    # Using 'cubic' for that smooth curve look
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # 4. Create 3D Surface Plot
    fig = go.Figure(data=[go.Surface(
        x=grid_x[:,0], 
        y=grid_y[0,:], 
        z=grid_z.T, 
        colorscale='earth', # FIXED: Swapped 'Terrain' for 'earth'
        colorbar=dict(title="Altitude (m)")
    )])

    # 5. Animation Setup (360 Rotation)
    frames = [go.Frame(layout=dict(scene_camera=dict(eye=dict(
        x=1.8 * np.cos(np.radians(a)), 
        y=1.8 * np.sin(np.radians(a)), 
        z=0.7)))) for a in range(0, 360, 5)]

    fig.update_layout(
        title=f"3D Surface Topography: {detected_muni}",
        scene=dict(
            xaxis_title='Lon',
            yaxis_title='Lat',
            zaxis_title='Alt (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5) 
        ),
        updatemenus=[dict(
            type='buttons',
            buttons=[dict(label='Play 360° Rotation',
                          method='animate',
                          args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])]
        )]
    )

    fig.frames = frames
    print("Opening 3D model in your browser...")
    fig.show()

if __name__ == "__main__":
    run_topography_surface_viz()
