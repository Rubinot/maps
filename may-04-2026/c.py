import json
import pandas as pd

def analyze_nepal_data(file_path):
    # Load the GeoJSON data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract properties from each feature in the collection
    records = [feature['properties'] for feature in data['features']]
    df = pd.DataFrame(records)

    # 1. Calculate number of Districts within each STATE_CODE/Province
    state_summary = df.groupby(['STATE_CODE', 'Province']).agg(
        No_of_Districts=('DISTRICT', 'nunique')
    ).reset_index()

    # 2. Calculate no. of GaPa_NaPa and Type_GN counts within each DISTRICT
    district_detail = df.groupby(['STATE_CODE', 'DISTRICT', 'Type_GN']).agg(
        No_of_GaPa_NaPa=('GaPa_NaPa', 'count')
    ).reset_index()

    # Merge state info into district detail for a complete view
    final_report = pd.merge(district_detail, state_summary, on='STATE_CODE')

    # Display the results
    print("--- Administrative Data Summary ---")
    print(final_report[['STATE_CODE', 'Province', 'No_of_Districts', 'DISTRICT', 'Type_GN', 'No_of_GaPa_NaPa']])

# Usage
# analyze_nepal_data('your_file.json')
