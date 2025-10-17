"""
Create sample Excel file for testing.

Creates a reference parts Excel file for the ManBat project.
"""
import pandas as pd
from pathlib import Path

# Sample parts for ManBat project
parts = [
    {"Part": "Head", "Description": "Main head piece", "Qty": 1},
    {"Part": "Body", "Description": "Torso/body", "Qty": 1},
    {"Part": "Left Wing", "Description": "Left wing assembly", "Qty": 1},
    {"Part": "Right Wing", "Description": "Right wing assembly", "Qty": 1},
    {"Part": "Left Arm", "Description": "Left arm", "Qty": 1},
    {"Part": "Right Arm", "Description": "Right arm", "Qty": 1},
    {"Part": "Left Leg", "Description": "Left leg", "Qty": 1},
    {"Part": "Right Leg", "Description": "Right leg", "Qty": 1},
    {"Part": "Tail", "Description": "Tail piece", "Qty": 1},
    {"Part": "Base", "Description": "Display base", "Qty": 1},
    {"Part": "Base Part", "Description": "Base component", "Qty": 10},
]

# Create DataFrame
df = pd.DataFrame(parts)

# Save to Excel
output_file = "300915_ManBat_parts.xlsx"
df.to_excel(output_file, index=False, sheet_name="Parts")

print(f"Created: {output_file}")
print(f"Parts: {len(parts)}")
print("\nContents:")
for i, part in enumerate(parts, 1):
    print(f"  {i}. {part['Part']}")

