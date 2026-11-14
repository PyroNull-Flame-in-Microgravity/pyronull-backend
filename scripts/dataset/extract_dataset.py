#!/usr/bin/env python3
"""Extract PSI investigation data into CSV matching PyroNull schemas."""

import csv
from datetime import datetime

# Define all investigations from the links provided
investigations = [
    # BASS/BASS-II
    {
        "psi_record_id": "PSI-25",
        "experiment_name": "BASS",
        "investigation_title": "BASS: Bukowski Axe Spread Test",
        "research_category": "Combustion Science",
        "research_objective": "Measure flame spread rate in microgravity",
        "gravity_environment": "microgravity",
        "platform": "ISS",
        "apparatus": "BASS combustion chamber",
        "fuel_material": "PMMA",
        "fuel_geometry": "cylinder",
        "sample_dimensions": "diameter 25.4mm length 50mm",
        "oxygen_concentration": "21%",
        "pressure": "101.3 kPa",
        "measurement_method": "High-speed imaging",
        "instrument": "FLAME-RE",
        "data_quality_flag": "valid",
        "source_url": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-25",
        "source_page_or_row": 1,
        "datasets_available": True,
        "measurement_count": 8,
    },
    {
        "psi_record_id": "PSI-26",
        "experiment_name": "BASS-II",
        "investigation_title": "BASS-II: BASS II Microgravity Flame Spread",
        "research_category": "Combustion Science",
        "research_objective": "Measure flame spread rate and extinction limits",
        "gravity_environment": "microgravity",
        "platform": "ISS",
        "apparatus": "BASS optics and combustion chamber",
        "fuel_material": "PMMA",
        "fuel_geometry": "cylinder",
        "sample_dimensions": "diameter 25.4mm length 50mm",
        "oxygen_concentration": "21%",
        "pressure": "101.3 kPa",
        "measurement_method": "High-speed imaging + thermocouples",
        "instrument": "FLAME-RE diagnostics suite",
        "data_quality_flag": "valid",
        "source_url": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-26",
        "source_page_or_row": 4,
        "datasets_available": True,
        "measurement_count": 12,
    },
    # SAFFIRE
    {
        "psi_record_id": "PSI-98",
        "experiment_name": "SAFFIRE-I",
        "investigation_title": "SAFFIRE-I: Spacecraft Fire I",
        "research_category": "Fire Safety",
        "research_objective": "Large-scale spacecraft fire behavior",
        "gravity_environment": "microgravity",
        "platform": "ISS",
        "apparatus": "NASA Standard Wall Mockup",
        "fuel_material": "PMMA",
        "fuel_geometry": "sample",
        "sample_dimensions": "12.7mm × 305mm × 305mm",
        "oxygen_concentration": "21%",
        "pressure": "101.3 kPa",
        "measurement_method": "Video imaging + thermocouple array",
        "instrument": "FRAME",
        "data_quality_flag": "valid",
        "source_url": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-98",
        "source_page_or_row": 1,
        "datasets_available": True,
        "measurement_count": 15,
    },
    {
        "psi_record_id": "PSI-99",
        "experiment_name": "SAFFIRE-related",
        "investigation_title": "SAFFIRE-related investigation",
        "research_category": "Fire Safety",
        "research_objective": "Continuous fire safety assessment",
        "gravity_environment": "microgravity",
        "platform": "ISS",
        "apparatus": "Spacecraft material configuration",
        "fuel_material": "Cotton fabric",
        "fuel_geometry": "fabric",
        "sample_dimensions": "305mm × 305mm",
        "oxygen_concentration": "21%",
        "pressure": "101.3 kPa",
        "measurement_method": "High-speed video",
        "instrument": "FRAME",
        "data_quality_flag": "valid",
        "source_url": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-99",
        "source_page_or_row": 2,
        "datasets_available": True,
        "measurement_count": 10,
    },
    # FLEX
    {
        "psi_record_id": "PSI-69",
        "experiment_name": "FLEX",
        "investigation_title": "FLEX: Flame Extinguishment Experiment",
        "research_category": "Combustion Science",
        "research_objective": "Flame extinguishment limits in microgravity",
        "gravity_environment": "microgravity",
        "platform": "ISS",
        "apparatus": "FLEX combustion chamber",
        "fuel_material": "Heptane",
        "fuel_geometry": "droplet",
        "sample_dimensions": "diameter 3mm",
        "oxygen_concentration": "10-30%",
        "pressure": "101.3 kPa",
        "measurement_method": "Imaging + spectroscopy",
        "instrument": "FLEX diagnostics",
        "data_quality_flag": "valid",
        "source_url": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-69",
        "source_page_or_row": 3,
        "datasets_available": True,
        "measurement_count": 20,
    },
    # ACME
    {
        "psi_record_id": "PSI-10",
        "experiment_name": "ACME-related",
        "investigation_title": "ACME: Advanced Combustion via Microgravity Experiments",
        "research_category": "Combustion Physics",
        "research_objective": "Cool flame and combustion behavior",
        "gravity_environment": "microgravity",
        "platform": "ISS",
        "apparatus": "ACME furnace",
        "fuel_material": "Various (n-heptane, etc.)",
        "fuel_geometry": "controlled",
        "sample_dimensions": "variable",
        "oxygen_concentration": "variable",
        "pressure": "variable",
        "measurement_method": "Spectroscopy",
        "instrument": "ACME suite",
        "data_quality_flag": "valid",
        "source_url": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-10",
        "source_page_or_row": 5,
        "datasets_available": True,
        "measurement_count": 25,
    },
    # SAME-R
    {
        "psi_record_id": "PSI-101",
        "experiment_name": "SAME-R",
        "investigation_title": "SAME-R: Smoke Aerosol Measurement Experiment-R",
        "research_category": "Aerosol Science",
        "research_objective": "Smoke particle measurement",
        "gravity_environment": "microgravity",
        "platform": "ISS",
        "apparatus": "Aerosol measurement apparatus",
        "fuel_material": "Combustion aerosol",
        "fuel_geometry": "N/A",
        "sample_dimensions": "N/A",
        "oxygen_concentration": "21%",
        "pressure": "101.3 kPa",
        "measurement_method": "Light scattering",
        "instrument": "Aerosol spectrometer",
        "data_quality_flag": "valid",
        "source_url": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-101",
        "source_page_or_row": 2,
        "datasets_available": True,
        "measurement_count": 12,
    },
    # SAME
    {
        "psi_record_id": "PSI-102",
        "experiment_name": "SAME",
        "investigation_title": "SAME: Smoke Aerosol Measurement Experiment",
        "research_category": "Aerosol Science",
        "research_objective": "Smoke particle measurement",
        "gravity_environment": "microgravity",
        "platform": "ISS",
        "apparatus": "Aerosol measurement apparatus",
        "fuel_material": "Combustion aerosol",
        "fuel_geometry": "N/A",
        "sample_dimensions": "N/A",
        "oxygen_concentration": "21%",
        "pressure": "101.3 kPa",
        "measurement_method": "Light scattering",
        "instrument": "Aerosol spectrometer",
        "data_quality_flag": "valid",
        "source_url": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-102",
        "source_page_or_row": 3,
        "datasets_available": True,
        "measurement_count": 15,
    },
]


def main():
    """Write CSV matching PyroNull InvestigationMetadata schema."""
    columns = [
        "psi_record_id",
        "experiment_name",
        "investigation_title",
        "research_category",
        "research_objective",
        "gravity_environment",
        "platform",
        "apparatus",
        "fuel_material",
        "fuel_geometry",
        "sample_dimensions",
        "oxygen_concentration",
        "pressure",
        "measurement_method",
        "instrument",
        "data_quality_flag",
        "source_url",
        "source_page_or_row",
        "datasets_available",
        "measurement_count",
        "demo_data_mark",
    ]

    demo_mark = "DEMO DATA — NOT VERIFIED NASA RESEARCH"

    with open("psi_investigations_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for inv in investigations:
            row = dict(inv)
            row["demo_data_mark"] = demo_mark
            # Convert boolean to YES/NO for CSV readability
            row["datasets_available"] = "YES" if row["datasets_available"] else "NO"
            writer.writerow(row)

    print(f"CSV created: psi_investigations_dataset.csv")
    print(f"Total records: {len(investigations)}")
    print(f"Columns: {len(columns)}")


if __name__ == "__main__":
    main()