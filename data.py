import pandas as pd
import random

# Define 50 Karnataka places (updated to reflect the correct number of unique places)
karnataka_places = [
    "Bangalore", "Mysore", "Hubli", "Mangalore", "Belgaum", "Davangere", "Ballari",
    "Gulbarga", "Bijapur", "Shimoga", "Tumkur", "Raichur", "Hassan", "Mandya",
    "Chitradurga", "Udupi", "Chikmagalur", "Bagalkot", "Hospet", "Bidar",
    "Chikballapur", "Kolar", "Gadag", "Haveri", "Karwar", "Koppal", "Yadgir",
    "Kodagu", "Ramanagara", "Sirsi", "Dharwad", "Chamarajanagar", "Bhadravati",
    "Madikeri", "Sakleshpur", "Tiptur", "Sira", "Doddaballapur", "Kanakapura",
    "Malur", "Hunsur", "Kollegal", "Arsikere", "Nanjangud", "Muddebihal",
    "Athani", "Gokak", "Ilkal", "Jamkhandi", "Ramdurg", "Saundatti", "Banavasi",
    "Basavana Bagewadi", "Kundapur", "Ankola", "Byndoor", "Siruguppa", "Sindhanur",
    "Pavagada", "Chintamani", "Mulbagal", "Kundgol", "Nargund", "Ron", "Yellapur",
    "Shiggaon", "Savanur", "Hangal", "Terdal", "Mudhol", "Navalgund", "Harapanahalli",
    "Sandur", "Mudalgi", "Gokarna", "Honnavar", "Siddapura", "Dandeli", "Kalghatgi",
    "Haliyal", "Mundgod", "Kumta", "Sirsi", "Karkala", "Kundapura", "Bhatkal", 
    "Hebri", "Padubidri", "Surathkal", "Shivamogga", "Tirthahalli", "Bhadravathi"
]

# Since there are only 50 unique places, we will sample all of them
synthetic_places = karnataka_places  # Use all the available places

# Define crops and their price range (per quintal)
crops = {
    "RICE": (2000, 3000),
    "WHEAT": (1800, 2500),
    "SUGARCANE": (300, 500),
    "COTTON": (4000, 6000),
    "MAIZE": (1500, 2200),
    "GROUNDNUT": (3500, 5000),
    "MOONG": (4000, 7000),
    "ARHAR": (5000, 7500)
}

# Define year range and rainfall range (in mm)
years = list(range(2019, 2024))
rainfall_range = (400, 1200)

# Generate synthetic dataset
synthetic_data = []
for place in synthetic_places:
    for year in random.sample(years, 5):  # Select 5 random years per place
        crop = random.choice(list(crops.keys()))
        price = random.randint(*crops[crop])
        rainfall = random.randint(*rainfall_range)
        synthetic_data.append({
            "Place": place,
            "Crop": crop,
            "Crop Price (per Quintal)": price,
            "Year": year,
            "Average Rainfall (mm)": rainfall
        })

# Convert to DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# Save to CSV
synthetic_output_path = "karnataka_synthetic_crop_data.csv"
synthetic_df.to_csv(synthetic_output_path, index=False)

print(f"Dataset created and saved as {synthetic_output_path}")
