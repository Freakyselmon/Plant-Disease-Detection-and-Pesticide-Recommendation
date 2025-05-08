# This module provides a function to get pesticide recommendations based on the disease.
# app/recommender.py 
import pandas as pd

# Load full dataset
df = pd.read_csv("data/disease_pesticide.csv")

# Normalize the disease column (remove extra spaces, lowercase, etc.)
df["Plant Disease"] = df["Plant Disease"].str.strip().str.lower()

def get_recommendation(disease_name):
    disease_name = disease_name.strip().lower()
    row = df[df["Plant Disease"] == disease_name]

    if row.empty:
        return "No pesticide recommendation found for this disease."
    
    result = row.iloc[0]
    return {
        "pesticide": result["Pesticide"],
        "dosage": result["Dosage"],
        "application_method": result["Application Method"],
        "frequency": result["Frequency"],
        "effective_against": result["Effective Against"]
    }

