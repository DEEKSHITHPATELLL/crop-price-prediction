from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg') 
app = Flask(__name__, static_folder='static')
model_path = r'C:\Users\Deekshith Patel L L\OneDrive\Desktop\Dsp Project\server\model\crop_data.pkl'
data_path = r'C:\Users\Deekshith Patel L L\OneDrive\Desktop\Dsp Project\server\data\crop_data.csv'
if os.path.exists(model_path):
    print("Model found, loading...")
    model = joblib.load(model_path)
else:
    print(f"Model file not found at {model_path}")
    model = None  
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    print(f"Dataset file not found at {data_path}")
    data = pd.DataFrame()
if not data.empty:
    label_encoder_place = LabelEncoder()
    label_encoder_crop = LabelEncoder()
    label_encoder_place.fit(data['Place'])
    label_encoder_crop.fit(data['Crop'])
else:
    label_encoder_place = label_encoder_crop = None
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

@app.route('/')
def index():
    """Render the homepage with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    if model is None or data.empty:
        return "Error: Model or dataset not loaded. Please check the file paths and try again."
    place = request.form.get('place', '').capitalize()
    crop = request.form.get('crop', '').upper()
    year = int(request.form.get('year', 0))
    rainfall = float(request.form.get('rainfall', 0))
    month_str = request.form.get('month', 'January') 
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    month = month_mapping.get(month_str, 1)  

    print(f"User Input - Place: {place}, Crop: {crop}, Year: {year}, Rainfall: {rainfall}, Month: {month}")
    try:
        place_encoded = label_encoder_place.transform([place])[0]
        crop_encoded = label_encoder_crop.transform([crop])[0]
    except ValueError as e:
        return f"Error: Invalid input value. {str(e)}"

    print(f"Encoded Place: {place_encoded}, Encoded Crop: {crop_encoded}")
    input_data = pd.DataFrame({
        'Place': [place_encoded],
        'Crop': [crop_encoded],
        'Year': [year],
        'Average Rainfall (mm)': [rainfall],
        'Month': [month]  
    })
    try:
        predicted_price = model.predict(input_data)[0]
    except Exception as e:
        return f"Error during prediction: {str(e)}"

    print(f"Predicted Price: {predicted_price}")
    min_price, max_price = crops.get(crop, (0, 0))
    past_data = data[(data['Place'] == place_encoded) & (data['Crop'] == crop_encoded) & (data['Year'] < year)].sort_values('Year', ascending=False).head(2)

    print(f"Past Data:\n{past_data}")

    if not past_data.empty:
        past_data['Crop Price (per Quintal)'] = pd.to_numeric(past_data['Crop Price (per Quintal)'], errors='coerce')
        print(f"Cleaned Past Data (numeric prices):\n{past_data}")

        if past_data['Crop Price (per Quintal)'].isnull().any():
            print("NaN values found in past prices, filling with mean price of current year.")
            mean_price_current_year = data[(data['Place'] == place_encoded) & (data['Crop'] == crop_encoded) & (data['Year'] == year)]
            if not mean_price_current_year.empty:
                mean_price = mean_price_current_year['Crop Price (per Quintal)'].mean()
                print(f"Filling NaN with current year's mean price: {mean_price}")
                past_data['Crop Price (per Quintal)'] = past_data['Crop Price (per Quintal)'].fillna(mean_price)
            else:
                fallback_price = data[(data['Place'] == place_encoded) & (data['Crop'] == crop_encoded)]['Crop Price (per Quintal)'].mean()
                print(f"Filling NaN with overall mean price: {fallback_price}")
                past_data['Crop Price (per Quintal)'] = past_data['Crop Price (per Quintal)'].fillna(fallback_price)

        past_prices = past_data['Crop Price (per Quintal)'].tolist()
        past_years = past_data['Year'].tolist()
        print(f"Past Years: {past_years}, Past Prices: {past_prices}")
    else:
        fallback_data = data[(data['Place'] == place_encoded) & (data['Crop'] == crop_encoded)]
        fallback_data['Crop Price (per Quintal)'] = pd.to_numeric(fallback_data['Crop Price (per Quintal)'], errors='coerce')
        fallback_price = fallback_data['Crop Price (per Quintal)'].fillna(fallback_data['Crop Price (per Quintal)'].mean()).mean()
        print(f"Filling past prices with fallback price: {fallback_price}")
        past_prices = [fallback_price] * 2
        past_years = [year - 2, year - 1]

    past_years_and_prices = list(zip(past_years, past_prices))
    monthly_prices = data[(data['Place'] == place_encoded) & (data['Crop'] == crop_encoded) & (data['Year'] == year)]
    monthly_prices = monthly_prices[['Month', 'Crop Price (per Quintal)']].sort_values('Month')

    print(f"Monthly Prices for {crop} in {year}:\n{monthly_prices}")
    return render_template(
        'index.html',
        prediction=predicted_price,
        place=place,
        year=year,
        rainfall=rainfall,
        crop=crop,
        month=month,
        min_price=min_price,
        max_price=max_price,
        past_years_and_prices=past_years_and_prices,
        years=past_years + [year],
        prices=past_prices + [predicted_price],
        monthly_prices=monthly_prices.to_dict(orient='records') 
    )

@app.route('/analyze_year', methods=['GET'])
def analyze_year():
    """Analyze crop prices over the last two years."""
    place = request.args.get('place', '').capitalize()
    crop = request.args.get('crop', '').upper()
    year = int(request.args.get('year', 0))
    past_data = data[(data['Place'] == place) & (data['Crop'] == crop) & (data['Year'] < year)].sort_values('Year', ascending=False).head(2)
    past_prices = past_data[['Year', 'Crop Price (per Quintal)']].to_dict(orient='records')
    plt.figure(figsize=(10, 5))
    plt.plot(past_data['Year'], past_data['Crop Price (per Quintal)'], marker='o')
    plt.title(f'Crop Prices for {crop} in {place} Over the Last Two Years')
    plt.xlabel('Year')
    plt.ylabel('Price (per Quintal)')
    plt.grid()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close() 

    return render_template('analyze_year.html', past_prices=past_prices, place=place , crop=crop, plot_url=plot_url)
@app.route('/analyze_month', methods=['GET'])
def analyze_month():
    """Analyze crop prices for a specific month and compare with other months in different years."""
    place = request.args.get('place', '').capitalize()
    crop = request.args.get('crop', '').upper()
    year_str = request.args.get('year', '0')
    month_str = request.args.get('month', '1')  

    try:
        year = int(year_str)
        month = int(month_str)  
    except ValueError:
        year = 0
        month = 1
    if data.empty:
        return "Error: No data available for analysis. Please check the dataset."
    available_data = data[(
        data['Place'] == place) & 
        (data['Crop'] == crop)
    ][['Year', 'Month', 'Crop Price (per Quintal)']].dropna()

    if available_data.empty:
        return f"No data available for the crop {crop} in {place}."
    available_data['Month-Year'] = available_data['Month'].astype(str) + '-' + available_data['Year'].astype(str)
    historical_data = available_data.groupby('Month-Year')['Crop Price (per Quintal)'].mean().reset_index()
    comparison_data = data[(
        (data['Place'] == place) & 
        (data['Crop'] == crop) & 
        (data['Month'] == month) 
    )][['Year', 'Month', 'Crop Price (per Quintal)']].sort_values(['Year'])
    comparison_labels = comparison_data['Year'].astype(str)  
    comparison_prices = comparison_data['Crop Price (per Quintal)'] 
    plt.figure(figsize=(10, 5))
    plt.bar(comparison_labels, comparison_prices, color='orange', label=f'{month_str} Prices')
    all_months_data = data[
    (data['Place'] == place) & 
    (data['Crop'] == crop) & 
    (data['Year'] == year) 
][['Month', 'Crop Price (per Quintal)']].sort_values('Month')
    all_months_data['Crop Price (per Quintal)'] += np.random.uniform(-50, 50, size=len(all_months_data))
    plt.bar(all_months_data['Month'], 
        all_months_data['Crop Price (per Quintal)'], 
        label=f'{year} Monthly Prices', 
        color='blue')
    plt.title(f'Crop Prices for {crop} in {place} - {month_str} Comparison')
    plt.xlabel('Year')
    plt.ylabel('Price (per Quintal)')
    plt.grid(True)
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  
    return render_template(
        'analyze_month.html',
        comparison_data=comparison_data.to_dict(orient='records'),
        historical_data=historical_data.to_dict(orient='records'),
        place=place,
        crop=crop,
        year=year,
        plot_url=plot_url,
        selected_month=month,
        message=f"Comparing prices for {month_str} across multiple years."
    )
if __name__ == '__main__':
    app.run(debug=True)