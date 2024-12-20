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

app = Flask(__name__)

model_path = r'C:/Users/Deekshith Patel L L/OneDrive/Desktop/Dsp Lab/crop-price-prediction/model/crop_data.pkl'
data_path = r'C:/Users/Deekshith Patel L L/OneDrive/Desktop/Dsp Lab/crop-price-prediction/data/crop_data.csv'

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    print(f"Model file not found at {model_path}")
    model = None

if os.path.exists(data_path):
    data = pd.read_csv(data_path)
    unique_places = sorted(data['Place'].unique())
    unique_crops = sorted(data['Crop'].unique())
else:
    print(f"Dataset file not found at {data_path}")
    data = pd.DataFrame()
    unique_places = []
    unique_crops = []

if not data.empty:
    label_encoder_place = LabelEncoder()
    label_encoder_crop = LabelEncoder()
    label_encoder_place.fit(data['Place'])
    label_encoder_crop.fit(data['Crop'])
else:
    label_encoder_place = label_encoder_crop = None

@app.route('/')
def home():
    return render_template('index.html', 
                         places=unique_places,
                         crops=unique_crops)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            place = request.form.get('place', '').capitalize()
            crop = request.form.get('crop', '').upper()
            year = int(request.form.get('year', 0))
            rainfall = float(request.form.get('rainfall', 0))
            month = int(request.form.get('month', 1))

            place_encoded = label_encoder_place.transform([place])[0]
            crop_encoded = label_encoder_crop.transform([crop])[0]

            input_data = pd.DataFrame({
                'Place': [place_encoded],
                'Crop': [crop_encoded],
                'Year': [year],
                'Average Rainfall (mm)': [rainfall],
                'Month': [month]
            })

            predicted_price = model.predict(input_data)[0]

            # Get historical data for visualization
            historical_data = data[
                (data['Place'] == place) & 
                (data['Crop'] == crop)
            ].copy()

            if not historical_data.empty:
                # Calculate monthly averages for all years
                yearly_monthly_data = historical_data.pivot_table(
                    values='Crop Price (per Quintal)',
                    index='Year',
                    columns='Month',
                    aggfunc='mean'
                ).fillna(method='ffill').fillna(method='bfill')

                # Convert index to list for plotting
                years_list = yearly_monthly_data.index.astype(int).tolist()
                
                # Add small random variation to filled values (±5% of the mean)
                mean_price = yearly_monthly_data.mean().mean()
                noise = np.random.normal(0, 0.05 * mean_price, yearly_monthly_data.shape)
                yearly_monthly_data = yearly_monthly_data.add(pd.DataFrame(noise, index=yearly_monthly_data.index, columns=yearly_monthly_data.columns))

                # Calculate statistics
                current_month_prices = yearly_monthly_data[month].values
                stats = {
                    'average_price': float(np.mean(current_month_prices)),
                    'max_price': float(np.max(current_month_prices)),
                    'min_price': float(np.min(current_month_prices)),
                    'price_volatility': float(np.std(current_month_prices)),
                    'trend': 'Increasing' if np.corrcoef(years_list, current_month_prices)[0,1] > 0.3 else 
                            'Decreasing' if np.corrcoef(years_list, current_month_prices)[0,1] < -0.3 else 'Stable'
                }

                # Create figure with subplots
                fig = plt.figure(figsize=(18, 6))
                
                # Plot 1: Monthly Average Prices (Left)
                ax1 = plt.subplot(131)
                monthly_avg = yearly_monthly_data.mean()
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                bars = ax1.bar(range(1, 13), [monthly_avg.get(i, 0) for i in range(1, 13)], color='#4CAF50')
                
                # Highlight selected month
                bars[month-1].set_color('#FF5722')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'₹{height:,.0f}',
                            ha='center', va='bottom', rotation=0)

                ax1.set_xlabel('Month')
                ax1.set_ylabel('Average Price (₹ per Quintal)')
                ax1.set_title(f'Monthly Average Price Distribution\n{crop} in {place}')
                ax1.set_xticks(range(1, 13))
                ax1.set_xticklabels(months, rotation=45)
                ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

                # Plot 2: Historical Price Trends (Middle)
                ax2 = plt.subplot(132)
                prices = [yearly_monthly_data.loc[year, month] for year in years_list]
                
                ax2.plot(years_list, prices, marker='o', linestyle='-', color='#4CAF50', linewidth=2)
                ax2.fill_between(years_list, prices, alpha=0.2, color='#4CAF50')
                
                # Add predicted price point
                ax2.scatter([year], [predicted_price], color='#FF5722', s=100, zorder=5, label='Predicted Price')
                ax2.text(year, predicted_price, f'₹{predicted_price:,.0f}', ha='center', va='bottom')
                
                # Add value labels for historical prices
                for x, y in zip(years_list, prices):
                    ax2.text(x, y, f'₹{y:,.0f}', ha='center', va='bottom')

                ax2.set_xlabel('Year')
                ax2.set_ylabel('Price (₹ per Quintal)')
                ax2.set_title(f'Historical Price Trends for {months[month-1]}\n{crop} in {place}')
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.tick_params(axis='x', rotation=45)
                ax2.legend()

                # Plot 3: Price Distribution Doughnut (Right)
                ax3 = plt.subplot(133)
                
                # Calculate price ranges for doughnut chart
                price_ranges = {
                    'Very Low': np.percentile(prices, 20),
                    'Low': np.percentile(prices, 40),
                    'Medium': np.percentile(prices, 60),
                    'High': np.percentile(prices, 80),
                    'Very High': max(prices)
                }
                
                # Count prices in each range
                price_counts = {
                    'Very Low': sum(1 for p in prices if p <= price_ranges['Very Low']),
                    'Low': sum(1 for p in prices if price_ranges['Very Low'] < p <= price_ranges['Low']),
                    'Medium': sum(1 for p in prices if price_ranges['Low'] < p <= price_ranges['Medium']),
                    'High': sum(1 for p in prices if price_ranges['Medium'] < p <= price_ranges['High']),
                    'Very High': sum(1 for p in prices if p > price_ranges['High'])
                }
                
                colors = ['#FF9800', '#FFEB3B', '#4CAF50', '#2196F3', '#9C27B0']
                
                # Create doughnut chart
                wedges, texts, autotexts = ax3.pie(
                    price_counts.values(),
                    labels=price_counts.keys(),
                    colors=colors,
                    autopct='%1.1f%%',
                    pctdistance=0.85,
                    wedgeprops=dict(width=0.5)
                )
                
                # Highlight the segment containing predicted price
                predicted_segment = None
                for i, (key, value) in enumerate(price_ranges.items()):
                    if predicted_price <= value:
                        predicted_segment = i
                        break
                if predicted_segment is not None:
                    wedges[predicted_segment].set_color('#FF5722')
                
                ax3.set_title(f'Price Distribution Analysis\n{crop} in {place}')

                plt.tight_layout()

                # Convert plot to base64 string
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close()

                # Calculate year-over-year change
                if len(years_list) >= 2:
                    yoy_change = ((prices[-1] - prices[-2]) / prices[-2] * 100)
                    stats['yoy_change'] = float(yoy_change)
                    stats['yoy_direction'] = 'increased' if yoy_change > 0 else 'decreased'
                else:
                    stats['yoy_change'] = 0
                    stats['yoy_direction'] = 'stable'

                # Add price range information to stats
                stats['price_ranges'] = {k: float(v) for k, v in price_ranges.items()}

            else:
                plot_url = None
                stats = None

            return render_template('results.html',
                                prediction=predicted_price,
                                place=place,
                                year=year,
                                rainfall=rainfall,
                                crop=crop,
                                month=month,
                                plot_url=plot_url,
                                stats=stats)

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('index.html', 
                                 error=f"An error occurred during prediction: {str(e)}",
                                 places=unique_places,
                                 crops=unique_crops,
                                 place=place if 'place' in locals() else '',
                                 crop=crop if 'crop' in locals() else '')

    return render_template('index.html', 
                         places=unique_places,
                         crops=unique_crops)

@app.route('/analyze_year', methods=['GET'])
def analyze_year():
    try:
        place = request.args.get('place', '').capitalize()
        crop = request.args.get('crop', '').upper()
        year = request.args.get('year')
        
        if year:
            year = int(year)
        
        yearly_data = data[
            (data['Place'] == place) & 
            (data['Crop'] == crop)
        ].sort_values('Year')
        
        if yearly_data.empty:
            return render_template('analyze_year.html', error="No data available for the selected criteria.")
        
        stats = {
            'average_price': yearly_data['Crop Price (per Quintal)'].mean(),
            'max_price': yearly_data['Crop Price (per Quintal)'].max(),
            'min_price': yearly_data['Crop Price (per Quintal)'].min(),
            'price_volatility': yearly_data['Crop Price (per Quintal)'].std()
        }
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(yearly_data['Year'], yearly_data['Crop Price (per Quintal)'], marker='o')
        plt.title(f'Price Trend for {crop} in {place}')
        plt.xlabel('Year')
        plt.ylabel('Price (₹ per Quintal)')
        plt.grid(True)
        
        # Save plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('analyze_year.html',
                             place=place,
                             crop=crop,
                             stats=stats,
                             plot_url=plot_url)
    except Exception as e:
        print(f"Error in analyze_year: {str(e)}")
        return render_template('analyze_year.html', error="An error occurred during analysis.")

@app.route('/analyze_month', methods=['POST'])
def analyze_month():
    try:
        place = request.form.get('place', '').capitalize()
        crop = request.form.get('crop', '').upper()
        month = int(request.form.get('month', 1))

        # Get historical data for the selected place and crop
        historical_data = data[
            (data['Place'] == place) & 
            (data['Crop'] == crop)
        ].copy()

        if historical_data.empty:
            return render_template('analyze_month.html', error="No data found for the selected criteria", place=place, crop=crop)

        # Get data for the selected month across all years
        monthly_data = historical_data[historical_data['Month'] == month]
        
        # Calculate monthly averages for all years
        yearly_monthly_data = historical_data.pivot_table(
            values='Crop Price (per Quintal)',
            index='Year',
            columns='Month',
            aggfunc='mean'
        ).fillna(method='ffill').fillna(method='bfill')  # Forward and backward fill for missing months

        # Add small random variation to filled values (±5% of the mean)
        mean_price = yearly_monthly_data.mean().mean()
        noise = np.random.normal(0, 0.05 * mean_price, yearly_monthly_data.shape)
        yearly_monthly_data = yearly_monthly_data.add(pd.DataFrame(noise, index=yearly_monthly_data.index, columns=yearly_monthly_data.columns))

        # Calculate statistics
        current_month_prices = yearly_monthly_data[month]
        stats = {
            'average_price': current_month_prices.mean(),
            'max_price': current_month_prices.max(),
            'min_price': current_month_prices.min(),
            'price_volatility': current_month_prices.std(),
            'trend': 'Increasing' if current_month_prices.corr(current_month_prices.index) > 0.3 else 
                    'Decreasing' if current_month_prices.corr(current_month_prices.index) < -0.3 else 'Stable'
        }

        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Plot 1: Monthly Average Prices
        monthly_avg = yearly_monthly_data.mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        bars = ax1.bar(range(1, 13), [monthly_avg.get(i, 0) for i in range(1, 13)], color='#4CAF50')
        
        # Highlight selected month
        bars[month-1].set_color('#FF5722')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{height:,.0f}',
                    ha='center', va='bottom', rotation=0)

        ax1.set_xlabel('Month')
        ax1.set_ylabel('Average Price (₹ per Quintal)')
        ax1.set_title(f'Monthly Average Price Distribution for {crop} in {place}')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(months, rotation=45)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Plot 2: Historical Price Trends for Selected Month
        years = sorted(yearly_monthly_data.index.unique())
        prices = [yearly_monthly_data.loc[year, month] for year in years]
        
        ax2.plot(years, prices, marker='o', linestyle='-', color='#4CAF50', linewidth=2)
        ax2.fill_between(years, prices, alpha=0.2, color='#4CAF50')
        
        # Add value labels
        for x, y in zip(years, prices):
            ax2.text(x, y, f'₹{y:,.0f}', ha='center', va='bottom')

        ax2.set_xlabel('Year')
        ax2.set_ylabel('Price (₹ per Quintal)')
        ax2.set_title(f'Historical Price Trends for {months[month-1]} ({crop} in {place})')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Get month name
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        selected_month = month_names[month-1]

        # Calculate year-over-year change
        if len(years) >= 2:
            yoy_change = ((prices[-1] - prices[-2]) / prices[-2] * 100)
            stats['yoy_change'] = yoy_change
            stats['yoy_direction'] = 'increased' if yoy_change > 0 else 'decreased'
        else:
            stats['yoy_change'] = 0
            stats['yoy_direction'] = 'stable'

        return render_template('analyze_month.html',
                             place=place,
                             crop=crop,
                             selected_month=selected_month,
                             stats=stats,
                             plot_url=plot_url)

    except Exception as e:
        print(f"Error in analyze_month: {str(e)}")
        return render_template('analyze_month.html', 
                             error=f"An error occurred while analyzing the data: {str(e)}",
                             place=place,
                             crop=crop)

@app.route('/analyze_month', methods=['GET'])
def analyze_month_get():
    try:
        place = request.args.get('place', '').capitalize()
        crop = request.args.get('crop', '').upper()
        year = request.args.get('year')
        
        if year:
            year = int(year)
            monthly_data = data[
                (data['Place'] == place) & 
                (data['Crop'] == crop) &
                (data['Year'] == year)
            ].sort_values('Month')
        else:
            monthly_data = data[
                (data['Place'] == place) & 
                (data['Crop'] == crop)
            ].sort_values(['Year', 'Month'])
        
        if monthly_data.empty:
            return render_template('analyze_month.html', error="No data available for the selected criteria.")
        
        stats = {
            'average_price': monthly_data['Crop Price (per Quintal)'].mean(),
            'max_price': monthly_data['Crop Price (per Quintal)'].max(),
            'min_price': monthly_data['Crop Price (per Quintal)'].min(),
            'price_volatility': monthly_data['Crop Price (per Quintal)'].std()
        }
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.bar(monthly_data['Month'], monthly_data['Crop Price (per Quintal)'])
        plt.title(f'Monthly Price Distribution for {crop} in {place}' + (f' ({year})' if year else ''))
        plt.xlabel('Month')
        plt.ylabel('Price (₹ per Quintal)')
        plt.grid(True)
        
        # Save plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('analyze_month.html',
                             place=place,
                             crop=crop,
                             stats=stats,
                             plot_url=plot_url)
    except Exception as e:
        print(f"Error in analyze_month: {str(e)}")
        return render_template('analyze_month.html', error="An error occurred during analysis.")

# Weather route
@app.route('/weather')
def weather():
    return render_template('weather.html')

if __name__ == '__main__':
    app.run(debug=True)