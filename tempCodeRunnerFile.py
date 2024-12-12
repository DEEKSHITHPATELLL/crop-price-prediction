
@app.route('/analyze_year', methods=['GET'])
def analyze_year():
    """Analyze crop prices over the last two years."""
    place = request.args.get('place', '').capitalize()
    crop = request.args.get('crop', '').upper()
    year = int(request.args.get('year', 0))

    # Fetch past prices for the same crop and location for the last two years
    past_data = data[(data['Place'] == place) & (data['Crop'] == crop) & (data['Year'] < year)].sort_values('Year', ascending=False).head(2)
    past_prices = past_data[['Year', 'Crop Price (per Quintal)']].to_dict(orient='records')

    # Plotting the graph for past prices
    plt.figure(figsize=(10, 5))
    plt.plot(past_data['Year'], past_data['Crop Price (per Quintal)'], marker='o')
    plt.title(f'Crop Prices for {crop} in {place} Over the Last Two Years')
    plt.xlabel('Year')
    plt.ylabel('Price (per Quintal)')
    plt.grid()

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Close the plot to free memory

    return render_template('analyze_year.html', past_prices=past_prices, place=place , crop=crop, plot_url=plot_url)

@app.route('/analyze_month', methods=['GET'])
def analyze_month():
    """Analyze crop prices in the same year across different months."""
    place = request.args.get('place', '').capitalize()
    crop = request.args.get('crop', '').upper()
    year = int(request.args.get('year', 0))

    # Fetch crop prices for the same crop in the same year but different months
    monthly_prices = data[(data['Place'] == place) & (data['Crop'] == crop) & (data['Year'] == year)]
    monthly_prices = monthly_prices[['Month', 'Crop Price (per Quintal)']].sort_values('Month')

    # Plotting the graph for monthly prices
    plt.figure(figsize=(10, 5))
    plt.bar(monthly_prices['Month'], monthly_prices['Crop Price (per Quintal)'], color='skyblue')
    plt.title(f'Monthly Crop Prices for {crop} in {place} ({year})')
    plt.xlabel('Month')
    plt.ylabel('Price (per Quintal)')
    plt.xticks(monthly_prices['Month'])
    plt.grid(axis='y')

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Close the plot to free memory

    return render_template('analyze_month.html', monthly_prices=monthly_prices.to_dict(orient='records'), place=place, crop=crop, year=year, plot_url=plot_url)
