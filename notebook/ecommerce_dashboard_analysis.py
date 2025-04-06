
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import folium

# Load sample data
df = pd.read_csv('ecommerce_sample_data.csv', parse_dates=['PurchaseDate'])

# -------------------- Predictive Modeling: Likelihood of Return --------------------
# Create a binary target: Returned (1 if PurchaseFrequency > 1, else 0)
df['Returned'] = df['PurchaseFrequency'].apply(lambda x: 1 if x > 1 else 0)

# Select features for modeling
features = ['Age', 'OrderValue']
X = df[features]
y = df['Returned']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# -------------------- Geographic Heatmap --------------------
# Create base map
geo_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=4)

# Add circle markers
for i, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        popup=f"{row['Location']} - ${row['OrderValue']}",
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(geo_map)

# Save map
geo_map.save('ecommerce_geo_map.html')

# Save classification report
report_df.to_csv('predictive_model_report.csv')
