# Restaurant Growth Potential Modeling & Strategic Classification System

An end-to-end data science project for classifying restaurants by structural growth characteristics, calculating a transparent Growth Potential Index (GPI), and generating strategic recommendations.

## Dataset

`data/SkyCity Auckland Restaurants & Bars.csv`

The supplied dataset contains 1,696 restaurant records and 30 columns.

## Objectives

- Perform EDA on restaurant operations, demand, revenue, costs and channels.
- Engineer business-oriented features.
- Group similar restaurants with unsupervised learning.
- Evaluate K-Means solutions with Silhouette Score and Davies-Bouldin Index.
- Use PCA for dimensionality reduction and visualization.
- Calculate a 0–100 Growth Potential Index.
- Assign strategic recommendations.
- Provide an interactive Streamlit dashboard.

## Main algorithms

- StandardScaler
- PCA
- K-Means Clustering
- Agglomerative/Hierarchical Clustering
- Silhouette Score
- Davies-Bouldin Index
- Weighted GPI business scoring model

## Libraries

Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Plotly, Streamlit and Joblib.

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Run `train_model.py` before opening the dashboard. It creates the model files and output CSV files.

## Project flow

CSV → preprocessing → EDA → feature engineering → feature selection → scaling → PCA → K-Means → cluster evaluation → cluster interpretation → GPI → recommendations → Streamlit.

## GPI

The GPI combines growth, profitability, cost resilience, channel balance and scalability. The weights are configurable in `src/gpi_model.py`.

Negative total profit receives an explicit penalty so that strong order growth cannot hide a loss-making restaurant.

## Important modeling decisions

RestaurantID and RestaurantName are retained for identification but excluded from clustering. Categorical identity fields are not automatically one-hot encoded because the primary clustering objective is based on quantitative business structure. Channel shares are interpreted according to their definitions; `InStoreShare` is not forced to add with delivery-channel shares to 100%.

The number of clusters is tested from 2 to 8. A five-cluster solution is preferred only when its combined clustering quality is not substantially worse than the metric-based best solution, because the project brief gives five archetype examples but does not justify forcing five clusters.

## Outputs

- `outputs/cleaned_restaurants.csv`
- `outputs/restaurant_predictions.csv`
- `outputs/cluster_profiles.csv`
- `outputs/clustering_metrics.csv`
- `outputs/model_summary.json`
- `outputs/eda_plots/`
- `models/scaler.pkl`
- `models/pca.pkl`
- `models/kmeans_model.pkl`

## Limitations

The GPI is a decision-support score, not a causal forecast of future revenue. The dataset appears to represent a single structured business snapshot, so true time-series forecasting and causal expansion effects cannot be established from it alone. Recommendations should therefore be treated as analytical guidance.
