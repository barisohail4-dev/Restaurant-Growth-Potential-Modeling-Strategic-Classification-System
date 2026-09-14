# Restaurant Growth Potential Modeling & Strategic Classification System

## Abstract

This project develops a data-driven framework for identifying restaurant growth archetypes and quantifying growth potential using operational, financial, channel and logistics indicators. The workflow combines exploratory data analysis, feature engineering, standardized unsupervised clustering, PCA-based visualization, and a transparent Growth Potential Index (GPI). The final system converts analytical results into strategic recommendations such as Expand, Optimize, Rebalance Channels, and Hold / Stabilize.

## 1. Introduction

Restaurant performance is influenced by demand, growth, channel mix, operating costs, profitability and delivery reach. A single profitability metric cannot fully represent expansion readiness. This project therefore evaluates restaurants across multiple dimensions and groups structurally similar businesses.

## 2. Dataset

The dataset contains 1,696 restaurant records and 30 fields covering restaurant identity, cuisine, segment, subregion, growth, orders, revenue, costs, profits, delivery logistics and channel shares.

## 3. Methodology

### 3.1 Data preprocessing

The dataset is loaded with Pandas and checked for shape, data types, missing values, duplicates, categorical distributions, descriptive statistics and potential outliers. Restaurant identifiers are retained for reporting but excluded from clustering.

### 3.2 Feature engineering

Business indicators include TotalRevenue, TotalNetProfit, ProfitMargin, GrowthMomentum, AggregatorDependence, DeliveryOrderShare, CostPressure, SelfDeliveryProfitPerOrder, AggregatorNetProfit and Scale.

### 3.3 Clustering

Numerical business features are standardized using StandardScaler. K-Means is evaluated for multiple values of K using inertia, Silhouette Score and Davies-Bouldin Index. Agglomerative clustering provides a secondary comparison. PCA is used for two-dimensional visualization and latent-structure analysis.

### 3.4 Growth Potential Index

GPI is a weighted business score combining growth, profitability, cost resilience, channel balance and scalability. Negative-profit restaurants receive an explicit penalty. The score is normalized to 0–100 and categorized into Very High, High, Moderate and Low Potential.

### 3.5 Recommendations

Recommendations are generated from measurable business signals rather than cluster IDs alone. High growth with healthy profitability can support expansion; high aggregator dependence with weak aggregator economics can support channel rebalancing; high cost pressure or negative profit can support optimization; otherwise the recommendation is to hold/stabilize.

## 4. Results

Run `python train_model.py` to generate the actual clustering metrics, cluster profiles, GPI distribution and recommendation counts for the supplied dataset. The values in `outputs/` are the reproducible results produced by the pipeline.

## 5. Dashboard

The Streamlit dashboard provides executive KPIs, restaurant exploration, PCA cluster visualization, GPI rankings and strategic recommendations.

## 6. Limitations

GPI is a transparent analytical score rather than a future-growth prediction model. The data does not establish causal relationships. Additional historical monthly observations would be required for time-series forecasting and stronger validation of expansion outcomes.

## 7. Conclusion

The framework transforms multi-dimensional restaurant data into interpretable restaurant archetypes, growth scores and strategic actions. It provides a practical bridge between unsupervised machine learning and business decision support.
