from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.data_preprocessing import load_data, clean_data, profile_data
from src.feature_engineering import engineer_features, get_clustering_features
from src.clustering import fit_models
from src.gpi_model import calculate_gpi, add_gpi_category, DEFAULT_WEIGHTS
from src.recommendations import generate_recommendations, profile_clusters
from src.visualization import save_eda_plots

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'SkyCity Auckland Restaurants & Bars.csv'
OUTPUT_DIR = ROOT / 'outputs'
MODEL_DIR = ROOT / 'models'
PLOT_DIR = OUTPUT_DIR / 'eda_plots'


def name_clusters(df):
    """Assign data-driven names to the selected 5-cluster solution.

    The naming logic is based on the actual profile patterns in this supplied dataset.
    If a different K is selected, generic names are used so we never claim unsupported
    archetypes.
    """
    d = df.copy()
    k = d['Cluster'].nunique()
    if k != 5:
        d['ClusterName'] = d['Cluster'].map(lambda x: f'Cluster {x}')
        return d

    p = d.groupby('Cluster').agg(
        growth=('GrowthMomentum','mean'),
        orders=('MonthlyOrders','mean'),
        instore=('InStoreShare','mean'),
        ue=('UE_share','mean'),
        dd=('DD_share','mean'),
        sd=('SD_share','mean'),
        cost=('CostPressure','mean'),
        radius=('DeliveryRadiusKM','mean'),
        delivery_cost=('DeliveryCostPerOrder','mean'),
        margin=('ProfitMargin','mean'),
        gpi=('GPI','mean')
    )

    # Identify characteristic clusters by relative profiles.
    names = {}
    names[int(p['cost'].idxmax())] = 'High Cost / Low Return'
    remaining = [i for i in p.index if i not in names]
    if remaining:
        names[int(p.loc[remaining, 'sd'].idxmax())] = 'Scalable Self-Delivery Leaders'
    remaining = [i for i in p.index if i not in names]
    if remaining:
        # Highest in-store share among remaining tends to represent local/stable businesses.
        names[int(p.loc[remaining, 'instore'].idxmax())] = 'Stable Local Performers'
    remaining = [i for i in p.index if i not in names]
    if remaining:
        # Highest delivery radius among remaining captures wider delivery reach.
        names[int(p.loc[remaining, 'radius'].idxmax())] = 'Wide-Reach Delivery Operators'
    remaining = [i for i in p.index if i not in names]
    if remaining:
        names[int(remaining[0])] = 'Aggregator-Dependent Performers'

    d['ClusterName'] = d['Cluster'].map(names).fillna('Cluster')
    return d


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    raw = load_data(DATA_PATH)
    clean = clean_data(raw)
    prof = profile_data(clean)
    clean.to_csv(OUTPUT_DIR / 'cleaned_restaurants.csv', index=False)

    data = engineer_features(clean)
    save_eda_plots(data, PLOT_DIR)

    X = get_clustering_features(data)
    result = fit_models(X, MODEL_DIR, preferred_k=5, random_state=42)
    data['Cluster'] = result['labels']
    data['PCA1'] = result['X_pca'][:,0]
    data['PCA2'] = result['X_pca'][:,1]
    data['HierarchicalCluster'] = result['hierarchical_labels']

    data, weights = calculate_gpi(data, DEFAULT_WEIGHTS)
    data = add_gpi_category(data)
    data = name_clusters(data)
    data = generate_recommendations(data)

    # Final profile after names/GPI are attached.
    profile = profile_clusters(data)
    profile.to_csv(OUTPUT_DIR / 'cluster_profiles.csv', index=False)

    output_cols = [
        'RestaurantID','RestaurantName','CuisineType','Segment','Subregion',
        'GrowthFactor','AOV','MonthlyOrders','InStoreOrders','UberEatsOrders',
        'DoorDashOrders','SelfDeliveryOrders','InStoreRevenue','UberEatsRevenue',
        'DoorDashRevenue','SelfDeliveryRevenue','COGSRate','OPEXRate','CommissionRate',
        'DeliveryRadiusKM','DeliveryCostPerOrder','SD_DeliveryTotalCost',
        'InStoreNetProfit','UberEatsNetProfit','DoorDashNetProfit','SelfDeliveryNetProfit',
        'InStoreShare','UE_share','DD_share','SD_share',
        'TotalRevenue','TotalNetProfit','ProfitMargin','GrowthMomentum',
        'AggregatorDependence','TotalDeliveryOrders','DeliveryOrderShare','CostPressure',
        'SelfDeliveryProfitPerOrder','AggregatorNetProfit','Scale','RevenuePerOrder',
        'NetProfitPerOrder','Cluster','ClusterName','GPI','GPI_Category','Recommendation',
        'GPI_Growth','GPI_Profitability','GPI_CostResilience','GPI_ChannelBalance',
        'GPI_Scalability','PCA1','PCA2','HierarchicalCluster'
    ]
    final = data[[c for c in output_cols if c in data.columns]]
    final.to_csv(OUTPUT_DIR / 'restaurant_predictions.csv', index=False)

    metrics = result['metrics'].copy()
    metrics.to_csv(OUTPUT_DIR / 'clustering_metrics.csv', index=False)

    summary = {
        'rows_raw': int(raw.shape[0]),
        'columns_raw': int(raw.shape[1]),
        'rows_clean': int(clean.shape[0]),
        'selected_k': int(result['selected_k']),
        'pca_2d_explained_variance': result['explained_variance_2d'],
        'pca_components_for_80_percent_variance': int(result['pca_80_components']),
        'gpi_weights': weights,
        'average_gpi': float(final['GPI'].mean()),
        'recommendation_counts': final['Recommendation'].value_counts().to_dict(),
        'gpi_category_counts': final['GPI_Category'].value_counts().to_dict(),
        'cluster_counts': final['ClusterName'].value_counts().to_dict(),
    }
    (OUTPUT_DIR / 'model_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('Training completed successfully.')
    print(f'Rows: {len(final)} | Selected K: {result["selected_k"]}')
    print('\nClustering metrics:')
    print(metrics.round(4).to_string(index=False))
    print('\nCluster sizes:')
    print(final['ClusterName'].value_counts().to_string())
    print('\nGPI categories:')
    print(final['GPI_Category'].value_counts().to_string())
    print('\nRecommendations:')
    print(final['Recommendation'].value_counts().to_string())

if __name__ == '__main__':
    main()
