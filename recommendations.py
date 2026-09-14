import pandas as pd


def assign_recommendation(row):
    # Rule-based business recommendations are intentionally based on measurable
    # characteristics rather than cluster IDs alone.
    if row['TotalNetProfit'] < 0 or row['ProfitMargin'] < 0:
        if row['CostPressure'] >= row.get('_cost_pressure_median', row['CostPressure']) or row['ProfitMargin'] < -0.02:
            return 'Optimize'
    if row['AggregatorDependence'] >= 0.70 and row['AggregatorNetProfit'] <= 0:
        return 'Rebalance Channels'
    if row['GPI'] >= 80 and row['GrowthMomentum'] > 0 and row['TotalNetProfit'] > 0:
        return 'Expand'
    if row['AggregatorDependence'] >= 0.75 and row['CommissionRate'] > 0:
        return 'Rebalance Channels'
    return 'Hold / Stabilize'


def generate_recommendations(df):
    d = df.copy()
    median_cost = d['CostPressure'].median()
    d['_cost_pressure_median'] = median_cost
    d['Recommendation'] = d.apply(assign_recommendation, axis=1)
    return d.drop(columns=['_cost_pressure_median'])


def profile_clusters(df):
    numeric = [
        'GrowthFactor','AOV','MonthlyOrders','TotalRevenue','TotalNetProfit',
        'ProfitMargin','AggregatorDependence','DeliveryOrderShare','CostPressure',
        'DeliveryRadiusKM','DeliveryCostPerOrder','Scale','GPI'
    ]
    available = [c for c in numeric if c in df.columns]
    return df.groupby(['Cluster','ClusterName'])[available].agg(['mean','median']).round(3).reset_index()
