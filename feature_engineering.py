import numpy as np
import pandas as pd


def safe_divide(a, b):
    return a.div(b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)


def engineer_features(df):
    d = df.copy()
    d['TotalRevenue'] = d[['InStoreRevenue','UberEatsRevenue','DoorDashRevenue','SelfDeliveryRevenue']].sum(axis=1)
    d['TotalNetProfit'] = d[['InStoreNetProfit','UberEatsNetProfit','DoorDashNetProfit','SelfDeliveryNetProfit']].sum(axis=1)
    d['ProfitMargin'] = safe_divide(d['TotalNetProfit'], d['TotalRevenue'])
    d['GrowthMomentum'] = d['GrowthFactor'] - 1.0
    d['AggregatorDependence'] = d['UE_share'] + d['DD_share']
    d['TotalDeliveryOrders'] = d[['UberEatsOrders','DoorDashOrders','SelfDeliveryOrders']].sum(axis=1)
    d['DeliveryOrderShare'] = safe_divide(d['TotalDeliveryOrders'], d['MonthlyOrders'])
    d['CostPressure'] = d['COGSRate'] + d['OPEXRate']
    d['SelfDeliveryProfitPerOrder'] = safe_divide(d['SelfDeliveryNetProfit'], d['SelfDeliveryOrders'])
    d['AggregatorNetProfit'] = d['UberEatsNetProfit'] + d['DoorDashNetProfit']
    d['Scale'] = d['MonthlyOrders'] * d['GrowthFactor']
    # Additional transparent business indicators for the dashboard.
    d['RevenuePerOrder'] = safe_divide(d['TotalRevenue'], d['MonthlyOrders'])
    d['NetProfitPerOrder'] = safe_divide(d['TotalNetProfit'], d['MonthlyOrders'])
    d['SelfDeliveryCostTotalCheck'] = d['SelfDeliveryOrders'] * d['DeliveryCostPerOrder']
    return d.replace([np.inf, -np.inf], np.nan).fillna(0)


# Initial clustering feature set. It avoids IDs/names and excludes highly redundant
# channel revenue/profit totals from the distance model. The derived ProfitMargin
# captures overall financial quality without giving each channel equal extra weight.
CLUSTER_FEATURES = [
    'GrowthFactor','AOV','MonthlyOrders',
    'InStoreShare','UE_share','DD_share','SD_share',
    'COGSRate','OPEXRate','CommissionRate',
    'DeliveryRadiusKM','DeliveryCostPerOrder','ProfitMargin'
]


def get_clustering_features(df):
    missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f'Missing clustering features: {missing}')
    return df[CLUSTER_FEATURES].copy()
