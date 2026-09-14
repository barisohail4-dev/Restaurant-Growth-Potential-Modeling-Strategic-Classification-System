from pathlib import Path
import pandas as pd
import numpy as np

EXPECTED_COLUMNS = [
    'CuisineType','RestaurantID','RestaurantName','Segment','Subregion',
    'GrowthFactor','AOV','MonthlyOrders','InStoreOrders','InStoreRevenue',
    'UberEatsOrders','DoorDashOrders','SelfDeliveryOrders','UberEatsRevenue',
    'DoorDashRevenue','SelfDeliveryRevenue','COGSRate','OPEXRate','CommissionRate',
    'DeliveryRadiusKM','DeliveryCostPerOrder','SD_DeliveryTotalCost',
    'InStoreNetProfit','UberEatsNetProfit','DoorDashNetProfit','SelfDeliveryNetProfit',
    'InStoreShare','UE_share','DD_share','SD_share'
]


def load_data(path):
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing expected columns: {missing}')
    return df


def clean_data(df):
    out = df.copy()
    numeric_cols = out.select_dtypes(include=np.number).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    # No missing values are expected in the supplied dataset. If future data contains
    # missing numeric values, median imputation keeps the pipeline robust.
    for c in numeric_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())
    categorical_cols = out.select_dtypes(exclude=np.number).columns
    for c in categorical_cols:
        out[c] = out[c].fillna('Unknown')
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def profile_data(df):
    return {
        'shape': df.shape,
        'missing_values': int(df.isna().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'numeric_summary': df.describe(include=[np.number]).T,
        'categorical_summary': df.describe(include=['object']).T,
    }
