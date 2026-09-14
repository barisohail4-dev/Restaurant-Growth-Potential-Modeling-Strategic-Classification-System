import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DEFAULT_WEIGHTS = {
    'growth': 0.25,
    'profitability': 0.25,
    'cost_resilience': 0.20,
    'channel_balance': 0.15,
    'scalability': 0.15,
}


def _minmax(s):
    lo, hi = float(s.min()), float(s.max())
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    return (s - lo) / (hi - lo) * 100.0


def _inverse_minmax(s):
    return 100.0 - _minmax(s)


def calculate_gpi(df, weights=None):
    d = df.copy()
    w = DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)
    total_w = sum(w.values())
    if total_w <= 0:
        raise ValueError('GPI weights must sum to a positive value.')
    w = {k: v / total_w for k, v in w.items()}

    # Growth: direct growth momentum.
    growth_score = _minmax(d['GrowthMomentum'])

    # Profitability: combine margin and profit per order. Margin is the main signal;
    # profit per order adds a scale-aware check.
    profit_margin_score = _minmax(d['ProfitMargin'])
    profit_order_score = _minmax(d['NetProfitPerOrder'])
    profitability_score = 0.7 * profit_margin_score + 0.3 * profit_order_score

    # Cost resilience: lower COGS+OPEX pressure and lower delivery cost are better.
    cost_pressure_score = _inverse_minmax(d['CostPressure'])
    delivery_cost_score = _inverse_minmax(d['DeliveryCostPerOrder'])
    cost_score = 0.75 * cost_pressure_score + 0.25 * delivery_cost_score

    # Channel balance: lower dependence on third-party aggregators is treated as
    # more resilient. This does not mean delivery is bad; it means concentration risk.
    channel_score = _inverse_minmax(d['AggregatorDependence'])

    # Scalability: demand scale plus delivery reach. We use rank-percentile-like
    # min-max components so large raw order counts do not dominate the final score.
    scale_score = _minmax(d['Scale'])
    radius_score = _minmax(d['DeliveryRadiusKM'])
    scalability_score = 0.65 * scale_score + 0.35 * radius_score

    gpi = (
        w['growth'] * growth_score +
        w['profitability'] * profitability_score +
        w['cost_resilience'] * cost_score +
        w['channel_balance'] * channel_score +
        w['scalability'] * scalability_score
    )

    # Explicitly penalize negative-profit restaurants. This prevents a high growth
    # or scale score from masking a structurally loss-making business.
    negative_profit_penalty = np.where(d['TotalNetProfit'] < 0, 15.0, 0.0)
    d['GPI'] = np.clip(gpi - negative_profit_penalty, 0, 100).round(2)
    d['GPI_Growth'] = growth_score.round(2)
    d['GPI_Profitability'] = profitability_score.round(2)
    d['GPI_CostResilience'] = cost_score.round(2)
    d['GPI_ChannelBalance'] = channel_score.round(2)
    d['GPI_Scalability'] = scalability_score.round(2)
    return d, w


def categorize_gpi(score):
    if score >= 80:
        return 'Very High Potential'
    if score >= 60:
        return 'High Potential'
    if score >= 40:
        return 'Moderate Potential'
    return 'Low Potential'


def add_gpi_category(df):
    d = df.copy()
    d['GPI_Category'] = d['GPI'].apply(categorize_gpi)
    return d
