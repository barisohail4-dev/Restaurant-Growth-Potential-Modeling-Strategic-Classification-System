import matplotlib.pyplot as plt
import seaborn as sns


def save_eda_plots(df, out_dir):
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    categorical = ['CuisineType','Segment','Subregion']
    for col in categorical:
        plt.figure(figsize=(10,5))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Restaurant Count by {col}')
        plt.xlabel(col)
        plt.ylabel('Restaurants')
        plt.xticks(rotation=35, ha='right')
        plt.tight_layout()
        plt.savefig(out_dir / f'{col.lower()}_distribution.png', dpi=150)
        plt.close()

    for col in ['GrowthFactor','AOV','MonthlyOrders','TotalRevenue','TotalNetProfit','CostPressure']:
        if col in df:
            plt.figure(figsize=(8,5))
            sns.histplot(df[col], kde=True, color='steelblue')
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.savefig(out_dir / f'{col.lower()}_distribution.png', dpi=150)
            plt.close()

    pairs = [
        ('GrowthFactor','MonthlyOrders'),
        ('MonthlyOrders','TotalRevenue'),
        ('TotalRevenue','TotalNetProfit'),
        ('CostPressure','TotalNetProfit')
    ]
    for x,y in pairs:
        plt.figure(figsize=(8,5))
        sns.scatterplot(data=df, x=x, y=y, alpha=0.55, color='steelblue')
        plt.title(f'{x} vs {y}')
        plt.tight_layout()
        plt.savefig(out_dir / f'{x.lower()}_vs_{y.lower()}.png', dpi=150)
        plt.close()

    numeric = df.select_dtypes(include='number')
    plt.figure(figsize=(14,10))
    sns.heatmap(numeric.corr(), cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(out_dir / 'correlation_heatmap.png', dpi=150)
    plt.close()
