from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score


def evaluate_kmeans(X_scaled, k_values=range(2, 9), random_state=42):
    rows = []
    models = {}
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(X_scaled)
        rows.append({
            'k': k,
            'inertia': model.inertia_,
            'silhouette_score': silhouette_score(X_scaled, labels),
            'davies_bouldin_score': davies_bouldin_score(X_scaled, labels)
        })
        models[k] = model
    return pd.DataFrame(rows), models


def choose_k(metrics, preferred_k=5):
    # Score candidates on normalized silhouette and inverse DB index. This is a
    # transparent compromise between mathematical quality and business usability.
    m = metrics.copy()
    sil_min, sil_max = m.silhouette_score.min(), m.silhouette_score.max()
    db_min, db_max = m.davies_bouldin_score.min(), m.davies_bouldin_score.max()
    m['sil_norm'] = (m.silhouette_score - sil_min) / (sil_max - sil_min + 1e-12)
    m['db_norm'] = (db_max - m.davies_bouldin_score) / (db_max - db_min + 1e-12)
    m['combined_score'] = 0.65 * m['sil_norm'] + 0.35 * m['db_norm']
    # Prefer a business-interpretable five-cluster solution when it is not grossly
    # inferior. The final report still shows the metric-based optimum.
    best_metric_k = int(m.loc[m['combined_score'].idxmax(), 'k'])
    preferred_row = m.loc[m['k'] == preferred_k]
    if not preferred_row.empty:
        pref_score = float(preferred_row['combined_score'].iloc[0])
        best_score = float(m['combined_score'].max())
        selected = preferred_k if pref_score >= 0.70 * best_score else best_metric_k
    else:
        selected = best_metric_k
    return int(selected), m


def fit_models(X, output_dir, preferred_k=5, random_state=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_scaled)
    explained = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_80 = int(np.argmax(explained >= 0.80) + 1)
    pca_2d = PCA(n_components=2, random_state=random_state)
    X_pca = pca_2d.fit_transform(X_scaled)

    metrics, models = evaluate_kmeans(X_scaled, random_state=random_state)
    selected_k, metric_table = choose_k(metrics, preferred_k=preferred_k)
    kmeans = models[selected_k]
    labels = kmeans.predict(X_scaled)
    hierarchical = AgglomerativeClustering(n_clusters=selected_k)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)

    joblib.dump(scaler, output_dir / 'scaler.pkl')
    joblib.dump(pca_2d, output_dir / 'pca.pkl')
    joblib.dump(kmeans, output_dir / 'kmeans_model.pkl')

    return {
        'scaler': scaler,
        'pca': pca_2d,
        'pca_full': pca_full,
        'kmeans': kmeans,
        'hierarchical': hierarchical,
        'X_scaled': X_scaled,
        'X_pca': X_pca,
        'labels': labels,
        'hierarchical_labels': hierarchical_labels,
        'metrics': metrics,
        'metric_table': metric_table,
        'selected_k': selected_k,
        'pca_80_components': n_components_80,
        'explained_variance_2d': float(pca_2d.explained_variance_ratio_.sum()),
    }
