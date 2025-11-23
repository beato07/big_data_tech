import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import plotly.graph_objects as go


def task_1():
    df = pd.read_csv('insurance.csv')

    df_clean = df.drop_duplicates()
    print(f'Number of unique values: {len(df_clean)}')

    features = df_clean[['age', 'bmi', 'children']]

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    features_df = pd.DataFrame(features_normalized, columns=features.columns)
    print("\nСтатистики после нормализации:")
    print(features_df.describe())

    # Кластеризация
    inertias = []
    silhouette_scores = []

    for k in range(2, 10):
        model = KMeans(n_clusters=k, random_state=123).fit(features_normalized)
        inertias.append(model.inertia_)
        silhouette_scores.append(silhouette_score(features_normalized, model.labels_))

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(2, 10), inertias, 'bo-')
    plt.title('Метод локтя')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Inertia')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(range(2, 10), silhouette_scores, 'ro-')
    plt.title('Силуэтный коэффициент')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Silhouette Score')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    best_k = range(2, 10)[np.argmax(silhouette_scores)]
    print(f'Рекомендуемое количество кластеров: {best_k}')

    return best_k


def task_2():
    df = pd.read_csv('insurance.csv')

    df_clean = df.drop_duplicates()
    print(f'Number of unique values: {len(df_clean)}')

    features = df_clean[['age', 'bmi', 'children']]

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    features_df = pd.DataFrame(features_normalized, columns=features.columns)
    print("\nСтатистики после нормализации:")
    print(features_df.describe())

    model12 = AgglomerativeClustering(6, compute_distances=True)
    clastering = model12.fit(features_df)
    features_df['Claster'] = clastering.labels_

    fig = go.Figure(data=[go.Scatter3d(
        x=features_df['age'],
        y=features_df['bmi'],
        z=features_df['children'],
        mode='markers',
        marker_color=features_df['Claster'],
        marker_size=4)])
    fig.show()


if __name__ == '__main__':
    task_1()
    task_2()
