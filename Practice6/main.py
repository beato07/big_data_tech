# Стандартные библиотеки Python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Машинное обучение (scikit-learn)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Визуализация
import plotly.graph_objects as go


def show_menu():
    print("\n" + "=" * 50)
    print("МЕНЮ ВЫБОРА ЗАДАНИЙ")
    print("=" * 50)
    print("1. K-MEANS - Кластеризация с автоматическим подбором кластеров.")
    print("2. АГЛОМЕРАТИВНАЯ КЛАСТЕРИЗАЦИЯ - Иерархический метод.")
    print("3. DBSCAN - Плотностная кластеризация.")
    print("0. Выход")
    print("=" * 50)


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

    model12 = AgglomerativeClustering(3, compute_distances=True)
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


def task_3():
    df = pd.read_csv('insurance.csv')

    df_clean = df.drop_duplicates()
    print(f'Number of unique values: {len(df_clean)}')

    features = df_clean[['age', 'bmi', 'children']]

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    features_df = pd.DataFrame(features_normalized, columns=features.columns)
    print("\nСтатистики после нормализации:")
    print(features_df.describe())

    model13 = DBSCAN(eps=0.5, min_samples=5).fit(features_df)
    features_df['Claster'] = model13.labels_

    fig = go.Figure(data=[go.Scatter3d(
        x=features_df['age'],
        y=features_df['bmi'],
        z=features_df['children'],
        mode='markers',
        marker_color=features_df['Claster'],
        marker_size=4)])

    fig.show()


def main():
    task_functions = {
        '1': task_1,
        '2': task_2,
        '3': task_3
    }

    while True:
        show_menu()
        choice = input("\nВведите номер задания (0-3): ").strip()

        if choice == '0':
            print("Выход из программы.")
            break
        elif choice in task_functions:
            task_functions[choice]()
        else:
            print("Неверный выбор. Попробуйте снова.")

        input("\nНажмите Enter для продолжения...")


if __name__ == '__main__':
    main()
