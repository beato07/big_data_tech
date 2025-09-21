import pandas as pd
from sklearn.datasets import load_wine
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
from sklearn import preprocessing


def load_clean_wine_data(df):
    if df.isna().any().any():
        df_cleaned = df.dropna()
        print(f"Rows deleted: {len(df_cleaned)}")
    return df.info()


def draw_bar_chart():
    labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
    values = [4500,2500,1053,500]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values,
                         marker=dict(color=values,
                                     coloraxis='coloraxis',
                                     line=dict(color='#000000', width=2))))
    fig.update_layout(
        title='Bar chart',
        title_x=0.5,
        title_font_size=20,
        xaxis_title='Gas', xaxis_title_font_size=14,
        yaxis_title='Volume', yaxis_title_font_size=14,
        height=700)
    fig.update_xaxes(
        tickangle=315,
        gridwidth=2,
        gridcolor='ivory')
    fig.update_yaxes(
        gridwidth=2,
        gridcolor='ivory')
    fig.show()


def draw_pie_chart():
    labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
    values = [4500,2500,1053,500]

    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    fig = go.Figure(data = [go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors,
                    line=dict(color='#000000', width=2)))])
    fig.update_layout(
        title='Pie chart',
        title_x=0.5,
        title_font_size=20,
        height=700,
        font_size=14)
    fig.show()


def customize_plot(ax, x_values, y_values, y_label):
    ax.plot(x_values, y_values,
            c='crimson',
            marker='o',
            markerfacecolor='white',
            markeredgecolor='black',
            markeredgewidth=2)

    ax.set_xlabel('Alcohol')
    ax.set_ylabel(f'{y_label.title()}')
    ax.set_title(f'{y_label.title()} dependence on alcohol')

    ax.grid(True, color='mistyrose', linewidth=2)


def draw_line_graph():
    data = load_wine(as_frame=True)
    df = data.data

    # Сортируем по alcohol
    sorted_df = df.sort_values(by=['alcohol'])
    x_values = sorted_df['alcohol']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

    y_values = sorted_df['malic_acid']
    ax1.scatter(x_values, y_values)
    customize_plot(ax1, x_values, y_values, 'malic acid')

    y_values = sorted_df['proanthocyanins']
    ax2.scatter(x_values, y_values)
    customize_plot(ax2, x_values, y_values, 'proanthocyanins')

    y_values = sorted_df['proline']
    ax3.scatter(x_values, y_values)
    customize_plot(ax3, x_values, y_values, 'proline')

    plt.tight_layout()
    plt.show()


def plot_tsne():
    mnist = fetch_openml('mnist_784', version=1)
    x, y = mnist.data, mnist.target

    random_images = np.random.choice(len(x), 1000, replace=False)
    x_sample = x.iloc[random_images]
    y_sample = y[random_images]

    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x_sample)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    perplexity = [5, 25, 50]
    for i, perplexity in enumerate(perplexity):
        T = TSNE(n_components=2, perplexity=perplexity, random_state=123)
        TSNE_features = T.fit_transform(x_scaled)

        mnist_df = pd.DataFrame(data=TSNE_features, columns=['x', 'y'])
        mnist_df['label'] = y_sample.values

        sns.scatterplot(x='x', y='y', hue='label', data=mnist_df, palette='bright', ax=axes[i])
        axes[i].set_title(f'perplexity={perplexity}')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Выгрузка многомерных данных
    data = load_wine(as_frame=True)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    print(df)

    # Вывод информации о данных, используя info(), info()
    print(load_clean_wine_data(df))

    # Построение столбчатой диаграммы
    draw_bar_chart()

    # Построение круговой диаграммы
    draw_pie_chart()

    # Построение линейного графика
    draw_line_graph()

    # T-SNE визуализация для набора данных MNIST
    plot_tsne()