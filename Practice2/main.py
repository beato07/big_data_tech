import time

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.datasets import fetch_openml, load_wine
from sklearn.manifold import TSNE

# Проверяем доступность модуля UMAP
UMAP_AVAILABLE = True
try:
    import umap
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP модуль не найден. Задание 7 будет недоступно.")


def load_wine_dataset():
    """
    Загружает данные wine dataset.

    Возвращает DataFrame с данными о вине.
    """

    data = load_wine(as_frame=True)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df


def load_clean_wine_data(df):
    """
    Загружает и очищает данные wine dataset.

    Проверяет наличие пустых значений и удаляет строки с пропущенными данными.
    Соответствует заданию 2: проверка данных на пустые значения и их обработка.
    """

    if df.isna().any().any():
        df_cleaned = df.dropna()
        print(f"Rows deleted: {len(df_cleaned)}")
    return df.info()


def draw_bar_chart():
    """
    Строит столбчатую диаграмму с использованием Plotly.

    Соответствует заданию 3: построение столбчатой диаграммы с заданными параметрами.
    Включает: цветовое кодирование столбцов, черные границы толщиной 2px,
    заголовок по центру размером 20px, подписи осей размером 16px,
    поворот меток на 315°, сетку цвета ivory толщиной 2px, высоту 700px.
    """

    labels = ['Oxygen', 'Hydrogen', 'Carbon_Dioxide', 'Nitrogen']
    values = [4500, 2500, 1053, 500]

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
    """
    Строит круговую диаграмму с использованием Plotly.

    Соответствует заданию 4: построение круговой диаграммы.
    Использует данные из столбчатой диаграммы, черные границы толщиной 2px,
    читаемые категории, заголовок по центру, высота 700px.
    """

    labels = ['Oxygen', 'Hydrogen', 'Carbon_Dioxide', 'Nitrogen']
    values = [4500, 2500, 1053, 500]

    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    fig = go.Figure(data=[go.Pie(
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
    """
    Настраивает внешний вид линейного графика.

    Соответствует заданию 5.1 и 5.2: стилизация линейных графиков.
    Применяет цвет линии crimson, белые точки с черными границами толщиной 2px,
    сетку цвета mistyrose толщиной 2px.
    """

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
    """
    Строит линейные графики зависимости показателей от содержания алкоголя.

    Соответствует заданию 5: построение линейных графиков с изучением зависимостей.
    Анализирует зависимость 3 параметров (malic_acid, proanthocyanins, proline)
    от содержания алкоголя в вине с использованием matplotlib.
    """

    df = load_wine_dataset()

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
    """
    Выполняет визуализацию многомерных данных с использованием t-SNE.

    Соответствует заданию 6: визуализация MNIST с t-SNE для разных значений перплексии.
    Использует выборку из 1000 изображений MNIST, тестирует перплексию 5, 25, 50,
    измеряет время выполнения алгоритма.
    """
    mnist = fetch_openml('mnist_784', version=1)
    x, y = mnist.data, mnist.target

    random_img = np.random.choice(len(x), 1000, replace=False)
    x_sample = x.iloc[random_img]
    y_sample = y[random_img]

    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x_sample)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    start_time = time.time()
    perplexity = [5, 25, 50]
    for i, perplexity in enumerate(perplexity):
        tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=123)
        tsne_features = tsne_model.fit_transform(x_scaled)

        mnist_df = pd.DataFrame(data=tsne_features, columns=['x', 'y'])
        mnist_df['label'] = y_sample.values

        sns.scatterplot(x='x', y='y', hue='label', data=mnist_df, palette='bright', ax=ax[i])

        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].set_title(f'perplexity={perplexity}')

    plt.tight_layout()
    plt.show()

    end_time = time.time()
    print(f'TSNE total time: {end_time - start_time}')


def plot_umap():
    """
    Выполняет визуализацию многомерных данных с использованием UMAP.

    Соответствует заданию 7: визуализация с UMAP для разных параметров n_neighbors и min_dist.
    Тестирует различные комбинации параметров: n_neighbors (5, 25, 50) и min_dist (0.1, 0.6),
    измеряет время выполнения для сравнения с t-SNE.
    """
    mnist = fetch_openml('mnist_784', version=1)
    x, y = mnist.data, mnist.target

    random_img = np.random.choice(len(x), 1000, replace=False)
    x_sample = x.iloc[random_img]
    y_sample = y[random_img]

    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x_sample)

    n_n = (5, 25, 50)
    m_d = (0.1, 0.6)

    fig, ax = plt.subplots(len(n_n), len(m_d), figsize=(12, 18))

    start_time = time.time()
    for i, n_neighbors in enumerate(n_n):
        for j, min_dist in enumerate(m_d):
            embedding = umap.UMAP(
                n_neighbors=n_neighbors, min_dist=min_dist, random_state=123
            ).fit_transform(x_scaled)

            df_um = pd.DataFrame({
                'x': embedding[:, 0],
                'y': embedding[:, 1],
                'label': y_sample.values
            })

            sns.scatterplot(
                x='x',
                y='y',
                hue='label',
                data=df_um,
                ax=ax[i, j]
            )
            ax[i, j].set_xlabel('')
            ax[i, j].set_ylabel('')
            ax[i, j].set_title(f'n_neighbors={n_neighbors}, min_dist={min_dist},')

    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    plt.show()

    end_time = time.time()
    print(f'UMAP total time: {end_time - start_time}')


def show_menu():
    """Отображает меню выбора заданий и обрабатывает пользовательский ввод."""

    print("\n" + "=" * 50)
    print("МЕНЮ ВЫБОРА ЗАДАНИЙ")
    print("=" * 50)
    print("1. Загрузить и показать данные wine dataset")
    print("2. Проверить данные на пустые значения")
    print("3. Построить столбчатую диаграмму (Plotly)")
    print("4. Построить круговую диаграмму (Plotly)")
    print("5. Построить линейные графики (Matplotlib)")
    print("6. Визуализация t-SNE (MNIST)")
    if UMAP_AVAILABLE:
        print("7. Визуализация UMAP (MNIST)")
        print("8. Выполнить все задания")
    else:
        print("7. Визуализация UMAP (НЕДОСТУПНО - модуль не установлен)")
        print("8. Выполнить все доступные задания")
    print("0. Выход")
    print("=" * 50)


def task_1():
    """Выполняет задание 1: загрузка и отображение данных."""

    print("\nВыполняется задание 1: Загрузка многомерных данных")
    df = load_wine_dataset()
    print(f"Загружены данные wine dataset:")
    print(f"Размер данных: {df.shape}")
    print(f"Столбцы: {list(df.columns)}")
    print(df.head())
    return df


def task_2():
    """Выполняет задание 2: анализ данных."""

    print("\nВыполняется задание 2: Анализ данных")
    df = load_wine_dataset()
    print("Информация о данных:")
    load_clean_wine_data(df)


def task_3():
    """Выполняет задание 3: столбчатая диаграмма."""

    print("\nВыполняется задание 3: Столбчатая диаграмма")
    draw_bar_chart()


def task_4():
    """Выполняет задание 4: круговая диаграмма."""

    print("\nВыполняется задание 4: Круговая диаграмма")
    draw_pie_chart()


def task_5():
    """Выполняет задание 5: линейные графики."""

    print("\nВыполняется задание 5: Линейные графики")
    draw_line_graph()


def task_6():
    """Выполняет задание 6: t-SNE визуализация."""

    print("\nВыполняется задание 6: t-SNE визуализация")
    plot_tsne()


def task_7():
    """Выполняет задание 7: UMAP визуализация."""

    if not UMAP_AVAILABLE:
        print("\nЗадание 7 недоступно: модуль UMAP не установлен.")
        print("Установите UMAP командой: pip install umap-learn")
        return

    print("\nВыполняется задание 7: UMAP визуализация")
    plot_umap()


def task_all():
    """Выполняет все задания последовательно."""

    if UMAP_AVAILABLE:
        print("\nВыполняются все задания:")
    else:
        print("\nВыполняются все доступные задания (кроме UMAP):")

    task_1()
    task_2()
    task_3()
    task_4()
    task_5()
    task_6()
    if UMAP_AVAILABLE:
        task_7()
    else:
        print("\nЗадание 7 (UMAP) пропущено - модуль не установлен.")


def main():
    """
    Основная функция с интерактивным меню.
    Предоставляет пользователю выбор заданий через консольное меню.
    """

    task_functions = {
        '1': task_1,
        '2': task_2,
        '3': task_3,
        '4': task_4,
        '5': task_5,
        '6': task_6,
        '7': task_7,
        '8': task_all
    }

    while True:
        show_menu()
        choice = input("\nВведите номер задания (0-8): ").strip()

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
