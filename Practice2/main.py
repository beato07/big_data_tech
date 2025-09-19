import pandas as pd
from sklearn.datasets import load_wine
import plotly.graph_objects as go

import matplotlib.pyplot as plt


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

    fig1, ax1 = plt.subplots()
    y_values = sorted_df['malic_acid']
    ax1.scatter(x_values, y_values)
    customize_plot(ax1, x_values, y_values, 'malic acid')
    plt.show()

    fig2, ax2 = plt.subplots()
    y_values = sorted_df['proanthocyanins']
    ax2.scatter(x_values, y_values)
    customize_plot(ax2, x_values, y_values, 'proanthocyanins')
    plt.show()

    fig3, ax3 = plt.subplots()
    y_values = sorted_df['proline']
    ax3.scatter(x_values, y_values)
    customize_plot(ax3, x_values, y_values, 'proline')
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
