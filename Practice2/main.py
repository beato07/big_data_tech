import pandas as pd
from sklearn.datasets import load_wine
import plotly.graph_objects as go


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
