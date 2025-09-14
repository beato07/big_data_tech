import pandas as pd
from sklearn.datasets import load_wine
import plotly.graph_objects as go


data = load_wine(as_frame=True)
df = pd.DataFrame(data.data, columns=data.feature_names)


def load_wine_data():
    print(df)
    return df


def load_clean_wine_data():
    if df.isna().any().any():
        df_cleaned = df.dropna()
        print(f"Rows deleted: {len(df_cleaned)}")
    print(df.info())
    print(df.info())


def draw_bar_chart():
    x_data = ['14.09.2015', '06.06.2017', '01.03.2020', '20.07.2023']
    y_date = [1813.70, 4206.14, 2750.52, 749.99]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_data, y=y_date,
                         marker=dict(color=y_date,
                                     coloraxis='coloraxis',
                                     line=dict(color='black', width=2))))
    fig.update_layout(
        title="Diagram",
        title_x=0.5,
        title_font_size=20,
        xaxis_title = 'Date', xaxis_title_font_size=14,
        yaxis_title='Value', yaxis_title_font_size=14,
        height=700)
    fig.update_xaxes(
        tickangle=315,
        gridwidth=2,
        gridcolor='ivory')
    fig.update_yaxes(
        gridwidth=2,
        gridcolor='ivory')
    fig.show()


if __name__ == '__main__':

    # Выгрузка многомерных данных
    res1 = load_wine_data()

    # Вывод информации о данных, используя info(), info()
    load_clean_wine_data()

    # Построение столбчатой диаграммы
    draw_bar_chart()
