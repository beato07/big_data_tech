import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def task_1():
    """
    Находит и интерпретирует корреляцию между переменными "Улица" и "Гараж".
    Строит диаграмму рассеяния для вышеупомянутых переменных.
    """
    street = np.array([80, 98, 75, 91, 78])
    garage = np.array([100, 82, 105, 89, 102])

    print(f"\nКорреляция Пирсона между 'Улица' и 'Гараж': {np.corrcoef(street, garage)[0, 1]}")

    plt.scatter(street, garage, color='crimson')
    plt.title('Диаграмма рассеяния', fontsize=20)
    plt.xlabel('Улица (количество автомобилей)')
    plt.ylabel('Гараж (количество автомобилей)')
    plt.grid()
    plt.tight_layout()
    plt.show()


def task_2():
    """
    Выводит данные, проводит предобработку и описывает признаки.
    Строит корреляционную матрицу по одной целевой переменной.
    Реализует и визуализирует регрессию на графике.
    Отображает наклон, сдвиг и MSE.
    """
    sl = pd.read_csv('insurance.csv')
    print(f"\nРазмерность данных: {sl.shape}")
    print(f"\nКолонки: {list(sl.columns)}")
    print(f"\nПроверка на пропущенные значения:\n{sl.isnull().sum()}")

    df = pd.DataFrame(sl , columns=['age', 'bmi', 'children', 'charges'])
    corr_matrix = df.corr().round(3)

    print(f"\nМАТРИЦА КОРРЕЛЯЦИИ:\n{corr_matrix}")
    print(f"\nНАИБОЛЕЕ КОРРЕЛИРУЮЩАЯ ПЕРЕМЕННАЯ С CHARGES: {corr_matrix['charges'].nlargest(2).index[-1]}")

    model = LinearRegression()
    x = df[['age']]
    y = df['charges']
    model.fit(x,y)

    model_y_sk = model.coef_[0] * x + model.intercept_
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, model_y_sk, linewidth=2, color='r',
             label=f'linear_model = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
    plt.scatter(x, y, alpha=0.7)
    plt.grid()
    plt.xlabel('age')
    plt.ylabel('charges')
    plt.legend()
    plt.show()

    print(f"\n=== ПАРАМЕТРЫ РЕГРЕССИОННОЙ МОДЕЛИ ===")
    print(f"    Угол наклона: {model.coef_[0]:.4f}")
    print(f"    Коэффициент сдвига: {model.intercept_:.2f}")
    print(f"    MSE: {mean_squared_error(model_y_sk, y):.2f}")


if __name__ == '__main__':
    task_1()
    task_2()
