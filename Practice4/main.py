import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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


def task_3():
    """
    Выполняет комплексный статистический анализ влияния региона и пола на индекс массы тела (BMI).

    Функция проводит серию статистических тестов для исследования взаимосвязи между
    географическим регионом, полом и показателями BMI на данных медицинского страхования.

    Анализ включает:
    1. Предварительный анализ данных (размерность, пропущенные значения, уникальные регионы)
    2. Однофакторный ANOVA тест для проверки влияния региона на BMI
    3. Двухфакторный ANOVA тест для проверки влияния региона и пола на BMI
    4. Попарные сравнения t-критерием Стьюдента между регионами
    5. Post-hoc тесты Тьюки для детального анализа различий
    6. Визуализацию результатов
    """
    sl = pd.read_csv('insurance.csv')
    print(f"\nРазмерность данных: {sl.shape}")
    print(f"\nКолонки: {list(sl.columns)}")
    print(f"\nПроверка на пропущенные значения:\n{sl.isnull().sum()}")
    print(f"\nСписок уникальных  регионов: {list(sl['region'].unique())}")

    groups = sl.groupby('region')['bmi'].apply(list)

    southwest = groups['southwest']
    southeast = groups['southeast']
    northwest = groups['northwest']
    northeast = groups['northeast']

    print(sl[['region', 'bmi']].head(), end="\n\n")

    print(f"=== ANOVA тест через через библиотеку Scipy ===")
    result = stats.f_oneway(southwest, southeast, northwest, northeast)
    print(f"F-статистика: {result.statistic:.4f}")
    print(f"P-значение: {result.pvalue:.30f}")

    model = ols('age ~ bmi', data=sl).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)

    print(f"\n=== ANOVA тест через библиотеку Statsmodels ===")
    print(anova_results, end='\n\n')

    print(f"=== Перебор пар с помощью t критерия Стьюдента ===")
    region_pairs = []
    regions = list(sl['region'].unique())
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            region_pairs.append((regions[i], regions[j]))

    for region1, region2 in region_pairs:
        print(region1, region2)
        print(stats.ttest_ind(groups[region1], groups[region2]))

    tukey = pairwise_tukeyhsd(endog=sl['bmi'], groups=sl['region'], alpha=0.05)

    tukey.plot_simultaneous()
    plt.vlines(x=31.5, ymin=-0.5, ymax=3.5, colors='r')
    plt.title('Post-hoc тест Тьюки: сравнение BMI по регионам')
    plt.tight_layout()
    plt.show()

    print(f"\n{tukey.summary()}")

    model = ols('bmi ~ C(region) + C(sex) + C(region):C(sex)', data=sl).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)
    print(f"\n=== Двухфакторный ANOVA тест через библиотеку Statsmodels ===")
    print(anova_results, end='\n\n')

    sl['combination'] = sl['region'] + " / " + sl['sex']

    tukey = pairwise_tukeyhsd(endog=sl['bmi'], groups=sl['combination'], alpha=0.05)

    tukey.plot_simultaneous()
    plt.title('Post-hoc тест Тьюки: сравнение BMI по комбинациям региона и пола')
    plt.tight_layout()
    plt.show()

    print(tukey.summary())


if __name__ == '__main__':
    #task_1()
    #task_2()
    task_3()
