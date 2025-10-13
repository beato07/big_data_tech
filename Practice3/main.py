import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def show_menu():
    print("\n" + "=" * 50)
    print("МЕНЮ ВЫБОРА ЗАДАНИЙ")
    print("=" * 50)
    print("1. Загрузить данные из файла \"insurance.csv\".")
    print("2. С помощью метода describe() посмотреть статистику по данным.")
    print("3. Построить гистограммы для числовых показателей.")
    print("4. Найти меры центральной тенденции и меры разброса для индекса массы тела (bmi) и расходов (charges).")
    print("5. Построить box-plot для числовых показателей.")
    print("6. Используя признак charges или imb, проверить, выполняется ли центральная предельная теорема.")
    print("7. Построить 95% и 99% доверительный интервал для среднего значения расходов и среднего значения индекса массы тела.")
    print("8. Проверить распределения следующих признаков на нормальность: индекс массы тела, расходы.")
    print("9. Загрузить данные из файла \"ECDCCases.csv\".")
    print("10. Проверить в данных наличие пропущенных значений. Вывести количество пропущенных значений в процентах.")
    print("11. Посмотреть статистику по данным, используя describe(). Посмотреть, для каких стран количество смертей в день превысило 3000 и сколько таких дней было.")
    print("12. Найти дублирование данных. Удалить дубликаты.")
    print("13. Загрузить данные из файла “bmi.csv”. Взять оттуда две выборки.")
    print('14. Кубик бросили 600 раз. С помощью критерия Хи-квадрат проверить, является ли полученное распределение равномерным.')
    print('15. С помощью критерия Хи-квадрат проверить, являются ли переменные зависимыми.')
    print('16. Выполняет все задания последовательно.')
    print("0. Выход")
    print("=" * 50)


def task_1():
    df = pd.read_csv('insurance.csv')
    print(df)


def task_2():
    df = pd.read_csv('insurance.csv')
    print(df.describe())


def task_3():
    df = pd.read_csv('insurance.csv')
    df.hist(color='green', edgecolor='black')
    plt.tight_layout()
    plt.show()


def task_4():
    df = pd.read_csv('insurance.csv')
    print('\nМеры центральной тенденции для индекса массы тела (bmi) и расходов (charges)')

    moda = df[['bmi', 'charges']].mode().iloc[0]
    med = df[['bmi', 'charges']].median()
    mean = df[['bmi', 'charges']].mean()

    print(f'\nМода:     \n{moda.to_string()}\n'
          f'\nМедиана:  \n{med.to_string()}\n'
          f'\nСреднее:  \n{mean.to_string()}')

    print('\nМеры разброса для индекса массы тела (bmi) и расходов (charges)')

    raz = df[['bmi', 'charges']].max() - df[['bmi', 'charges']].min()
    std = df[['bmi', 'charges']].std()
    Q1 = df[['bmi', 'charges']].quantile(0.25)
    Q3 = df[['bmi', 'charges']].quantile(0.75)
    IQR = Q3 - Q1

    print(f'\nРазмах:                   \n{raz.to_string()}\n'
          f'\nСтандартное отклонение:   \n{std.to_string()}\n'
          f'\nМежквартильный размах:    \n{IQR.to_string()}\n')


def task_5():
    df = pd.read_csv('insurance.csv')
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].boxplot(df['age'], tick_labels=['age'], vert=False, patch_artist=True)
    ax[0, 0].grid(True)
    ax[0, 1].boxplot(df['bmi'], tick_labels=['bmi'], vert=False, patch_artist=True)
    ax[0, 1].grid(True)
    ax[1, 0].boxplot(df['children'], tick_labels=['children'], vert=False, patch_artist=True)
    ax[1, 0].grid(True)
    ax[1, 1].boxplot(df['charges'], tick_labels=['charges'], vert=False, patch_artist=True)
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def task_6():
    df = pd.read_csv('insurance.csv')

    sample_sizes = [5, 30, 100, 500]  # Размеры выборок
    n_samples = 300 # Количество выборок

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.ravel()

    for i, n in enumerate(sample_sizes):
        means = []
        for _ in range(n_samples):
            sample = np.random.choice(df['charges'], size=n)
            means.append(np.mean(sample))

        print(f'Стандартное отклонение = {np.std(means):.2f}, Среднее = {np.mean(means):.2f}')

        ax[i].hist(means, bins=15, color='green', edgecolor='black')
        ax[i].set_title(f'Выборка = {n}')
        ax[i].set_xlabel('Выборочное среднее')
        ax[i].set_ylabel('Частота')

    plt.tight_layout()
    plt.show()


def task_7():
    df = pd.read_csv('insurance.csv')

    mean_charges = df['charges'].mean()
    mean_bmi = df['bmi'].mean()

    ci_95_c = stats.t.interval(0.95, len(df['charges']) - 1, loc=mean_charges, scale=stats.sem(df['charges']))
    ci_99_c = stats.t.interval(0.99, len(df['charges']) - 1, loc=mean_charges, scale=stats.sem(df['charges']))

    ci_95_b = stats.t.interval(0.95, len(df['bmi']) - 1, loc=mean_bmi, scale=stats.sem(df['bmi']))
    ci_99_b = stats.t.interval(0.99, len(df['bmi']) - 1, loc=mean_bmi, scale=stats.sem(df['bmi']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(['95% ДИ', '99% ДИ'], [mean_charges, mean_charges],
                 yerr=[[mean_charges - ci_95_c[0], mean_charges - ci_99_c[0]],
                       [ci_95_c[1] - mean_charges, ci_99_c[1] - mean_charges]],
                 fmt='o', capsize=5)
    ax1.set_title('Доверительные интервалы для charges')
    ax1.set_ylabel('Charges')
    ax1.grid(True)

    ax2.errorbar(['95% ДИ', '99% ДИ'], [mean_bmi, mean_bmi],
                yerr=[[mean_bmi - ci_95_b[0], mean_bmi - ci_99_b[0]],
                      [ci_95_b[1] - mean_bmi, ci_99_b[1] - mean_bmi]],
                fmt='o', capsize=5)
    ax2.set_title('Доверительные интервалы для BMI')
    ax2.set_ylabel('BMI')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def task_8():
    df = pd.read_csv('insurance.csv')

    print(stats.kstest(df['bmi'], 'norm', args=(df['bmi'].mean(), df['bmi'].std())))
    print(stats.kstest(df['charges'], 'norm', args=(df['charges'].mean(), df['charges'].std())))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    stats.probplot(df['bmi'], dist='norm', plot=ax1)
    plt.title('Q-Q plot: BMI')

    stats.probplot(df['charges'], dist='norm', plot=ax2)
    plt.title('Q-Q plot: Charges')

    plt.tight_layout()
    plt.show()


def task_9():
    df = pd.read_csv('ECDCCases.csv')
    print(df)


def task_10():
    df = pd.read_csv('ECDCCases.csv')

    print(f'Кол-во пропущенных значений составляет {(df.isnull().sum().sum() / df.size) * 10000:.2f}%')

    missing_sorted = (df.isnull().sum()).sort_values(ascending=False)
    columns_to_remove = missing_sorted.head(2).index.tolist()
    df = df.drop(columns_to_remove, axis=1)

    categorical_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            df.fillna({column: 'other'}, inplace=True)

    for column in numeric_columns:
        if df[column].isnull().sum() > 0:
            df.fillna({column: df[column].median()}, inplace=True)

    print(df.isnull().sum())


def task_11():
    df = pd.read_csv('ECDCCases.csv')
    high_deaths = df[df['deaths'] > 3000]
    country_high_deaths = high_deaths.groupby('countriesAndTerritories').size().reset_index(name='days')
    print(country_high_deaths)


def task_12():
    df = pd.read_csv('ECDCCases.csv')
    df_cleaned = df.drop_duplicates()
    print(df_cleaned)


def task_13():
    df = pd.read_csv('bmi.csv')
    northwest_bmi = df[df['region'] == 'northwest']['bmi']
    southwest_bmi = df[df['region'] == 'southwest']['bmi']

    print("\n=== ПРОВЕРКА НА НОРМАЛЬНОСТЬ (Шапиро-Уилк) ===")
    stat_nw, p_nw = stats.shapiro(northwest_bmi)
    stat_sw, p_sw = stats.shapiro(southwest_bmi)
    print(f"Статистика: {stat_nw} , {stat_sw}\np-значение: {p_nw}, {p_sw}")

    print("\n=== ПРОВЕРКА ГОМОГЕННОСТИ ДИСПЕРСИЙ (Бартлетт) ===")
    bartlett_test, p_value = stats.bartlett(northwest_bmi, southwest_bmi)
    print(f"Статистика: {bartlett_test}\np-значение: {p_value}")

    print("\n=== T-КРИТЕРИЙ СТЬЮДЕНТА ===")
    t_stat, p_value = stats.ttest_ind(northwest_bmi, southwest_bmi, equal_var=True)
    print(f"Статистика: {t_stat}\np-значение: {p_value}")


def task_14():
    print("\n=== КРИТЕРИЙ ХИ-КВАДРАТ ===")
    data = [97, 98, 109, 95, 97, 104]
    stat, p_value = stats.chisquare(data)
    print(f"Статистика: {stat}\np-значение: {p_value}")


def task_15():
    data = pd.DataFrame({'Женат': [89, 17, 11, 43, 22, 1],
                  'Гражданский брак': [80, 22, 20, 35, 6, 4],
                  'Не состоит в отношениях': [35, 44, 35, 6, 8, 22]})

    data.index = ['Полный рабочий день', 'Частичная занятость',
                  'Временно не работает', 'На домохозяйстве',
                  'На пенсии', 'Учёба']

    stat, p_value = stats.chi2_contingency(data)[:2]

    print(f"Хи-квадрат: {stat:.2f}")
    print(f"p-value: {p_value:.25f}")


def task_all():
    task_1()
    task_2()
    task_3()
    task_4()
    task_5()
    task_6()
    task_7()
    task_8()
    task_9()
    task_10()
    task_11()
    task_12()
    task_13()
    task_14()
    task_15()


def main():
    task_functions = {
        '1': task_1,
        '2': task_2,
        '3': task_3,
        '4': task_4,
        '5': task_5,
        '6': task_6,
        '7': task_7,
        '8': task_8,
        '9': task_9,
        '10': task_10,
        '11': task_11,
        '12': task_12,
        '13': task_13,
        '14': task_14,
        '15': task_15,
        '16': task_all
    }

    while True:
        show_menu()
        choice = input("\nВведите номер задания (0-16): ").strip()

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
