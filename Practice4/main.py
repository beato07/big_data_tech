import numpy as np
import matplotlib.pyplot as plt


def task_1():
    '''
    Находит и интерпретирует корреляцию между переменными "Улица" и "Гараж".
    Строит диаграмму рассеяния для вышеупомянутых переменных.
    '''
    street = np.array([80, 98, 75, 91, 78])
    garage = np.array([100, 82, 105, 89, 102])

    print(f"Корреляция Пирсона между 'Улица' и 'Гараж': {np.corrcoef(street, garage)[0, 1]}")

    plt.scatter(street, garage, color='crimson')
    plt.title('Диаграмма рассеяния', fontsize=20)
    plt.xlabel('Улица (количество автомобилей)')
    plt.ylabel('Гараж (количество автомобилей)')
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    task_1()
