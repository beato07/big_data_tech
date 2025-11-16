import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def task_1():
    '''
    Строит гистограмму, которая показывает баланс классов.
    '''
    data = load_wine()
    y = data.target
    class_names = data.target_names

    class_counts = [sum(y == i) for i in range(len(class_names))]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(class_names, class_counts, color=['skyblue', 'lightgreen', 'salmon'])

    plt.title('Баланс классов в датасете Wine', fontsize=14)
    plt.xlabel('Классы', fontsize=12)
    plt.ylabel('Количество образцов', fontsize=12)

    for bar, count in zip(bars, class_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=12)

    plt.grid(axis='y', alpha=0.5, linestyle='--')
    plt.ylim(0, max(class_counts) + 5)
    plt.show()

    print(data.target_names)


def task_2():
    '''
    Разбивает выборку на тренировочную и тестовую.
    Тренировочная для обучения модели, тестовая для проверки ее качества.
    '''
    data = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target,
                                                        train_size=0.8, shuffle=True, random_state=271)

    print(f'Размер для признаков обучающей выборки {x_train.shape}\n'
          f'Размер для признаков тестовой выборки {x_test.shape}\n'
          f'Размер для целевого показателя обучающей выборки {y_train.shape}\n'
          f'Размер для показателя тестовой выборки {y_test.shape}')


if __name__ == '__main__':
    task_1()
    task_2()
