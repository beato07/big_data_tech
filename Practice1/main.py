from math import pi
from sklearn.datasets import fetch_california_housing
import pandas as pd


def get_figure_area(figures, name, a, b, r):
    if name.lower() == "треугольник":
        figures |= {f"{name.capitalize()}": 0.5 * a * b}
    elif name.lower() == "прямоугольник":
        figures |= {f"{name.capitalize()}": a * b}
    elif name.lower() == "круг":
        figures |= {f"{name.capitalize()}": pi * pow(r, 2)}


def print_areas():
    figures = {}
    exit = ""
    while exit != 'n':
        name = input("Введите имя фигуры: ")
        data = input("Введите три целых числа через пробел (Сторона a, сторона b, радиус (только для круга): ")
        try:
            a, b, r = map(float, data.split())
            get_figure_area(figures, name, a, b, r)
        except ValueError:
            print("Ошибка: необходимо ввести ровно три числа...")
        exit = input("Продолжить заполнение словаря? (y/n): ")
    print(*[f"{key}: {value}" for key, value in figures.items()], sep='\n')


def calculate(x, y, op):
    if op == "+":
        print(x + y)
        return x + y
    elif op == "-":
        print(x - y)
        return x - y
    elif op == "/":
        print(x / y)
        return x / y
    elif op == "//":
        print(x // y)
        return x // y
    elif op == "abs":
        print(abs(x), abs(y))
        return abs(x), abs(y)
    elif op == "**":
        print(x ** y)
        return x ** y


def sum_squares(num):
    print(sum(value ** 2 for value in num))
    return sum(value ** 2 for value in num)


def get_sequence(N):
    num = []
    cnt = 0
    for i in range(1, N + 1):
        for j in range(i):
            if cnt != N:
                num.append(i)
                cnt += 1
            else:
                break
    print(*num)
    return num


def get_letters_sum(A, B):
    if len(A) != len(B):
        print("Списки не равны...")
        return -1
    res = {}
    for letter, value in zip(B, A):
        res[letter] = res.get(letter, 0) + value
    print(res)
    return res


def avg(df):
    return df.mean()


def str_to_morse(message):
    morse = {'a': '.-', 'b': '-…', 'c': '-.-.', 'd': '-..',
             'e': '.', 'f': '..-.', 'g': '--.', 'h': '….',
             'i': '..', 'j': '.---', 'k': '-.-', 'l': '.-..',
             'm': '--', 'n': '-.', 'o': '---', 'p': '.--.',
             'q': '--.-', 'r': '.-.', 's': '…', 't': '-',
             'u': '..-', 'v': '…-', 'w': '.--', 'x': '-..-',
             'y': '-.--', 'z': '--..'}
    morse_code = ""
    for char in message.lower():
        if char == ' ':
            morse_code += '\n'
        elif char in morse:
            morse_code += morse[char] + ' '
    print(morse_code)
    return morse_code


if __name__ == '__main__':
    # Площадь фигур
    print_areas()

    # Различные операции с двумя переменными
    res1 = calculate(-42, -52, 'abs')

    # Сумма квадратов значений списка
    num = list(map(int, input("Введите числа через пробел, пока их сумма не будет равна нулю: ").split()))
    while sum(num) != 0:
        num.append(int(input("Сумма элементов списка не равна нулю. Введите дополнительные значения в список: ")))
    res2 = sum_squares(num)

    # Последовательность чисел, длинною N
    res3 = get_sequence(7)

    # Сумма букв на основе двух списков
    res4 = get_letters_sum([1, 2, 3, 4, 2, 1, 3, 4, 5, 6, 5, 4, 3, 2], ['a', 'b', 'c', 'c', 'c', 'b', 'a', 'c', 'a', 'a', 'b', 'c', 'b', 'a'])

    # Операции с датасетом
    data = fetch_california_housing(as_frame=True) #as_frame=True - данные должны быть возвращены в виде Pandas DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Метод info()
    df.info()

    # Проверка на пропущенные значения
    print(df.isna().sum())

    # Вывод записей, где возраст домов > 50 и население более 2500 человек
    print(df.loc[(df['HouseAge'] > 50) & (df['Population'] > 2500), ['HouseAge', 'Population']])

    # Максимальное и минимальное значение медианной стоимости дома
    df['MedHouseVal'] = data.target
    print(f"{min(df['MedHouseVal']) * 100000:.2f}$, {max(df['MedHouseVal']) * 100000:.2f}$")

    # Метод apply()
    print(df.apply(avg, axis=0))

    # Текст в азбуку морзе
    res5 = str_to_morse("Help me")