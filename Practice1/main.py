from math import pi


def get_figure_area(figures, figure_name, a, b, r):
    if figure_name.lower() == "треугольник":
        figures |= {f"{figure_name.capitalize()}": 0.5 * a * b}
    elif figure_name.lower() == "прямоугольник":
        figures |= {f"{figure_name.capitalize()}": a * b}
    elif figure_name.lower() == "круг":
        figures |= {f"{figure_name.capitalize()}": pi * pow(r, 2)}


def print_areas():
    dictionary = {}
    exit = ""
    while exit != "n":
        figure_name = input("Введите имя фигуры: ")
        data = input("Введите три целых числа через пробел (Сторона a, сторона b, радиус (только для круга): ")
        try:
            a, b, r = map(float, data.split())
            get_figure_area(dictionary, figure_name, a, b, r)
        except ValueError:
            print("Ошибка: необходимо ввести ровно три числа...")
        exit = input("Продолжить заполнение словаря? (y/n): ")
    res = input("Вывести словарь? (y/n) ")
    if res != "n":
        for key, value in dictionary.items():
            print(f"{key}: {value}")


def calculate(x, y, op):
    if op == "+":
        print(x + y)
    elif op == "-":
        print(x - y)
    elif op == "/":
        print(x - y)
    elif op == "//":
        print(x - y)
    elif op == "abs":
        print(abs(x), abs(y))
    elif op == "**":
        print(x ** y)


def sum_squares(num):
    cnt = 0
    for square in range(len(num)):
        cnt += num[square] ** 2
    print(cnt)


def generate_sequence(N):
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


if __name__ == '__main__':

    # Площадь фигур
    print_areas()
    # Различные операции с двумя переменными
    calculate(-42, -52, 'abs')
    # Сумма квадратов значений списка
    num = list(map(int, input("Введите числа через пробел, пока их сумма не будет равна нулю: ").split()))
    while sum(num) != 0:
        num.append(int(input("Сумма элементов списка не равна нулю. Введите дополнительные значения в список: ")))
    sum_squares(num)
    # Последовательность чисел, длинною N
    generate_sequence(7)