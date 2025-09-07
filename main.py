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
            print("Ошибка: необходимо ввести ровно три числа.")
        exit = input("Продолжить заполнение словаря? (y/n): ")
    res = input("Вывести словарь? (y/n) ")
    if res != "n":
        for key, value in dictionary.items():
            print(f"{key}: {value}")


if __name__ == '__main__':
    print_areas()