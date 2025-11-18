import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


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


def task_3():
    '''
    Применяет алгоритмы классификации: логическую регрессию, SVM, KNN.
    Строит матрицу ошибок по результатам работы модели.
    '''
    data = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target,
                                                        train_size=0.8, shuffle=True, random_state=271)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    # Логистическая регрессия
    LR = LogisticRegression(max_iter=1000, random_state=271)
    LR.fit(X_train_scaled, y_train)
    y_pred_lr = LR.predict(X_test_scaled)
    accuracy_lr = LR.score(X_test_scaled, y_test)

    print('\n===ЛОГИЧЕСКАЯ РЕГРЕССИЯ===')
    print(f"Accuracy: {accuracy_lr:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr, target_names=data.target_names))

    # SVM
    param_kernel = ('linear', 'poly', 'rbf', 'sigmoid')
    parameters = {'kernel': param_kernel}
    model = SVC(random_state=271)
    grid_search_svm = GridSearchCV(estimator=model, param_grid=parameters, cv=6)
    grid_search_svm.fit(X_train_scaled, y_train)

    best_model_svm = grid_search_svm.best_estimator_
    svm_preds = best_model_svm.predict(X_test_scaled)

    accuracy_svm = best_model_svm.score(X_test_scaled, y_test)

    print('\n===SVM===')
    print(f"Accuracy: {accuracy_svm:.4f}")
    print(f"Best parameters: {grid_search_svm.best_params_}")
    print(f"Best CV score: {grid_search_svm.best_score_:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, svm_preds, target_names=data.target_names))

    # KNN
    number_of_neighbors = np.arange(3, 10)
    model_KNN = KNeighborsClassifier()
    params = {'n_neighbors': number_of_neighbors}

    grid_search = GridSearchCV(estimator=model_KNN, param_grid=params, cv=6)
    grid_search.fit(X_train_scaled, y_train)

    knn_preds = grid_search.predict(X_test_scaled)

    best_model_knn = grid_search.best_estimator_

    accuracy_knn = best_model_knn.score(X_test_scaled, y_test)

    print('\n===KNN===')
    print(f"Accuracy: {accuracy_knn:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, knn_preds, target_names=data.target_names))

    # Построение матриц ошибок
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=data.target_names)
    disp_lr.plot(ax=axes[0], cmap='Reds')
    axes[0].set_title('Матрица ошибок - Логистическая регрессия')

    cm_svm = confusion_matrix(y_test, svm_preds)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=data.target_names)
    disp_svm.plot(ax=axes[1], cmap='Greens')
    axes[1].set_title('Матрица ошибок - SVM')

    cm_knn = confusion_matrix(y_test, knn_preds)
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=data.target_names)
    disp_knn.plot(ax=axes[2], cmap='Blues')
    axes[2].set_title('Матрица ошибок - KNN')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    task_1()
    task_2()
    task_3()
