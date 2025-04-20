#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import io
import locale
import traceback
import os
import pandas as pd
import numpy as np
import argparse

print("Starting application...")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

try:
    from Ai import load_and_prepare_data, train_model, plot_results
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt

    print("All required modules imported successfully")

    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Train and evaluate a text classification model')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input data file')
    parser.add_argument('--output-path', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()

    print(f"Arguments parsed: input_path={args.input_path}, output_path={args.output_path}")

    # Проверяем существование директорий
    if not os.path.exists(args.input_path):
        print(f"Error: Input directory does not exist: {args.input_path}")
        sys.exit(1)

    if not os.path.exists(args.output_path):
        print(f"Creating output directory: {args.output_path}")
        os.makedirs(args.output_path)

    # Используем train_data.csv напрямую
    file_path = 'train_data.csv'
    print(f"Using input file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: Input file does not exist: {file_path}")
        sys.exit(1)

    print(f"Input file found: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")

    # Загружаем и подготавливаем данные
    print("Loading and preparing data...")
    X, y = load_and_prepare_data(file_path)
    print(f"Data loaded successfully. X shape: {len(X)}, y shape: {len(y)}")

    # Разделяем данные на обучающую и тестовую выборки
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Обучаем модель и получаем пайплайн и оценку
    print("Training model...")
    pipe, score = train_model(X_train, X_test, y_train, y_test)
    print(f"Model training complete. Score: {score:.4f}")

    # Строим матрицу путаницы
    print("Creating confusion matrix...")
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", 
                xticklabels=[f"Категория {i}" for i in np.unique(y)], 
                yticklabels=[f"Категория {i}" for i in np.unique(y)])
    plt.title('Матрица путаницы')
    plt.xlabel('Предсказанная категория')
    plt.ylabel('Настоящая категория')

    # Сохраняем график матрицы путаницы
    confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')
    print(f"Saving confusion matrix to: {confusion_matrix_path}")
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Визуализируем результаты (F1 оценка)
    print("Creating F1 score visualization...")
    plt.figure(figsize=(8, 3))
    plt.bar(["Logistic Regression"], [score], color='green')
    plt.title("Оценка модели Logistic Regression по Macro-F1")
    plt.ylabel("F1 score")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Сохраняем график F1 оценки
    f1_score_path = os.path.join(args.output_path, 'f1_score.png')
    print(f"Saving F1 score plot to: {f1_score_path}")
    plt.savefig(f1_score_path)
    plt.close()

    # Печатаем отчет о классификации
    print("Generating classification report...")
    classification_report_text = classification_report(y_test, y_pred)
    print(classification_report_text)

    # Сохраняем отчет о классификации
    report_path = os.path.join(args.output_path, 'classification_report.txt')
    print(f"Saving classification report to: {report_path}")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(classification_report_text)

    # Сохраняем предсказания для тестовой выборки
    print("Saving predictions...")
    predictions_df = pd.DataFrame(y_pred, columns=['Category'])
    predictions_path = os.path.join(args.output_path, 'predictions.csv')
    print(f"Saving predictions to: {predictions_path}")
    predictions_df.to_csv(predictions_path, index=False, header=False, encoding='utf-8')

    # Сохраняем выходные данные в output.csv
    print("Saving output data...")
    output_df = pd.DataFrame({
        'Text': X_test,
        'Category': y_pred
    })

    # Сохраняем в output.csv
    output_path = os.path.join(args.output_path, 'output.csv')
    print(f"Saving output data to: {output_path}")
    output_df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"Results saved in directory: {args.output_path}")
    print(f"Output data saved in file: {output_path}")
    print("Application completed successfully")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("Traceback:")
    traceback.print_exc()
    sys.exit(1)
