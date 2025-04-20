#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import io
import locale

# Устанавливаем кодировку для вывода в консоль (совместимо с Python 3.6)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Скачиваем стоп-слова
nltk.download('stopwords')
russian_stopwords = set(stopwords.words("russian"))

# Очистка текста
def clean_text(text):
    if pd.isna(text):  # Проверяем на NaN
        return ""
    text = str(text).lower()
    text = re.sub(r'[^а-яёa-z0-9 ]', ' ', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in russian_stopwords]
    return " ".join(tokens)

# Загрузка и подготовка данных
def load_and_prepare_data(file_path):
    try:
        # Читаем CSV файл с правильными именами столбцов
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Прочитано {len(df)} строк")
        
        # Очищаем текст
        df['Name'] = df['Name'].apply(clean_text)
        
        return df['Name'].values, df['Category'].values
    except Exception as e:
        print(f"Ошибка при загрузке данных: {str(e)}")
        raise

# Обучение модели
def train_model(X_train, X_test, y_train, y_test):
    # Модель Logistic Regression
    model = LogisticRegression(max_iter=1000)

    # Пайплайн с векторизацией и классификатором
    pipe = Pipeline([
        ("vectorizer", CountVectorizer(max_features=5000)),
        ("classifier", model)
    ])

    # Обучение модели
    pipe.fit(X_train, y_train)

    # Оценка модели
    y_pred = pipe.predict(X_test)
    score = f1_score(y_test, y_pred, average="macro")
    # Возвращаем пайплайн и оценку
    return pipe, score

# Визуализация
def plot_results(score):
    plt.figure(figsize=(8, 3))
    plt.bar(["Logistic Regression"], [score], color='green')
    plt.title("Оценка модели Logistic Regression по Macro-F1")
    plt.ylabel("F1 score")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
