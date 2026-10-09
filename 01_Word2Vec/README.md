<a id="top"></a>

<p align="right">
  <a href="#ru">🇷🇺 RU</a> | <a href="#en">🇬🇧 EN</a>
</p>

---

<a id="ru"></a>

# Семантические пространства литературных стилей (NLP, Word2Vec)

## 📌 Описание
Pet‑проект по реализации моделей Word2Vec (CBOW и Skip‑gram) с нуля и обучению их на корпусах произведений классических авторов (Шекспир, Достоевский, Кафка, Азимов).  
Цель — исследовать, как стилистика и жанровая специфика отражаются в векторных представлениях слов и сравнить авторские стили через:
- Word2Vec (CBOW, Skip-gram)
- PCA
- t-SNE
- Косинусные расстояния
- Анализ близости слов

Практическая ценность:
- Демонстрация различий в семантических связях слов у разных авторов.  
- Применение эмбеддингов для анализа тональности, тематического моделирования и стилистических исследований.  
- Визуализация литературных стилей через методы NLP.

---

## 🔧 Стек технологий
- Python  
- NumPy, Pandas
- NLP
  - gensim — Word2Vec, словари, модельные представления
  - nltk — токенизация, стоп-слова, частеречная обработка
- Глубокое обучение: Keras (реализация CBOW и Skip‑gram)  
- Matplotlib, Seaborn  
- Визуализация:t‑SNE, PCA (визуализация эмбеддингов), WordCloud, heatmaps
- Анализ данных: косинусная схожесть, кластеризация (K-Means)
- Работа с GPU в Google Colab

---

## 📊 Данные
- Источники: Project Gutenberg, Wikipedia dumps, Kaggle Datasets.  
- Авторы: Шекспир, Достоевский, Кафка, Азимов.  
- Объём: ~240 000 слов, сбалансированные корпуса по авторам.  

---
## Ключевые этапы проекта
1. **Подготовка среды** - настройка GPU, установка зависимостей
2. **Сбор данных** - загрузка и балансировка литературных корпусов  
3. **Предобработка** - токенизация, лемматизация, очистка текстов
4. **EDA** - анализ лексики, TF-IDF, статистика корпусов
5. **Word2Vec** - реализация и обучение моделей
6. **Анализ эмбеддингов** - сравнение семантических пространств
7. **Визуализация** - t-SNE, PCA, кластеризация, heatmaps
8. **Интерпретация** - стилистический анализ результатов

---

## 📈 Результаты
- Реализованы модели Word2Vec с нуля (CBOW и Skip‑gram) для 4 авторских корпусов.
- Клиенты (корпуса) содержат четкие и устойчивые стилевые паттерны
- Обучены отдельные эмбеддинговые пространства для каждого автора.  
- Визуализированы различия в семантических связях слов («love», «robot», «truth» и др.).  
- Построены scatter‑plot и heatmap для анализа кластеров слов.
- PCA и t-SNE подтверждают высокую разделимость авторов
- Подготовлен воспроизводимый Jupyter Notebook с комментариями и графиками.

Подтверждение гипотез
- **Гипотеза 1:** ✓ Модель улавливает стилистические особенности
  - Шекспир: "love" → "honour", "duty" 
  - Достоевский: "love" → "suffering", "sacrifice"
  - Азимов: "robot" → "law", "logic", "human"

- **Гипотеза 2:** ✓ Авторы имеют уникальные семантические "отпечатки"
- **Гипотеза 3:** ✓ Жанр определяет структуру векторных пространств

Стилистические различия можно использовать:
- для классификации автора по тексту
- для стилистического анализа литературы
- для NLP-моделей, имитирующих стиль
- для построения литературных рекомендаций

---
## 📁 Структура репозитория

`01_Word2Vec/`

```text
├──  `word2vec_literature/`            — данные для проекта
├──  `word2vec_literature_gpu/`        — данные для проекта
├──  `Colab_Pet-proj_Word2Vec.ipynb`   — основной ноутбук с полным анализом, выполненный в Google Colab  
├──  `Colab_Pet-proj_Word2Vec.pdf`     — экспорт в PDF  
├──  `Pet-proj_Word2Vec_Keras.ipynb`   — основной ноутбук с полным анализом, выполненный локально 
├──  `Pet-proj_Word2Vec_Keras.pdf`     — экспорт в PDF  
├──  `Pet-proj_Word2Vec_PyTorch.ipynb` — основной ноутбук с полным анализом, выполненный локально 
├──  `Pet-proj_Word2Vec_PyTorch.pdf`   — экспорт в PDF  
├──  `requirements.txt`                — зависимости проекта
└──  `README.md`                       — эта документация
```
---

## 🚀 Как запустить
1. Склонировать репозиторий:  
   ```bash
   git clone https://github.com/aspozd-DA-DS/Pet-projects.git
   ```
2. Перейти в папку проекта:
   ```bash
   cd Pet-projects/01_Word2Vec
   ```
3. Запустить ноутбук:
   ```bash
   jupyter notebook Pet-proj_Word2Vec_Keras.ipynb
   ```

   ```bash
   jupyter notebook Pet-proj_Word2Vec_PyTorch.ipynb
   ```

## 🏷 Topics
`NLP` ` Word2Vec` `Embeddings` `Text Analysis` `Python` `Keras` `Visualization` `gensim` `literature` `pca` `tsne` `machine-learning` `semantic-analysis`

<p align="right"><a href="#top">⬆ наверх</a></p>

---

<a id="en"></a>

# Semantic spaces of literary styles (NLP, Word2Vec)

## 📌 Description
A pet project implementing Word2Vec models (CBOW and Skip-gram) from scratch and training them on corpora of works by classic authors (Shakespeare, Dostoevsky, Kafka, Asimov).  
The goal is to explore how style and genre specifics are reflected in the vector representations of words, and to compare authorial styles using:
- Word2Vec (CBOW, Skip-gram)
- PCA
- t-SNE
- Cosine distances
- Word proximity analysis

Practical value:
- Demonstrating differences in the semantic relations between words across authors.  
- Applying embeddings to sentiment analysis, topic modeling, and stylistic research.  
- Visualizing literary styles with NLP methods.

---

## 🔧 Tech stack
- Python  
- NumPy, Pandas
- NLP
  - gensim — Word2Vec, dictionaries, model representations
  - nltk — tokenization, stop words, part-of-speech tagging
- Deep learning: Keras (implementation of CBOW and Skip-gram)  
- Matplotlib, Seaborn  
- Visualization: t-SNE, PCA (embedding visualization), WordCloud, heatmaps
- Data analysis: cosine similarity, clustering (K-Means)
- GPU usage in Google Colab

---

## 📊 Data
- Sources: Project Gutenberg, Wikipedia dumps, Kaggle Datasets.  
- Authors: Shakespeare, Dostoevsky, Kafka, Asimov.  
- Volume: ~240,000 words, corpora balanced by author.  

---
## Key project stages
1. **Environment setup** - GPU configuration, installing dependencies
2. **Data collection** - loading and balancing the literary corpora  
3. **Preprocessing** - tokenization, lemmatization, text cleaning
4. **EDA** - vocabulary analysis, TF-IDF, corpus statistics
5. **Word2Vec** - implementing and training the models
6. **Embedding analysis** - comparing semantic spaces
7. **Visualization** - t-SNE, PCA, clustering, heatmaps
8. **Interpretation** - stylistic analysis of the results

---

## 📈 Results
- Word2Vec models (CBOW and Skip-gram) were implemented from scratch for 4 authorial corpora.
- The corpora contain clear and stable stylistic patterns.
- Separate embedding spaces were trained for each author.  
- Differences in semantic associations were visualized for words such as "love", "robot", "truth", and others.  
- Scatter plots and heatmaps were built to analyze word clusters.
- PCA and t-SNE confirm high separability of the authors
- A reproducible Jupyter Notebook with comments and charts was prepared.

Hypothesis testing
- **Hypothesis 1:** ✓ The model captures stylistic features
  - Shakespeare: "love" → "honour", "duty" 
  - Dostoevsky: "love" → "suffering", "sacrifice"
  - Asimov: "robot" → "law", "logic", "human"

- **Hypothesis 2:** ✓ Authors have unique semantic "fingerprints"
- **Hypothesis 3:** ✓ Genre shapes the structure of vector spaces

The stylistic differences can be used:
- for classifying an author from a text
- for stylistic analysis of literature
- for NLP models that imitate a style
- for building literary recommendations

---
## 📁 Repository structure

`01_Word2Vec/`

```text
├──  `word2vec_literature/`            — project data
├──  `word2vec_literature_gpu/`        — project data
├──  `Colab_Pet-proj_Word2Vec.ipynb`   — main notebook with full analysis, run in Google Colab  
├──  `Colab_Pet-proj_Word2Vec.pdf`     — PDF export  
├──  `Pet-proj_Word2Vec_Keras.ipynb`   — main notebook with full analysis, run locally 
├──  `Pet-proj_Word2Vec_Keras.pdf`     — PDF export  
├──  `Pet-proj_Word2Vec_PyTorch.ipynb` — main notebook with full analysis, run locally 
├──  `Pet-proj_Word2Vec_PyTorch.pdf`   — PDF export  
├──  `requirements.txt`                — project dependencies
└──  `README.md`                       — this documentation
```
---

## 🚀 How to run
1. Clone the repository:  
   ```bash
   git clone https://github.com/aspozd-DA-DS/Pet-projects.git
   ```
2. Go to the project folder:
   ```bash
   cd Pet-projects/01_Word2Vec
   ```
3. Run the notebook:
   ```bash
   jupyter notebook Pet-proj_Word2Vec_Keras.ipynb
   ```

   ```bash
   jupyter notebook Pet-proj_Word2Vec_PyTorch.ipynb
   ```

## 🏷 Topics
`NLP` ` Word2Vec` `Embeddings` `Text Analysis` `Python` `Keras` `Visualization` `gensim` `literature` `pca` `tsne` `machine-learning` `semantic-analysis`

<p align="right"><a href="#top">⬆ back to top</a></p>
