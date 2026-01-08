# temporary-repository

import pandas as pd
import numpy as np

#### Загрузка
df = pd.read_csv('data.csv')


df = pd.read_csv(
    r"C:\Users\computer\Desktop\Prog5sem\portal_data.csv",
    sep='|',
    encoding='cp1251'
)

df.info()



#### Основная информация
print(df.shape)                     ### (n_rows, n_cols)
print(df.columns.tolist())          ### список признаков
print(df.dtypes)                    ### типы данных
print(df.info())                    ### память, ненулевые значения
print(df.describe())                ### статистика по числовым признакам
print(df.head())                    ### первые 5 строк
print(df.nunique())                 ### уникальные значения (полезно для категорий)


# Анализ пропусков

#### Количество пропусков по столбцам
missing = df.isnull().sum()
print(missing[missing > 0])

#### Доля пропусков (%)
missing_pct = df.isnull().mean() * 100
print(missing_pct[missing_pct > 0])


# Стратегии обработки:
Если пропусков менее 5% и они случайные, то удалить строки
Если пропусков много, но можно восстановить, то заполнить средним/медианой/модой
Если пропуски не случайны (например, «не указан доход» = 0), то создать бинарный индикатор + заполнить

### Числовые признаки → медиана (устойчива к выбросам)
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

### Категориальные → мода (самое частое значение)
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)



# Кодировка

Возраст, цена, температура Числовой (continuous) Не требует кодирования
Рейтинг (1–5), этап обучения Порядковый (ordinal) LabelEncoder или ручное маппинговое словарём
Пол, страна, цвет Номинальный (nominal) pd.get_dummies() или OneHotEncoder
ID, почта, комментарий Идентификатор / текст Обычно удаляют (если нет NLP)


#### One-Hot (для номинальных)
df = pd.get_dummies(df, columns=['country', 'color'], drop_first=True)

#### Label Encoding (для порядковых)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['education_level'] = le.fit_transform(df['education_level'])  #### предполагается порядок


# Визуализация выбросов


import matplotlib.pyplot as plt
import seaborn as sns

### Boxplot для числовых признаков
num_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 6))
df.boxplot(column=num_cols[:6])  # первые 6 числовых колонок
plt.xticks(rotation=45)
plt.title("Boxplots for outlier detection")
plt.show()

### Гистограмма + KDE
for col in num_cols[:3]:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


  ### Удалить один столбец
df = df.drop(columns=['unnecessary_col'])

### Удалить несколько столбцов
df = df.drop(columns=['col1', 'col2', 'col3'])

### Удалить столбцы по условию (например, все с >95% пропусков)
cols_to_drop = df.columns[df.isnull().mean() > 0.95]
df = df.drop(columns=cols_to_drop)

### Удалить столбцы с константными значениями
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=constant_cols)

### Удалить столбцы, похожие на ID (уникальных значений почти столько же, сколько строк)
id_like_cols = [col for col in df.columns if df[col].nunique() / len(df) > 0.99]
df = df.drop(columns=id_like_cols)


### Удалить строки с хотя бы одним пропуском
df = df.dropna()

### Удалить строки, где пропущены конкретные столбцы
df = df.dropna(subset=['important_col'])

### Удалить дубликаты (все столбцы)
df = df.drop_duplicates()

### Удалить дубликаты по определённым столбцам
df = df.drop_duplicates(subset=['name', 'email'])

### Удалить строки по условию (например, возраст < 0)
df = df[df['age'] >= 0]

### Удалить строки с выбросами (пример по IQR для одного столбца)
Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['income'] >= lower) & (df['income'] <= upper)]



### Заполнить все числовые столбцы медианой
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

### Заполнить категориальные столбцы модой
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col].fillna(mode_val[0], inplace=True)

### Создать бинарный индикатор пропуска (если важно знать, где был NaN)
df['income_was_missing'] = df['income'].isnull().astype(int)
df['income'].fillna(df['income'].median(), inplace=True)




#### Привести к числовому типу (с заменой некорректных значений на NaN)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

#### Привести к категории (экономит память для повторяющихся строк)
df['category'] = df['category'].astype('category')

#### Дата/время
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month



#### Посмотреть уникальные значения и их частоту
print(df['country'].value_counts(dropna=False))

### Сгруппировать редкие категории в 'Other'
threshold = 10  # меньше 10 наблюдений → Other
value_counts = df['category'].value_counts()
to_replace = value_counts[value_counts < threshold].index
df['category'] = df['category'].replace(to_replace, 'Other')

### One-Hot Encoding
df = pd.get_dummies(df, columns=['color', 'size'], drop_first=True)



### Найти строки с некорректными значениями (например, отрицательная цена)
invalid_prices = df[df['price'] < 0]
print(invalid_prices)

### Оставить только допустимые диапазоны
df = df[(df['age'] >= 0) & (df['age'] <= 120)]

### Убедиться, что целевая переменная имеет ожидаемые значения
assert set(df['target'].unique()) <= {0, 1}, "Target должен быть бинарным!"


from sklearn.model_selection import train_test_split

### Для задачи с таргетом y
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          ### ← сохраняет пропорции классов (для классификации!)
)


Использовать stratify=y для классификации и НЕ использовать для регрессии


Требуют масштабирования: Линейная регрессия, LogisticRegression, SVM, KNN, Neural Networks

Не требуют масштабириования: Деревья, Random Forest, XGBoost, LightGBM


from sklearn.preprocessing import StandardScaler

### 1. Создаём скалер
scaler = StandardScaler()

### 2. Обучаем ТОЛЬКО на train
scaler.fit(X_train)

### 3. Преобразуем ВСЕ наборы
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Если есть val:
### X_val_scaled = scaler.transform(X_val)




Как выбрать модель — простой алгоритм рассуждений
Когда тебе дают задачу на контрольной, первое, что нужно определить — это тип задачи. Если целевая переменная (таргет) — это категории (например, «спам / не спам», «болен / здоров»), то это задача классификации. Если таргет — число (цена, температура, время), то это регрессия. Если таргета вообще нет, а нужно найти структуру в данных — это кластеризация или другая задача без учителя.

После этого подумай: насколько важна интерпретируемость? Если нужно объяснить, почему модель приняла решение (например, в медицине или банковском скоринге), начинай с простых моделей — логистической регрессии или дерева решений. Если же важна только точность, а данные сложные и большие, смело бери ансамбли вроде случайного леса.

Обрати внимание и на размер данных. Если у тебя всего несколько сотен строк, сложные модели (нейросети, градиентный бустинг) могут переобучиться. В таких случаях лучше ограничиться линейными моделями или небольшими деревьями. Если данных много (десятки тысяч и больше), мощные модели покажут себя лучше.

Также учти, работаешь ли ты с числовыми признаками, категориями или их смесью. Некоторые модели (линейные) требуют One-Hot кодирования категорий, другие (деревья) спокойно работают с Label Encoding или даже с сырыми строками (в новых версиях sklearn). Если ты уже закодировал категории через get_dummies и получил разреженную матрицу, линейные модели справятся отлично.

Если в данных есть выбросы или пропуски, которые ты не смог полностью обработать, выбирай устойчивые модели — деревья и их ансамбли почти не страдают от выбросов, в отличие от линейных методов или SVM.

Наконец, если у тебя мало времени на настройку, возьми RandomForest — он почти не требует предобработки, редко сильно ошибается и работает «из коробки». Это универсальный выбор для контрольной, особенно если ты не уверен.



from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  #### для бинарной классификации




from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]



from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Структура обучения

### 1. Выбери модель
model = RandomForestClassifier(random_state=42)

### 2. Обучи на train
model.fit(X_train, y_train)

### 3. Предскажи на test
y_pred = model.predict(X_test)

### 4. (Опционально) Получи вероятности
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)



    Как оценивать модель — логика рассуждений
Когда модель обучена и сделаны предсказания, первое, что нужно сделать — выбрать правильную метрику, исходя из типа задачи и её контекста. В задачах классификации точность (accuracy) кажется очевидной, но она обманчива, если классы несбалансированы: например, при 95% здоровых и 5% больных пациентов модель, всегда предсказывающая «здоров», будет иметь accuracy = 0.95, но полностью бесполезна. В таких случаях смотри на precision, recall и F1-меру. Precision отвечает на вопрос: «Из всех, кого я пометил как больных, сколько действительно больны?» Recall — «Из всех реально больных, сколько я нашёл?». Если важно не пропустить ни одного случая (например, диагностика рака), максимизируй recall. Если ложные срабатывания дорого обходятся (например, блокировка легальных транзакций), важнее precision.

Если задача бинарной классификации и у тебя есть вероятности, построй ROC-кривую и посчитай AUC — эта метрика показывает, насколько хорошо модель разделяет классы независимо от порога. AUC = 0.5 — случайная модель, AUC = 1.0 — идеальная. Это особенно полезно, когда порог бинаризации ещё не выбран.

В задачах регрессии среднеквадратичная ошибка (MSE) чувствительна к большим ошибкам, потому что возводит их в квадрат. Если такие выбросы в прогнозах недопустимы (например, в финансах), MSE — хороший выбор. Если же важна средняя абсолютная ошибка в тех же единицах, что и таргет, используй MAE. Коэффициент детерминации R² показывает, насколько твои предсказания лучше, чем простое угадывание среднего значения: R² = 1 — идеально, R² = 0 — как у среднего, R² < 0 — хуже среднего.

После численной оценки обязательно визуализируй результаты. Confusion matrix даёт полную картину ошибок в классификации: где модель путает классы, какие ошибки чаще. Для регрессии — график истинных vs предсказанных значений: если точки лежат вдоль диагонали, всё хорошо; если есть систематический сдвиг или конус — модель недообучена или данные сложные.

Наконец, интерпретируемость — ключ к доверию. У линейных моделей смотри на коэффициенты: положительный — признак увеличивает прогноз, отрицательный — уменьшает. У деревьев и случайного леса — на feature importance: какие признаки вносят наибольший вклад в решение. Это не только помогает объяснить модель, но и выявить возможные утечки данных (например, если важен признак «ID клиента» — что-то пошло не так).




# Метрики

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

### Основные метрики (бинарная классификация)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

### Если есть вероятности — AUC
if 'y_proba' in locals():
    auc = roc_auc_score(y_test, y_proba)

### Подробный отчёт по классам
print(classification_report(y_test, y_pred))



import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.show()




from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # случайная модель
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()





from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")



plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Истинные vs Предсказанные')
plt.show()




### Для деревьев (RandomForest, XGBoost и др.)
importances = model.feature_importances_
features = X.columns  # или список названий признаков

### Для линейных моделей
### importances = np.abs(model.coef_)  # или model.coef_ напрямую

### Визуализация
plt.figure(figsize=(8, 6))
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
plt.title('Важность признаков')
plt.tight_layout()
plt.show()





Подбор всегда делается только на обучающих данных, но с использованием валидации — иначе ты просто переобучишься на тестовую выборку. Лучший способ — использовать кросс-валидацию внутри обучения, что реализовано в GridSearchCV и RandomizedSearchCV из sklearn. Эти инструменты автоматически делят train на фолды, перебирают параметры и выбирают комбинацию с лучшим средним результатом по фолдам.

Важно: если в твоём пайплайне есть предобработка (например, масштабирование), её тоже нужно включать в поиск — иначе при ручном масштабировании до GridSearchCV может произойти утечка данных. Для этого используется Pipeline.

Если времени мало или параметров много, лучше использовать RandomizedSearchCV — он пробует случайные комбинации из заданного распределения и часто находит хорошее решение быстрее, чем полный перебор.


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#### Определяем сетку гиперпараметров
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

#### Создаём модель
model = RandomForestClassifier(random_state=42)

#### Запускаем поиск
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                     # 5-фолдная кросс-валидация
    scoring='f1',             # метрика для оптимизации
    n_jobs=-1,                # использовать все ядра
    verbose=1                 # показывать прогресс
)

#### Обучаем (на X_train, y_train!)
grid_search.fit(X_train, y_train)

#### Лучшая модель и параметры
print("Лучшие параметры:", grid_search.best_params_)
print("Лучший скор (F1):", grid_search.best_score_)

#### Используем лучшую модель для предсказания
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)




from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 300),      # случайное целое от 50 до 300
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,                # попробовать 20 случайных комбинаций
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print("Лучшие параметры:", random_search.best_params_)
