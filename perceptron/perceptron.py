import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import glob

# 1. Загрузка и объединение всех данных
data_files = glob.glob('data/fishing_data_v*.csv')
dfs = []

for file in data_files:
    df = pd.read_csv(file)
    # Проверяем наличие всех нужных колонок
    if all(col in df.columns for col in ['fish_pos', 'zone_pos', 'fish_speed', 'action']):
        dfs.append(df)
    else:
        print(f"Файл {file} пропущен - отсутствуют нужные колонки")

if not dfs:
    raise ValueError("Не найдено ни одного подходящего файла данных")

data = pd.concat(dfs, ignore_index=True)

# 2. Создание дополнительных признаков
data["pos_diff"] = data["fish_pos"] - data["zone_pos"]  # Разница позиций
data["distance"] = np.abs(data["pos_diff"])  # Абсолютное расстояние
data["speed_dir"] = np.sign(data["fish_speed"])  # Направление движения (1 = вниз, -1 = вверх)

# 3. Разделение на признаки и целевую переменную
X = data[["fish_pos", "zone_pos", "fish_speed", "pos_diff", "distance", "speed_dir"]]
y = data["action"]

# 4. Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Разделение на тренировочные и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Создание и обучение перцептрона с улучшениями
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)

# Добавляем ручные фичи


print("\nНачинаем обучение...")
model.fit(X_train, y_train)

# 7. Оценка модели
print("\nОценка модели на тестовых данных:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Сохранение модели и нормализатора
model_name = "fishing_model_enhanced.pkl"
scaler_name = "scaler_enhanced.pkl"

joblib.dump(model, model_name)
joblib.dump(scaler, scaler_name)
print(f"\nМодель сохранена в {model_name}")
print(f"Нормализатор сохранен в {scaler_name}")

# 9. Дополнительная информация
print("\nСтатистика данных:")
print(f"Всего записей: {len(data)}")
print("Распределение классов:")
print(y.value_counts(normalize=True))
