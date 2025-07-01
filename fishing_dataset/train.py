from ultralytics import YOLO

# 1. Загрузка модели (без resume)
model = YOLO("../models/fish_detector_v4.pt")

# 2. Обучение с увеличенным количеством эпох
# model.train(
#     data="data.yaml",
#     epochs=450,  # 400 уже пройдено + 50 новых
#     imgsz=640,
#     batch=8,
#     lr0=0.001,  # Уменьшенный learning rate для дообучения
#     device='cpu',
#     name='fish_detector_v4',
#     pretrained=True  # Критически важно!
# )


results = model.val()  # Тестирование на валидационном наборе
print(f"mAP50-95: {results.box.map}")  # Должен быть выше, чем до дообучения
