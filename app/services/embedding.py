import os.path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from PIL import Image, ImageDraw, ImageFont


class FaceProcessor:
    _instance = None
    print("starting FaceProcessor")
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FaceProcessor, cls).__new__(cls)
            cls._instance.app = FaceAnalysis(name='buffalo_l', root='models')
            cls._instance.app.prepare(ctx_id=0)
        return cls._instance

    def get_faces(self, image):
        faces_data = self.app.get(np.array(image))
        face_data = get_faces_data(faces_data)
        return face_data

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        return self.get_faces(image)

    def send_background(self, image, embedding, score, image_path):
        image_data = self.app.get(np.array(image))
        draw = ImageDraw.Draw(image)  # Создаем объект для рисования на изображении
        font = ImageFont.load_default(25)  # Используем стандартный шрифт

        for data in image_data:
            if compute_sim(data.embedding, embedding) > 0.8:
                x1, y1, x2, y2 = map(int, data.bbox)
                # Рисуем прямоугольник
                draw.rectangle([x1, y1, x2, y2], outline="Red", width=4)

                # Добавляем текст в верхнюю часть прямоугольника
                text = f"{score}%"
                bbox = draw.textbbox((x1, y1), text, font=font)
                text_height = bbox[3] - bbox[1]
                text_x = x1
                text_y = y1 - text_height - 5 if y1 - text_height - 5 > 0 else y1 + text_height + 5
                draw.text((text_x, text_y), text, fill="red", font=font)

                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                # Сохраняем изображение
                image.save(image_path)
                return image_path

        return False

def get_faces_data(faces):
    """Возвращает данные о лице с максимальной площадью прямоугольника."""
    if not faces:
        return None
    return max(faces, key=lambda face: calculate_rectangle_area(face["bbox"]))


def calculate_rectangle_area(bbox):
    """Вычисляет площадь прямоугольника."""
    if len(bbox) != 4:
        raise ValueError("bbox должен содержать четыре координаты: x_min, y_min, x_max, y_max")
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def compute_sim(feat1, feat2):
    try:
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim
    except Exception:
        return None

# Создаем единственный экземпляр модели
model = FaceProcessor()
