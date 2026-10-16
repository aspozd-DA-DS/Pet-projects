import copy
import hashlib
import io
import json
import math
import os
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_drawable_canvas import st_canvas

# Базовая настройка страницы Streamlit
st.set_page_config(page_title="Capsule Annotator", layout="wide")

# Пути и константы приложения
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(BASE_DIR, "data_packet")
IMG_EXT = (".jpg", ".jpeg", ".png")
CANVAS_MAX = 640      # максимальная сторона холста в пикселях (помещается в правую половину экрана)
MIN_SIZE = 6          # игнорировать боксы меньше этого размера (px, исходный масштаб)
MAX_ZOOM = 4.0

# Справочники классов, форм, цветов и стилей
CLASSES = {0: "capsule", 1: "softgel_gelcap", 2: "meltlet", 3: "caplet"}
SHAPES = [
    "round", "oval", "elongated", "oblong", "capsule_shape",
    "triangular", "square", "pentagonal", "hexagonal", "diamond", "irregular", "other",
]
COLORS = [
    "white", "black", "gray", "brown", "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "ivory", "cream", "beige", "tan", "taupe", "charcoal",
    "mint", "olive", "lime", "teal", "navy", "cyan", "turquoise",
    "maroon", "burgundy", "coral", "magenta", "gold", "amber",
    "lavender", "lilac", "indigo", "bronze", "copper", "silver",
    "peach", "salmon", "rose", "khaki", "sky blue", "forest green", "plum", "violet",
    "aqua", "rust",
    "translucent", "multicolor", "other",
]
LINE_STYLES = ["solid", "dashed", "dotted"]
TRANSPARENCY = ["opaque", "transparent"]
TOOL_MODES = ["rect", "polygon", "point", "transform"]
TOOL_LABELS = ["t_rect", "t_poly", "t_points", "t_edit"]

# Словарь локализации интерфейса (английский / русский)
TEXT = {
    "en": {
        "title": "Capsule Annotator",
        "footer_data_note": "Data is processed in session memory and is not written to disk.",
        "footer_copyright": "© A.S.Pozdniakov 2026",
        "subtitle": "Zoom → outline objects (boxes or polygons) → classify → export",
        "sec_source": "Data source",
        "btn_demo": "Load demo folder (data_packet)",
        "demo_select": "Demo file (data_packet)",
        "no_demo_opt": "— none —",
        "upload_label": "Or upload your own images",
        "err_folder": "Folder not found",
        "no_demo": "Demo folder data_packet not found next to the app",
        "loaded": "Loaded {n} images",
        "no_images": "No .jpg / .png images in this folder",
        "choose": "Choose a demo file or upload images in the sidebar to start",
        "sec_mode": "Packet mode",
        "mode_single": "One packet per image",
        "mode_multi": "Several packets per image",
        "mode_hint": "One packet: all shapes you draw belong to it",
        "sec_style": "Annotation style",
        "sec_zoom": "Zoom frame",
        "font_color": "Font color",
        "packet_count": "Number of packets on the image",
        "packet_progress": "Packet {k} of {n}",
        "confirm_packet": "Confirm packet",
        "packets_done": "Packets confirmed: {k} of {n}",
        "all_packets_done": "All packets are annotated. Save the annotation.",
        "box_color": "Box color",
        "box_width": "Box thickness",
        "box_style": "Box line type",
        "poly_color": "Polygon color",
        "poly_width": "Polygon thickness",
        "poly_style": "Polygon line type",
        "font_size": "Label font size",
        "point_color": "Point color",
        "point_size": "Point size",
        "zoom_color": "Zoom frame color",
        "zoom_width": "Zoom frame thickness",
        "zoom_style": "Zoom frame line type",
        "pv_hint": "Settings come from the sidebar and apply to this preview and the export",
        "step3": "Step 3. Review the objects and save the annotation",
        "step4": "Step 4. Preview and display settings",
        "box_title": "Box", "poly_title": "Polygon",
        "color_lbl": "Color", "thickness_lbl": "Thickness", "linetype_lbl": "Line type",
        "line_solid": "Solid",
        "line_dashed": "Dashed",
        "line_dotted": "Dotted",
        "orig": "Original",
        "annotated": "Annotated",
        "step1": "Step 1. Draw a rectangle around the packet and apply zoom",
        "zoom_btn": "Apply zoom",
        "reset_zoom": "Reset zoom",
        "step2": "Step 2. Outline the objects",
        "tool": "Tool",
        "t_rect": "Box",
        "t_poly": "Polygon (click vertices, right-click to close)",
        "t_points": "Points (each object separately, then Finish object)",
        "t_edit": "Edit / move",
        "capture": "Capture objects",
        "finish_obj": "Finish object",
        "undo_pt": "Undo last point",
        "pts_hint": "Click points, then press Finish object. Repeat to mark several objects. Capture objects when done.",
        "nothing": "No shapes found on the canvas",
        "captured": "Captured {n} object(s)",
        "edit_objs": "Objects",
        "obj": "#{i}",
        "shape": "Shape",
        "color": "Color",
        "cls": "Class",
        "save": "Save annotation",
        "saved": "Annotation saved. See the preview below.",
        "no_objs_save": "Nothing to save — capture objects first",
        "preview": "Preview",
        "ctrl": "Display settings (applied to the preview and the export)",
        "results": "All annotations",
        "export": "Prepare export",
        "download": "Download ZIP",
        "export_hint": "ZIP contains: YOLO .txt, per-image .json, annotations_all.json and annotated .jpg for each image",
        "prev": "← Previous",
        "next": "Next →",
        "counter": "Image {i}/{n}: {name}",
        "orig_size": "Original size: {w} × {h}",
        "transp_lbl": "Transparency",
        "opaque": "Opaque",
        "transparent": "Transparent",
    },
    "ru": {
        "title": "Разметка капсул",
        "footer_data_note": "Данные обрабатываются в памяти сессии и не записываются на диск приложения.",
        "footer_copyright": "А.С.Поздняков 2026",
        "subtitle": "Зум → обводка объектов (боксы или полигоны) → классификация → выгрузка",
        "sec_source": "Источник данных",
        "btn_demo": "Загрузить демо-папку (data_packet)",
        "demo_select": "Демо-файл (data_packet)",
        "no_demo_opt": "— не выбрано —",
        "upload_label": "Или загрузите свои изображения",
        "err_folder": "Папка не найдена",
        "no_demo": "Демо-папка data_packet не найдена рядом с приложением",
        "loaded": "Загружено изображений: {n}",
        "no_images": "В папке нет изображений .jpg / .png",
        "choose": "Выберите демо-файл или загрузите изображения в боковой панели",
        "sec_mode": "Режим пакетиков",
        "mode_single": "Один пакетик на изображении",
        "mode_multi": "Несколько пакетиков на изображении",
        "mode_hint": "Один пакетик: все нарисованные фигуры относятся к нему",
        "sec_style": "Стиль разметки",
        "sec_zoom": "Рамка зума",
        "font_color": "Цвет шрифта",
        "packet_count": "Количество пакетиков на изображении",
        "packet_progress": "Пакетик {k} из {n}",
        "confirm_packet": "Зафиксировать пакетик",
        "packets_done": "Зафиксировано пакетиков: {k} из {n}",
        "all_packets_done": "Все пакетики размечены. Сохраните разметку.",
        "box_color": "Цвет бокса",
        "box_width": "Толщина бокса",
        "box_style": "Тип линии бокса",
        "poly_color": "Цвет полигона",
        "poly_width": "Толщина полигона",
        "poly_style": "Тип линии полигона",
        "font_size": "Размер шрифта подписи",
        "point_color": "Цвет точек",
        "point_size": "Размер точек",
        "zoom_color": "Цвет рамки зума",
        "zoom_width": "Толщина рамки зума",
        "zoom_style": "Тип линии рамки зума",
        "pv_hint": "Настройки берутся из боковой панели и применяются к предпросмотру и выгрузке",
        "step3": "Шаг 3. Проверьте объекты и сохраните разметку",
        "step4": "Шаг 4. Предпросмотр и настройки отображения",
        "box_title": "Бокс", "poly_title": "Полигон",
        "color_lbl": "Цвет", "thickness_lbl": "Толщина", "linetype_lbl": "Тип линии",
        "line_solid": "Сплошная",
        "line_dashed": "Пунктир",
        "line_dotted": "Точечная",
        "orig": "Исходное",
        "annotated": "С аннотацией",
        "step1": "Шаг 1. Выделите прямоугольником пакетик и примените зум",
        "zoom_btn": "Применить зум",
        "reset_zoom": "Сбросить зум",
        "step2": "Шаг 2. Обведите объекты",
        "tool": "Инструмент",
        "t_rect": "Бокс",
        "t_poly": "Полигон (ставьте вершины, правый клик — замкнуть)",
        "t_points": "Точки (каждый объект отдельно, затем «Завершить объект»)",
        "t_edit": "Редактирование / перемещение",
        "capture": "Зафиксировать объекты",
        "finish_obj": "Завершить объект",
        "undo_pt": "Удалить последнюю точку",
        "pts_hint": "Ставьте точки и нажмите «Завершить объект». Повторите для каждого объекта, затем «Зафиксировать объекты».",
        "nothing": "На холсте не найдено фигур",
        "captured": "Зафиксировано объектов: {n}",
        "edit_objs": "Объекты",
        "obj": "№{i}",
        "shape": "Форма",
        "color": "Цвет",
        "cls": "Класс",
        "save": "Сохранить разметку",
        "saved": "Разметка сохранена. Ниже — предпросмотр.",
        "no_objs_save": "Нечего сохранять — сначала зафиксируйте объекты",
        "preview": "Предпросмотр",
        "ctrl": "Настройки отображения (применяются к предпросмотру и выгрузке)",
        "results": "Все аннотации",
        "export": "Подготовить выгрузку",
        "download": "Скачать ZIP",
        "export_hint": "В ZIP: YOLO .txt, json на каждое изображение, annotations_all.json и аннотированный .jpg",
        "prev": "← Назад",
        "next": "Вперёд →",
        "counter": "Изображение {i}/{n}: {name}",
        "orig_size": "Исходный размер: {w} × {h}",
        "transp_lbl": "Прозрачность",
        "opaque": "Непрозрачная",
        "transparent": "Прозрачная",
    },
}


def t(key, **kw):
    """Возвращает локализованную строку по ключу с подстановкой параметров."""
    text = TEXT[L][key]
    return text.format(**kw) if kw else text


# ---------------------------------------------------------------------------
# Кэшированные вспомогательные функции (ускоряют приложение, избегая
# повторного чтения файлов при перерисовках)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_images(folder):
    """Возвращает отсортированный список изображений в папке."""
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


@st.cache_data(show_spinner=False)
def load_image(src):
    """src: путь к демо-файлу на диске или сырые байты загруженного файла."""
    buf = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    img = ImageOps.exif_transpose(Image.open(buf)).convert("RGB")
    return np.array(img)


@st.cache_resource(show_spinner=False)
def get_font(size):
    """Подбирает доступный TTF-шрифт заданного размера или системный по умолчанию."""
    for p in [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Детекция: форма и цвет (улучшенная версия исходных эвристик)
# ---------------------------------------------------------------------------
def foreground_mask(crop, region):
    """Отделяет объект от фона по цветовому расстоянию до медианы границы (Lab + Otsu)."""
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.bitwise_and(th, region)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(th)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return out


def classify_shape(fg):
    """Определяет форму объекта по маске переднего плана (круг, овал, капсула и т.д.)."""
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return "unknown"
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 20:
        return "unknown"
    peri = cv2.arcLength(cnt, True)
    circ = 4 * np.pi * area / (peri * peri + 1e-6)
    _, (rw, rh), _ = cv2.minAreaRect(cnt)
    aspect = max(rw, rh) / (min(rw, rh) + 1e-6)
    sol = area / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-6)
    verts = len(cv2.approxPolyDP(cnt, 0.03 * peri, True))

    if sol < 0.80:
        return "irregular"
    if verts == 3:
        return "triangular"
    if aspect < 1.3 and circ < 0.85:
        if verts == 5:
            return "pentagonal"
        if verts == 6:
            return "hexagonal"
    if aspect < 1.15:
        if circ > 0.85:
            return "round"
        if verts == 4:
            return "square"
        return "round" if circ > 0.75 else "other"
    if aspect < 1.6:
        return "oval"
    if aspect < 2.2:
        return "oblong"
    return "capsule_shape" if sol > 0.92 else "elongated"


def classify_color(px):
    """Определяет цвет объекта по пикселям переднего плана (HSV + эвристики)."""
    if px.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv = hsv.reshape(-1, 3).astype(np.float32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    s_m, v_m = float(np.median(s)), float(np.median(v))

    if v_m < 50:
        return "black"
    if s_m < 40:
        if v_m > 215:
            return "white"
        if v_m > 150:
            return "gray"
        return "charcoal"

    ang = np.deg2rad(h * 2.0)
    c, sn = np.cos(ang).mean(), np.sin(ang).mean()
    if math.hypot(c, sn) < 0.5 and s_m > 60:
        return "multicolor"
    hm = (math.degrees(math.atan2(sn, c)) % 360) / 2.0  # шкала оттенка OpenCV 0..180

    if s_m < 90 and v_m > 200:  # пастельные тона
        if 8 <= hm < 30:
            return "cream" if s_m < 50 else "beige"
        if hm >= 150 or hm < 8:
            return "pink"
        return "ivory"
    if 8 <= hm < 22 and v_m < 150:
        return "brown"
    if hm < 8 or hm >= 170:
        return "maroon" if v_m < 120 else "red"
    if hm < 16:
        return "orange"
    if hm < 22:
        return "amber"
    if hm < 35:
        return "yellow"
    if hm < 45:
        return "lime"
    if hm < 80:
        return "green"
    if hm < 95:
        return "teal"
    if hm < 105:
        return "cyan"
    if hm < 125:
        return "blue"
    if hm < 140:
        return "indigo"
    if hm < 150:
        return "purple"
    return "magenta" if v_m > 180 else "pink"


def analyze(img0, pts, kind, bbox):
    """Анализирует вырезанную область объекта и возвращает (форма, цвет)."""
    x1, y1, x2, y2 = bbox
    crop = img0[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown", "unknown"
    h, w = crop.shape[:2]
    region = np.zeros((h, w), np.uint8)
    if kind == "box":
        region[:] = 255
    else:
        poly = np.array([[px - x1, py - y1] for px, py in pts], np.int32)
        cv2.fillPoly(region, [poly], 255)
    fg = foreground_mask(crop, region)
    if fg.sum() < 20 * 255:
        return "unknown", "unknown"
    return classify_shape(fg), classify_color(crop[fg > 0])


# ---------------------------------------------------------------------------
# Геометрия: преобразование координат холста -> исходного изображения
# ---------------------------------------------------------------------------
def canvas_to_shapes(json_data, scale, ox, oy, px=0, py=0, dot_kind="points"):
    """Преобразует объекты с холста Streamlit в список фигур с координатами исходного изображения."""
    def to_orig(x, y):
        return [int(round((x - px) / scale)) + ox, int(round((y - py) / scale)) + oy]

    shapes, dots = [], []
    for obj in (json_data or {}).get("objects", []):
        kind = obj.get("type")
        sx, sy = obj.get("scaleX", 1.0), obj.get("scaleY", 1.0)
        if kind == "rect":
            x1, y1 = obj["left"], obj["top"]
            x2, y2 = x1 + obj["width"] * sx, y1 + obj["height"] * sy
            pts = [to_orig(x1, y1), to_orig(x2, y1), to_orig(x2, y2), to_orig(x1, y2)]
            shapes.append({"kind": "box", "points": pts})
        elif kind == "path":
            cmds = obj.get("path") or []
            if any(c and str(c[0]).lower() == "z" for c in cmds):  # замкнутый полигон (правый клик)
                raw = [(c[1], c[2]) for c in cmds if c and c[0] in ("M", "L") and len(c) >= 3]
                if "pathOffset" in obj:
                    po = obj["pathOffset"]
                else:  # холст не передаёт pathOffset: это центр точек пути
                    xs, ys = [p_[0] for p_ in raw], [p_[1] for p_ in raw]
                    po = {"x": (min(xs) + max(xs)) / 2, "y": (min(ys) + max(ys)) / 2}
                pts = [
                    to_orig(obj["left"] + (x - po["x"]) * sx, obj["top"] + (y - po["y"]) * sy)
                    for x, y in raw
                ]
                if len(pts) >= 3:
                    shapes.append({"kind": "polygon", "points": pts})
        elif kind == "polygon":
            po = obj.get("pathOffset", {"x": 0, "y": 0})
            pts = [
                to_orig(obj["left"] + (p["x"] - po["x"]) * sx,
                        obj["top"] + (p["y"] - po["y"]) * sy)
                for p in obj["points"]
            ]
            shapes.append({"kind": "polygon", "points": pts})
        elif kind == "circle":  # инструмент «точки»: каждый клик — точка, соединяются по порядку
            r = obj.get("radius", 0)
            dots.append(to_orig(obj["left"] + r * sx, obj["top"] + r * sy))
    if len(dots) >= 3:
        shapes.append({"kind": dot_kind, "points": dots})
    return shapes


def canvas_dots(json_data, scale, ox, oy, px, py):
    """Возвращает список координат точек (circle) с холста в системе исходного изображения."""
    out = []
    for obj in (json_data or {}).get("objects", []):
        if obj.get("type") == "circle":
            r = obj.get("radius", 0)
            sx, sy = obj.get("scaleX", 1.0), obj.get("scaleY", 1.0)
            cx, cy = obj["left"] + r * sx, obj["top"] + r * sy
            out.append([int(round((cx - px) / scale)) + ox, int(round((cy - py) / scale)) + oy])
    return out


def bbox_of(points, w, h):
    """Вычисляет ограничивающий прямоугольник по точкам с обрезкой по размеру изображения."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = max(0, min(xs)), max(0, min(ys))
    x2, y2 = min(w, max(xs)), min(h, max(ys))
    if x2 - x1 < MIN_SIZE or y2 - y1 < MIN_SIZE:
        return None
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Отрисовка и экспорт
# ---------------------------------------------------------------------------
def hex_to_rgb(hex_color):
    """Преобразует HEX-строку цвета в кортеж RGB."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def dashed_line(draw, p1, p2, fill, width, style):
    """Рисует линию с учётом стиля: сплошная, пунктир или точечная."""
    if style == "solid":
        draw.line([p1, p2], fill=fill, width=width)
        return
    dash = width * 4 if style == "dashed" else width
    gap = width * 2
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        draw.line([(x1 + ux * pos, y1 + uy * pos), (x1 + ux * end, y1 + uy * end)],
                  fill=fill, width=width)
        pos += dash + gap


def draw_outline(draw, pts, fill, width, style):
    """Рисует замкнутый контур по списку точек с учётом стиля линии."""
    closed = pts + [pts[0]]
    for a, b in zip(closed, closed[1:]):
        dashed_line(draw, a, b, fill, width, style)


def render_annotated(img0, objects, style):
    """Отрисовывает все объекты поверх исходного изображения с подписями классов."""
    base = Image.fromarray(img0).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font_size = style["font"]
    font = get_font(font_size)
    for o in objects:
        s = style["box"] if o["kind"] == "box" else style["poly"]
        rgb = hex_to_rgb(s["color"])
        if o["kind"] == "box":
            x1, y1, x2, y2 = o["bbox"]
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            pts = [tuple(p) for p in o["points"]]
        draw_outline(d, pts, rgb + (255,), s["width"], s["style"])
        if o["kind"] == "points":
            pr = style["point"]["size"]
            pc = hex_to_rgb(style["point"]["color"]) + (255,)
            for px_, py_ in pts:
                d.ellipse([px_ - pr, py_ - pr, px_ + pr, py_ + pr], fill=pc)
        lx = min(p[0] for p in pts)
        ly = max(0, min(p[1] for p in pts) - font_size - 6)
        text = CLASSES[o["class_id"]]
        d.text((lx, ly), text, fill=hex_to_rgb(style["font_color"]) + (255,), font=font,
               stroke_width=max(1, font_size // 12), stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(base, layer).convert("RGB")


def to_record(o):
    """Преобразует внутреннее представление объекта в JSON-запись для экспорта."""
    rec = {
        "packet": o.get("packet", 1),
        "bbox": o["bbox"],
        "class": CLASSES[o["class_id"]],
        "color": o["color"],
        "shape": o["shape"],
        "transparency": o.get("transparency", "opaque"),
        "kind": o["kind"],
    }
    if o["kind"] in ("polygon", "points"):
        rec["points"] = o["points"]
    return rec


def to_yolo(objects, w, h):
    """Формирует текстовое представление объектов в формате YOLO (bbox или segmentation)."""
    lines = []
    for o in objects:
        cid = o["class_id"]
        if o["kind"] == "box":
            x1, y1, x2, y2 = o["bbox"]
            lines.append(f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        else:  # полигон -> формат YOLO segmentation
            coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in o["points"])
            lines.append(f"{cid} {coords}")
    return "\n".join(lines)


def build_zip(results, style):
    """Собирает ZIP-архив с аннотациями: JSON, YOLO .txt, аннотированные изображения и общий JSON."""
    buf = io.BytesIO()
    all_ann = {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for stem, r in results.items():
            img0 = load_image(r["src"])
            payload = {"IMG": [to_record(o) for o in r["objects"]]}
            all_ann[stem] = payload
            z.writestr(f"{stem}.json", json.dumps(payload, indent=4, ensure_ascii=False))
            z.writestr(f"{stem}.txt", to_yolo(r["objects"], r["w"], r["h"]))
            jb = io.BytesIO()
            render_annotated(img0, r["objects"], style).save(jb, "JPEG", quality=95)
            z.writestr(f"{stem}_annotated.jpg", jb.getvalue())
        z.writestr("annotations_all.json", json.dumps(all_ann, indent=4, ensure_ascii=False))
    return buf.getvalue()


def draw_zoom_frames(img0, boxes, zst):
    """Рисует рамки зума на исходном изображении для предпросмотра."""
    im = Image.fromarray(img0).copy()
    d = ImageDraw.Draw(im)
    for ox1, oy1, ox2, oy2 in boxes:
        draw_outline(d, [(ox1, oy1), (ox2, oy1), (ox2, oy2), (ox1, oy2)],
                     hex_to_rgb(zst["color"]), zst["width"], zst["style"])
    return im


def draw_zoom_box(img0, box, zst, size):
    """Уменьшает изображение под размер превью и рисует на нём рамку зума."""
    im = Image.fromarray(img0).resize(size, Image.LANCZOS)
    if box is not None:
        k = size[0] / img0.shape[1]
        ox1, oy1, ox2, oy2 = [v * k for v in box]
        draw_outline(ImageDraw.Draw(im), [(ox1, oy1), (ox2, oy1), (ox2, oy2), (ox1, oy2)],
                     hex_to_rgb(zst["color"]), zst["width"], zst["style"])
    return im


# ---------------------------------------------------------------------------
# Состояние приложения и действия
# ---------------------------------------------------------------------------
# Инициализация словаря состояния при первом запуске
if "S" not in st.session_state:
    st.session_state.S = dict(
        folder=None, files=[], index=0, zoom=None, objects=[], packets=[], cap_id=0,
        canvas_json=None, groups=[], init_draw=None, results={}, preview=None, zip=None, ann_style=None,
    )
S = st.session_state.S


def set_source(src_map, source_key):
    """Устанавливает новый источник изображений и сбрасывает состояние разметки."""
    S.update(source_key=source_key, src_map=src_map, files=list(src_map), index=0,
             zoom=None, objects=[], packets=[], cap_id=0, canvas_json=None, groups=[], init_draw=None, results={},
             preview=None, zip=None, ann_style=None)


def go(delta):
    """Переключает текущее изображение на delta позиций и сбрасывает состояние."""
    S.update(index=S["index"] + delta, zoom=None, objects=[], packets=[],
             canvas_json=None, groups=[], init_draw=None, preview=None)


def _dash(style, width):
    """Возвращает параметры пунктира для объекта на холсте."""
    if style == "dashed":
        return [width * 4, width * 2]
    if style == "dotted":
        return [1, width * 2]
    return None


def restyle_json(json_data, rect_st=None, poly_st=None, point=None):
    """Применяет текущий стиль боковой панели к фигурам, уже находящимся на холсте."""
    if not json_data:
        return None
    out = copy.deepcopy(json_data)
    for obj in out.get("objects", []):
        kind = obj.get("type")
        if kind == "rect" and rect_st:
            obj.update(stroke=rect_st["color"], strokeWidth=rect_st["width"],
                       strokeDashArray=_dash(rect_st["style"], rect_st["width"]))
        elif kind == "polygon" and poly_st:
            obj.update(stroke=poly_st["color"], strokeWidth=poly_st["width"],
                       strokeDashArray=_dash(poly_st["style"], poly_st["width"]))
        elif kind == "circle" and point:
            color, size = point
            obj.update(stroke=color, fill=color, strokeWidth=size)
    return out


def overlay_groups(img, groups, zs, ox, oy, px, py, pending=None):
    """Фон холста: уже готовые объекты плюс ломаная линия из точек, которые ставятся сейчас."""
    if not groups and not pending:
        return img
    im = Image.fromarray(img).copy()
    d = ImageDraw.Draw(im)
    rgb = hex_to_rgb(DEF_POLY["color"])
    for g in groups:
        pts = [((x - ox) * zs + px, (y - oy) * zs + py) for x, y in g["points"]]
        draw_outline(d, pts, rgb, DEF_POLY["width"], "solid")
        if g["kind"] == "points":
            pr, pc = DEF_POINT["size"], hex_to_rgb(DEF_POINT["color"])
            for x, y in pts:
                d.ellipse([x - pr, y - pr, x + pr, y + pr], fill=pc)
    if pending and len(pending) >= 2:
        line = [((x - ox) * zs + px, (y - oy) * zs + py) for x, y in pending]
        d.line(line, fill=rgb, width=DEF_POLY["width"], joint="curve")
    return np.array(im)


def apply_zoom(json_data, base_scale, dw, dh, img0):
    """Вычисляет область зума по последнему прямоугольнику на холсте и переключает приложение в режим зума."""
    rects = [o for o in (json_data or {}).get("objects", []) if o.get("type") == "rect"]
    if not rects:
        st.warning(t("nothing"))
        return
    o = rects[-1]
    sx, sy = o.get("scaleX", 1.0), o.get("scaleY", 1.0)
    x1, y1 = max(0, int(o["left"])), max(0, int(o["top"]))
    x2 = min(dw, int(o["left"] + o["width"] * sx))
    y2 = min(dh, int(o["top"] + o["height"] * sy))
    if x2 - x1 < MIN_SIZE or y2 - y1 < MIN_SIZE:
        st.warning(t("nothing"))
        return
    ox1, oy1 = int(x1 / base_scale), int(y1 / base_scale)
    ox2, oy2 = int(x2 / base_scale), int(y2 / base_scale)
    crop = img0[oy1:oy2, ox1:ox2]
    if crop.size == 0:
        return
    zs = min(dw / crop.shape[1], dh / crop.shape[0], MAX_ZOOM)
    cw_d, ch_d = int(round(crop.shape[1] * zs)), int(round(crop.shape[0] * zs))
    pad = ((dw - cw_d) // 2, (dh - ch_d) // 2)
    S.update(zoom=dict(box=(ox1, oy1, ox2, oy2), scale=zs, pad=pad), objects=[],
             canvas_json=None, groups=[], init_draw=None, preview=None)
    st.rerun()


# ---------------------------------------------------------------------------
# Боковая панель
# ---------------------------------------------------------------------------
with st.sidebar:
    # Выбор языка интерфейса
    L = st.radio("Language / Язык", ["en", "ru"], horizontal=True,
                 format_func=lambda x: "English" if x == "en" else "Русский")

    # Секция выбора источника данных
    st.header(t("sec_source"))
    NO_DEMO = t("no_demo_opt")
    demo_files = list_images(DEMO_DIR) if os.path.isdir(DEMO_DIR) else []
    if not demo_files:
        st.caption(t("no_demo"))
    demo_choice = st.selectbox(t("demo_select"), [NO_DEMO] + demo_files, key="demo_choice")
    uploads = st.file_uploader(t("upload_label"), type=["jpg", "jpeg", "png"],
                               accept_multiple_files=True, key="uploads")

    # Определение источника данных и сброс состояния при его смене
    upload_names = tuple(u.name for u in uploads or [])
    if upload_names:
        source_key = ("upload", upload_names)
    elif demo_choice != NO_DEMO:
        source_key = ("demo", tuple(demo_files))
    else:
        source_key = None

    if source_key != S.get("source_key"):
        if source_key is None:
            set_source({}, None)
        elif source_key[0] == "upload":
            set_source({u.name: u.getvalue() for u in uploads}, source_key)
        else:
            set_source({f: os.path.join(DEMO_DIR, f) for f in demo_files}, source_key)

    # выбор демо-файла переводит просмотр на этот файл
    if demo_choice != NO_DEMO and not upload_names and demo_choice != S.get("last_demo"):
        if demo_choice in S["files"]:
            S["index"] = S["files"].index(demo_choice)
    S["last_demo"] = demo_choice

    if S["files"]:
        st.caption(t("loaded", n=len(S["files"])))

    # Секция режима пакетиков
    st.markdown("---")
    st.header(t("sec_mode"))
    mode_idx = st.radio(t("sec_mode"), [0, 1], index=0, label_visibility="collapsed",
                        format_func=lambda i: t("mode_single" if i == 0 else "mode_multi"))
    if mode_idx == 0:
        st.caption(t("mode_hint"))

    # Секция настроек рамки зума
    st.markdown("---")
    st.header(t("sec_zoom"))
    zoom_color = st.color_picker(t("zoom_color"), "#FF0000")
    zoom_width = st.slider(t("zoom_width"), 1, 10, 2)
    zoom_style = st.radio(t("zoom_style"), LINE_STYLES, horizontal=True,
                          format_func=lambda s: t("line_" + s))

    # Секция стиля разметки
    st.markdown("---")
    st.header(t("sec_style"))
    box_color = st.color_picker(t("box_color"), "#00FF00")
    box_width = st.slider(t("box_width"), 1, 10, 3)
    box_style = st.radio(t("box_style"), LINE_STYLES, horizontal=True,
                         format_func=lambda s: t("line_" + s))
    poly_color = st.color_picker(t("poly_color"), "#FFA500")
    poly_width = st.slider(t("poly_width"), 1, 10, 2)
    poly_style = st.radio(t("poly_style"), LINE_STYLES, horizontal=True,
                          format_func=lambda s: t("line_" + s))
    point_color = st.color_picker(t("point_color"), "#FF00FF")
    point_size = st.slider(t("point_size"), 1, 20, 5)
    st.markdown("---")
    font_color = st.color_picker(t("font_color"), "#FFFFFF")
    font_size = st.slider(t("font_size"), 10, 60, 24)

# Стили по умолчанию, используемые на холсте на шагах 1–3
DEF_ZOOM = dict(color="#FF0000", width=2, style="solid")
DEF_BOX = dict(color="#00FF00", width=3, style="solid")
DEF_POLY = dict(color="#FFA500", width=2, style="solid")
DEF_POINT = dict(color="#FF00FF", size=5)
# Настройки из боковой панели, применяемые только на шаге 4 (предпросмотр и экспорт)
zoom_st = dict(color=zoom_color, width=zoom_width, style=zoom_style)
box_st = dict(color=box_color, width=box_width, style=box_style)
poly_st = dict(color=poly_color, width=poly_width, style=poly_style)
point_st = dict(color=point_color, size=point_size)
default_style = dict(box=box_st, poly=poly_st, font=font_size, font_color=font_color,
                     zoom=zoom_st, point=point_st)

# ---------------------------------------------------------------------------
# Основная область
# ---------------------------------------------------------------------------
st.title(t("title"))
st.caption("Annotator✳Obj")
st.caption(t("footer_data_note"))
st.caption(t("footer_copyright"))
st.caption(t("subtitle"))


# Если изображения не загружены, показываем подсказку и останавливаем приложение
if not S["files"]:
    st.info(t("choose"))
    st.stop()

idx = min(S["index"], len(S["files"]) - 1)
S["index"] = idx
fname = S["files"][idx]
stem = os.path.splitext(fname)[0]
img0 = load_image(S["src_map"][fname])
h0, w0 = img0.shape[:2]

st.subheader(t("counter", i=idx + 1, n=len(S["files"]), name=fname))
st.caption(t("orig_size", w=w0, h=h0))

# Определение количества пакетиков для текущего режима
if mode_idx == 1:
    n_packets = int(st.number_input(t("packet_count"), min_value=1, max_value=50,
                                    value=2, step=1, key="n_packets"))
else:
    n_packets = 1

# Подготовка уменьшенного изображения для холста
base = min(CANVAS_MAX / w0, CANVAS_MAX / h0, 1.0)
FW, FH = max(1, round(w0 * base)), max(1, round(h0 * base))
if S["zoom"]:
    ox1, oy1, ox2, oy2 = S["zoom"]["box"]
    zs = S["zoom"]["scale"]
    px, py = S["zoom"]["pad"]
    region = img0[oy1:oy2, ox1:ox2]
    part = cv2.resize(region, None, fx=zs, fy=zs,
                      interpolation=cv2.INTER_AREA if zs < 1 else cv2.INTER_LINEAR)
    disp = np.full((FH, FW, 3), 40, np.uint8)
    disp[py:py + part.shape[0], px:px + part.shape[1]] = part
else:
    ox1 = oy1 = px = py = 0
    zs = FW / w0
    disp = cv2.resize(img0, (FW, FH), interpolation=cv2.INTER_AREA)
dh, dw = disp.shape[:2]
disp_hash = hashlib.md5(disp.tobytes()).hexdigest()[:10]

left_col, right_col = st.columns(2)

# Левая колонка: исходное изображение с рамкой зума
with left_col:
    st.markdown(f"**{t('orig')}**")
    st.image(draw_zoom_box(img0, S["zoom"]["box"] if S["zoom"] else None, DEF_ZOOM, (FW, FH)))

# Правая колонка: холст для шага 1 (выбор зума) или шага 2 (разметка)
with right_col:
    if S["zoom"] is None:
        # Шаг 1: выбор области зума
        zoom_restyled = S.pop("zoom_restyle", None)
        if mode_idx == 1:
            if len(S["packets"]) < n_packets:
                st.markdown(f"**{t('packet_progress', k=len(S['packets']) + 1, n=n_packets)}**")
            else:
                st.info(t("all_packets_done"))
        st.markdown(f"**{t('step1')}**")
        zc = st_canvas(
            fill_color="rgba(0,0,0,0)", stroke_color=DEF_ZOOM["color"], stroke_width=DEF_ZOOM["width"],
            background_image=Image.fromarray(disp), update_streamlit=True,
            height=dh, width=dw, drawing_mode="rect",
            key=f"zoom_{fname}_{S['cap_id']}_{len(S['packets'])}_{disp_hash}",
            initial_drawing=zoom_restyled,
        )
        S["zoom_json"] = zoom_restyled if zoom_restyled is not None else (zc.json_data or S.get("zoom_json"))
        if st.button(t("zoom_btn")):
            apply_zoom(zc.json_data, zs, dw, dh, img0)
    else:
        # Шаг 2: разметка объектов
        S["zoom_json"] = None
        st.markdown(f"**{t('step2')}**")
        st.caption(t("pts_hint"))
        tool = st.radio(t("tool"), [0, 1, 2, 3], horizontal=True, key="tool",
                        format_func=lambda i: t(TOOL_LABELS[i]))
        if tool == 0:
            stroke, sw, fill = DEF_BOX["color"], DEF_BOX["width"], "rgba(0,0,0,0)"
        elif tool == 2:
            stroke, sw, fill = DEF_POINT["color"], DEF_POINT["size"], DEF_POINT["color"]
        else:
            stroke, sw, fill = DEF_POLY["color"], DEF_POLY["width"], "rgba(0,0,0,0)"
        bg = overlay_groups(disp, S["groups"], zs, ox1, oy1, px, py)
        ac = st_canvas(
            fill_color=fill, stroke_color=stroke, stroke_width=sw,
            background_image=Image.fromarray(bg), update_streamlit=True,
            height=dh, width=dw, drawing_mode=TOOL_MODES[tool], display_toolbar=True,
            key=f"annot_{fname}_{len(S['packets'])}_{S['zoom']['box']}_{S.get('cv_ver', 0)}",
            initial_drawing=S.get("init_draw"),
        )
        S["canvas_json"] = ac.json_data or S["canvas_json"]

        kind_now = "points" if tool == 2 else "polygon"
        pending = canvas_dots(S["canvas_json"], zs, ox1, oy1, px, py)
        b_fin, b_cap, b_reset = st.columns(3)
        # Завершение текущего объекта из точек
        if b_fin.button(t("finish_obj"), disabled=tool != 2 or len(pending) < 3,
                        use_container_width=True):
            S["groups"].append({"kind": kind_now, "points": pending})
            rects = [o for o in (S["canvas_json"] or {}).get("objects", []) if o.get("type") == "rect"]
            S["init_draw"] = dict(S["canvas_json"] or {}, objects=rects)  # боксы сохраняются на новом холсте
            S["cv_ver"] = S.get("cv_ver", 0) + 1                            # новый холст: без оставшихся точек
            S["canvas_json"] = None
            st.rerun()
        # Захват всех нарисованных объектов
        if b_cap.button(t("capture"), use_container_width=True):
            shapes = canvas_to_shapes(S["canvas_json"], zs, ox1, oy1, px, py, dot_kind=kind_now)
            for g in S["groups"]:
                shapes.append({"kind": g["kind"], "points": g["points"]})
            objs = []
            for sh in shapes:
                pts = [[min(max(x, 0), w0), min(max(y, 0), h0)] for x, y in sh["points"]]
                bb = bbox_of(pts, w0, h0)
                if bb is None:
                    continue
                shp, col = analyze(img0, pts, sh["kind"], bb)
                objs.append({
                    "kind": sh["kind"], "points": pts, "bbox": bb,
                    "shape": shp, "color": col, "class_id": 0, "transparency": "opaque",
                    "zoom": list(S["zoom"]["box"]),
                })
            if objs:
                S["objects"] = objs
                S["cap_id"] += 1
                S["preview"] = None
                st.success(t("captured", n=len(objs)))
            else:
                st.warning(t("nothing"))
        # Сброс зума и возврат к шагу 1
        if b_reset.button(t("reset_zoom"), use_container_width=True):
            S.update(zoom=None, objects=[], canvas_json=None, preview=None, groups=[], init_draw=None)
            st.rerun()
        # Удаление последней поставленной точки
        if tool == 2 and st.button(t("undo_pt"), disabled=not pending):
            objs = list((S["canvas_json"] or {}).get("objects", []))
            circles = [i for i, o in enumerate(objs) if o.get("type") == "circle"]
            if circles:
                del objs[circles[-1]]
                S["init_draw"] = dict(S["canvas_json"] or {}, objects=objs)
                S["canvas_json"] = None
                S["cv_ver"] = S.get("cv_ver", 0) + 1
                st.rerun()

# ---- Редактор объектов (на всю ширину, под двумя панелями) -----------------------
if S["objects"]:
    st.markdown(f"**{t('step3')}**")
    st.subheader(t("edit_objs"))
    for i, o in enumerate(S["objects"]):
        c_img, c_cls, c_shape, c_col, c_tr = st.columns([1, 1, 1, 1, 1])
        x1, y1, x2, y2 = o["bbox"]
        c_img.image(img0[y1:y2, x1:x2], width=120, caption=t("obj", i=i + 1))
        c_img.caption(f"{x2 - x1} × {y2 - y1} px")
        key_base = f"{stem}_{S['cap_id']}_{i}"
        o["class_id"] = c_cls.selectbox(
            t("cls"), list(CLASSES), index=o["class_id"],
            format_func=lambda k: CLASSES[k], key=f"cls_{key_base}")
        o["shape"] = c_shape.selectbox(
            t("shape"), SHAPES,
            index=SHAPES.index(o["shape"]) if o["shape"] in SHAPES else len(SHAPES) - 1,
            key=f"shape_{key_base}")
        o["color"] = c_col.selectbox(
            t("color"), COLORS,
            index=COLORS.index(o["color"]) if o["color"] in COLORS else COLORS.index("other"),
            key=f"color_{key_base}")
        o["transparency"] = c_tr.selectbox(
            t("transp_lbl"), TRANSPARENCY,
            index=TRANSPARENCY.index(o.get("transparency", "opaque")),
            format_func=lambda k: t(k), key=f"transp_{key_base}")


# ---- Фиксация пакетика (в режиме нескольких пакетиков) ----------------------------
if mode_idx == 1:
    if st.button(t("confirm_packet"),
                 disabled=len(S["packets"]) >= n_packets or not S["objects"]):
        S["packets"].append([dict(o) for o in S["objects"]])
        S.update(zoom=None, objects=[], canvas_json=None, groups=[], init_draw=None, preview=None)
        st.rerun()
    st.caption(t("packets_done", k=len(S["packets"]), n=n_packets))

# ---- Сохранение -----------------------------------------------------------------
if st.button(t("save"), type="primary"):
    packets = S["packets"] + ([S["objects"]] if S["objects"] else [])
    objs_all = []
    for pno, pk in enumerate(packets, start=1):
        for o in pk:
            c = dict(o)
            c["packet"] = pno
            objs_all.append(c)
    if not objs_all:
        st.warning(t("no_objs_save"))
    else:
        S["results"][stem] = {
            "file": fname, "src": S["src_map"][fname], "w": w0, "h": h0,
            "objects": objs_all,
            "zooms": [list(z) for z in sorted({tuple(o["zoom"]) for o in objs_all if o.get("zoom")})],
        }
        S["preview"] = stem
        S["zip"] = None
        st.success(t("saved"))

# ---- Предпросмотр: исходное (слева) против аннотированного (справа) --------------
if S["preview"] == stem and stem in S["results"]:
    r = S["results"][stem]
    st.subheader(t("step4"))
    st.caption(t("pv_hint"))
    pv_style = default_style
    S["pv_style"] = pv_style
    annotated = render_annotated(img0, r["objects"], pv_style)
    original = draw_zoom_frames(img0, r.get("zooms", []), pv_style["zoom"])
    c_left, c_right = st.columns(2)
    c_left.image(original, caption=t("orig"))
    c_right.image(annotated, caption=t("annotated"))

# ---- Таблица всех аннотаций и экспорт --------------------------------------------
if S["results"]:
    st.divider()
    st.subheader(t("results"))
    rows = []
    for r in S["results"].values():
        for i, o in enumerate(r["objects"]):
            rows.append({
                "image": r["file"],
                "packet": o.get("packet", 1),
                "#": i + 1,
                "class": CLASSES[o["class_id"]],
                "shape": o["shape"],
                "color": o["color"],
                "transparency": o.get("transparency", "opaque"),
                "kind": o["kind"],
                "size_px": f"{o['bbox'][2] - o['bbox'][0]} × {o['bbox'][3] - o['bbox'][1]}",
                "bbox": str(o["bbox"]),
                "points": len(o["points"]),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Подготовка и скачивание ZIP-архива с результатами
    if st.button(t("export")):
        S["zip"] = build_zip(S["results"], S.get("pv_style") or default_style)
    if S["zip"]:
        st.download_button(t("download"), S["zip"], file_name="annotations.zip",
                           mime="application/zip")
    st.caption(t("export_hint"))

# ---- Навигация по изображениям ---------------------------------------------------
st.divider()
n1, n2 = st.columns(2)
n1.button(t("prev"), on_click=go, args=(-1,), disabled=idx == 0, use_container_width=True)
n2.button(t("next"), on_click=go, args=(1,), disabled=idx == len(S["files"]) - 1,
          use_container_width=True)