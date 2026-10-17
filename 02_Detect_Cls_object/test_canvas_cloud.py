import importlib.metadata as md

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Canvas test", layout="wide")
st.title("Тест холста: локально vs Streamlit Cloud")


def package_versions():
    names = ["streamlit", "streamlit-drawable-canvas",
             "streamlit-image-coordinates", "Pillow", "numpy"]
    out = {}
    for name in names:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "not installed"
    return out


def make_test_image(w=640, h=420):
    im = Image.new("RGB", (w, h), (235, 235, 235))
    d = ImageDraw.Draw(im)
    for x in range(0, w, 40):
        d.line([(x, 0), (x, h)], fill=(200, 200, 200))
    for y in range(0, h, 40):
        d.line([(0, y), (w, y)], fill=(200, 200, 200))
    for cx, cy in [(120, 120), (300, 200), (500, 300)]:
        d.ellipse([cx - 40, cy - 25, cx + 40, cy + 25], fill=(220, 60, 60))
    d.text((10, 10), "TEST IMAGE", fill=(0, 0, 0))
    return im


with st.expander("Версии пакетов", expanded=True):
    st.json(package_versions())

uploaded = st.file_uploader("Свое изображение (необязательно)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
else:
    img = make_test_image()

# Масштабируем до разумного размера, чтобы холст помещался на экран
MAX_W = 640
if img.width > MAX_W:
    k = MAX_W / img.width
    img = img.resize((MAX_W, round(img.height * k)))

st.caption(f"Размер изображения: {img.width} × {img.height}")

col1, col2, col3 = st.columns(3)

# 1. Контроль: обычный вывод изображения
with col1:
    st.subheader("1. st.image")
    st.image(img, width=img.width)
    st.caption("Если здесь картинки нет, проблема в самом сервере или в загрузке файла.")

# 2. Холст из streamlit-drawable-canvas с фоном
with col2:
    st.subheader("2. st_canvas (фон)")
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=img,
        update_streamlit=True,
        height=img.height,
        width=img.width,
        drawing_mode="rect",
        key="test_canvas",
    )
    n = len((canvas.json_data or {}).get("objects", []))
    st.caption(f"Объектов на холсте: {n}. Нарисуйте прямоугольник.")

# 3. Клики по изображению через streamlit_image_coordinates
with col3:
    st.subheader("3. image_coordinates (клики)")
    click = streamlit_image_coordinates(img, key="test_coords", width=img.width)
    if click:
        st.caption(f"Последний клик: x={click['x']}, y={click['y']}")
    else:
        st.caption("Кликните по изображению.")