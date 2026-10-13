
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2, numpy as np, json, os
from PIL import Image, ImageOps   # 🔧 FIX 1

st.set_page_config(layout="wide")
st.title("Streamlit Capsule Annotator — Crop-Zoom + YOLO + Annotated Image Saving")

OUTPUT_DIR = "01_Streamlit_result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = {0: 'capsule', 1: 'softgel_gelcap', 2: 'meltlet', 3: 'caplet'}
SHAPES = ["round", "oval", "elongated", "square", "other"]
COLORS = ["white", "yellow", "red", "black", "green", "brown", "gray", "orange", "other"]

MAX_W, MAX_H = 1200, 900

# === INIT SESSION STATE ===
if "state" not in st.session_state:
    st.session_state.state = dict(
        folder=None, files=[], index=0,
        img0=None, img=None, scale=1.0,
        crop_mode=True, crop_box=None,
        raw_boxes=[], boxes=[], canvas_json=None
    )

S = st.session_state.state

# === FOLDER INPUT ===
folder = st.text_input("Enter folder path with images:")

if folder:
    if os.path.isdir(folder) and folder != S["folder"]:
        S.update(dict(
            folder=folder,
            files=sorted([f for f in os.listdir(folder)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))]),
            index=0, crop_mode=True, crop_box=None,
            raw_boxes=[], boxes=[]
        ))
        st.success(f"Loaded {len(S['files'])} images.")
    elif not os.path.isdir(folder):
        st.error("Invalid folder path.")

# === LOAD IMAGE ===
if not S["files"]:
    st.stop()

idx = S["index"]
current_file = S["files"][idx]
path = os.path.join(S["folder"], current_file)

# 🔧 FIX 1: EXIF-safe image loading
img_pil = Image.open(path)
img_pil = ImageOps.exif_transpose(img_pil)
S["img0"] = np.array(img_pil.convert("RGB"))

h0, w0 = S["img0"].shape[:2]
base_scale = min(MAX_W / w0, MAX_H / h0, 1.0)

if S["crop_mode"]:
    S["img"] = cv2.resize(S["img0"], None, fx=base_scale, fy=base_scale)
    S["scale"] = base_scale

h, w = S["img"].shape[:2]

st.subheader(f"Image {idx+1}/{len(S['files'])}: {current_file}")
st.caption(f"Original size: {w0} × {h0}")

# === STEP 1: CROP SELECTION MODE ===
if S["crop_mode"]:
    st.info("Step 1: Draw a rectangle to select zoom area")

    crop_canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_color="red",
        stroke_width=2,
        background_image=Image.fromarray(S["img"]),
        update_streamlit=True,
        height=h, width=w,
        drawing_mode="rect",
        key=f"crop_{current_file}"
    )

    if st.button("Zoom to selection"):
        data = crop_canvas.json_data
        if data and data["objects"]:
            obj = data["objects"][-1]
            if obj["type"] == "rect":
                x1 = max(0, int(obj["left"]))
                y1 = max(0, int(obj["top"]))
                x2 = min(w - 1, int(obj["left"] + obj["width"]))
                y2 = min(h - 1, int(obj["top"] + obj["height"]))

                ox1, oy1 = int(x1 / base_scale), int(y1 / base_scale)
                ox2, oy2 = int(x2 / base_scale), int(y2 / base_scale)

                crop = S["img0"][oy1:oy2, ox1:ox2]
                zoom_scale = min(MAX_W / crop.shape[1], MAX_H / crop.shape[0], 3.0)

                S.update({
                    "img": cv2.resize(crop, None, fx=zoom_scale, fy=zoom_scale),
                    "scale": zoom_scale,
                    "crop_mode": False,
                    "crop_box": (ox1, oy1, ox2, oy2)
                })
                st.rerun()

    st.stop()

# === STEP 2: ANNOTATION MODE ===
st.success("Zoom applied. Now annotate the enlarged region.")

tool_mode = st.radio("Selection Tool:", ("Draw", "Edit/Move"), horizontal=True)
mode = "rect" if tool_mode == "Draw" else "transform"

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_color="green",
    stroke_width=2,
    background_image=Image.fromarray(S["img"]),
    update_streamlit=True,
    height=h, width=w,
    drawing_mode=mode,
    display_toolbar=True,
    key=f"annot_{current_file}"
)

S["canvas_json"] = canvas_result.json_data if canvas_result.json_data else S["canvas_json"]

# === DETECTION FUNCTIONS ===
def detect_color(img):
    if img.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = np.mean(hsv, axis=(0, 1)).astype(int)

    if v < 30: return "black"
    if s < 30 and v > 200: return "white"
    if s < 30 and v > 150: return "gray"

    if (h < 10 or h > 170) and s > 50: return "red"
    if 10 <= h < 25 and s > 50: return "orange"
    if 25 <= h < 35 and s > 50: return "yellow"
    if 35 <= h < 85 and s > 50: return "green"
    if 10 <= h < 25 and 30 <= s <= 100 and 50 <= v <= 150: return "brown"

    return "other"


def detect_shape(img):
    if img.size == 0:
        return "unknown"

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        return "unknown"

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)

    x, y, wd, ht = cv2.boundingRect(cnt)
    aspect_ratio = max(wd, ht) / (min(wd, ht) + 1e-6)

    if circularity > 0.85 and aspect_ratio < 1.2: return "round"
    if 0.6 < circularity <= 0.85 and 1.2 <= aspect_ratio < 2.0: return "oval"
    if circularity <= 0.6 and aspect_ratio >= 2.0: return "elongated"
    if 0.7 < circularity <= 0.9 and 1.0 <= aspect_ratio < 1.2:
        rect_area = wd * ht
        if 0.8 <= area / rect_area <= 1.0:
            return "square"

    return "other"

# === CAPTURE BOXES ===
if st.button("Capture boxes"):
    if S["canvas_json"] and "objects" in S["canvas_json"]:
        scale = S["scale"]
        ox1, oy1 = S["crop_box"][:2]
        new_raw = []

        for obj in S["canvas_json"]["objects"]:
            if obj["type"] != "rect":
                continue

            sX = obj.get("scaleX", 1.0)
            sY = obj.get("scaleY", 1.0)

            w_scaled = obj["width"] * sX
            h_scaled = obj["height"] * sY

            # 🔧 FIX 2: float → round at final stage
            x1 = float(obj["left"])
            y1 = float(obj["top"])
            x2 = x1 + w_scaled
            y2 = y1 + h_scaled

            orig_x1 = int(round(x1 / scale)) + ox1
            orig_y1 = int(round(y1 / scale)) + oy1
            orig_x2 = int(round(x2 / scale)) + ox1
            orig_y2 = int(round(y2 / scale)) + oy1

            crop_img = S["img0"][orig_y1:orig_y2, orig_x1:orig_x2]

            new_raw.append({
                "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                "auto_shape": detect_shape(crop_img),
                "auto_color": detect_color(crop_img)
            })

        S["raw_boxes"] = new_raw
        S["boxes"] = [{
            "bbox": rb["bbox"],
            "shape": rb["auto_shape"],
            "color": rb["auto_color"],
            "class_id": 0
        } for rb in new_raw]

        st.success(f"Captured {len(new_raw)} boxes.")
    else:
        st.error("No rectangles found.")

# === EDIT BOXES ===
if S["raw_boxes"]:
    st.subheader("Edit detected boxes")
    final_boxes = []

    for i, rb in enumerate(S["raw_boxes"]):
        x1, y1, x2, y2 = rb["bbox"]
        st.image(S["img0"][y1:y2, x1:x2], width=200)

        shape = st.selectbox(
            f"Shape {i+1}", SHAPES,
            index=SHAPES.index(rb["auto_shape"]) if rb["auto_shape"] in SHAPES else 0,
            key=f"shape_{i}"
        )

        color = st.selectbox(
            f"Color {i+1}", COLORS,
            index=COLORS.index(rb["auto_color"]) if rb["auto_color"] in COLORS else 8,
            key=f"color_{i}"
        )

        class_id = st.selectbox(
            f"Class {i+1}", [0, 1, 2, 3],
            format_func=lambda x: CLASSES[x],
            key=f"class_{i}"
        )

        final_boxes.append({
            "bbox": rb["bbox"],
            "shape": shape,
            "color": color,
            "class_id": class_id
        })

    S["boxes"] = final_boxes

# === SAVE FUNCTIONS ===
def save_image_with_boxes(img0, boxes, out_path):
    img_copy = img0.copy()
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        class_name = CLASSES[b["class_id"]]
        cv2.putText(img_copy, class_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(out_path, cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))


def save_yolo(img0, boxes, out_path):
    h0, w0 = img0.shape[:2]
    lines = [
        f"{b['class_id']} "
        f"{(b['bbox'][0] + b['bbox'][2]) / 2 / w0:.6f} "
        f"{(b['bbox'][1] + b['bbox'][3]) / 2 / h0:.6f} "
        f"{(b['bbox'][2] - b['bbox'][0]) / w0:.6f} "
        f"{(b['bbox'][3] - b['bbox'][1]) / h0:.6f}"
        for b in boxes
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

# === SAVE ANNOTATION ===
if st.button("Save annotation"):
    if not S["boxes"]:
        st.warning("No boxes.")
    else:
        file_stem = current_file.rsplit(".", 1)[0]

        formatted = [{
            "bbox": b["bbox"],
            "class": CLASSES[b["class_id"]],
            "color": b["color"],
            "shape": b["shape"]
        } for b in S["boxes"]]

        json_path = os.path.join(OUTPUT_DIR, file_stem + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"IMG": formatted}, f, indent=4, ensure_ascii=False)

        yolo_path = os.path.join(OUTPUT_DIR, file_stem + ".txt")
        save_yolo(S["img0"], S["boxes"], yolo_path)

        annotated_path = os.path.join(OUTPUT_DIR, file_stem + "_annotated.jpg")
        save_image_with_boxes(S["img0"], S["boxes"], annotated_path)

        all_annotations = {}
        for fname in S["files"]:
            stem = fname.rsplit(".", 1)[0]
            per_json = os.path.join(OUTPUT_DIR, stem + ".json")
            if os.path.exists(per_json):
                with open(per_json, "r", encoding="utf-8") as f:
                    all_annotations[stem] = json.load(f)

        with open(os.path.join(OUTPUT_DIR, "annotations_all.json"), "w", encoding="utf-8") as f:
            json.dump(all_annotations, f, indent=4, ensure_ascii=False)

        st.success("Saved JSON, YOLO TXT, annotated image and updated annotations_all.json")

# === NAVIGATION ===
col1, col2 = st.columns(2)

with col1:
    st.button(
        "← Previous",
        on_click=lambda: (S.update({"index": S["index"] - 1, "crop_mode": True}), st.rerun()),
        disabled=(idx == 0)
    )

with col2:
    st.button(
        "Next →",
        on_click=lambda: (S.update({"index": S["index"] + 1, "crop_mode": True}), st.rerun()),
        disabled=(idx == len(S["files"]) - 1)
    )
