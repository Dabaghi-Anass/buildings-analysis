"""
Building Detection API — CLIP fine-tuned + MLP classifier
No ensemble, no ResNet, no MobileNet.

Checkpoint layout (produced by the notebook):
  clip_mlp_checkpoints/
    clip_finetuned.pt   — fine-tuned CLIP vision_model state dict
    mlp_head.pt         — trained MLPHead state dict
    meta.json           — {"classes": [...], "mlp_hidden": [...], "mlp_dropout": 0.3, ...}
"""

import warnings
warnings.filterwarnings("ignore")

import json
import math
import os
import numpy as np
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor, CLIPConfig

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from shapely.geometry import shape, box as shapely_box, mapping
from dotenv import load_dotenv
import os

load_dotenv()


token = os.environ.get("HF_TOKEN")  
print(f"[INFO] Using Hugging Face token: {'yes' if token else 'no'}")
# ── Config ────────────────────────────────────────────────────────────────────
# BUILDINGS_PATH = "buildings.geojson"
BUILDINGS_PATH = "buildings_casablanca.geojson"
SAVE_DIR       = Path("clip_mlp_checkpoints")
CLIP_CKPT      = SAVE_DIR / "clip_finetuned.pt"
MLP_CKPT       = SAVE_DIR / "mlp_head.pt"
META_PATH      = SAVE_DIR / "meta.json"

PADDING_METERS = 2
TILE_SIZE      = 256
ZOOM           = 19
TILE_URL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery"
    "/WMTS/1.0.0/default028mm/MapServer/tile/{z}/{y}/{x}"
)
HEADERS = {"User-Agent": "BuildingBBoxExtractor/1.0"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── MLP architecture (must match the notebook exactly) ───────────────────────
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: list[int],
                 num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Global model cache ────────────────────────────────────────────────────────
model_cache: dict[str, Any] = {
    "clip_model": None,
    "processor":  None,
    "mlp":        None,
    "classes":    [],
}


def load_models() -> None:
    """Load CLIP (fine-tuned) + MLP once at startup."""
    print("[INFO] Loading CLIP + MLP models…")

    if not META_PATH.exists():
        raise RuntimeError(f"meta.json not found at {META_PATH}")

    with open(META_PATH) as f:
        meta = json.load(f)

    classes    = meta["classes"]
    hidden     = meta.get("mlp_hidden",   [512, 256])
    dropout    = meta.get("mlp_dropout",  0.3)
    num_classes = len(classes)

    # ── CLIP ─────────────────────────────────────────────────────────────────
    processor  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", token=token)
    # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    config = CLIPConfig()
    clip_model = CLIPModel._from_config(config).to(device)
    if CLIP_CKPT.exists():
        clip_model.vision_model.load_state_dict(
            torch.load(CLIP_CKPT, map_location=device)
        )
        print(f"[INFO] Fine-tuned CLIP weights loaded from {CLIP_CKPT}")
    else:
        print(f"[WARN] {CLIP_CKPT} not found — using vanilla ImageNet CLIP weights.")

    clip_model.eval()

    # ── MLP ──────────────────────────────────────────────────────────────────
    mlp = MLPHead(768, hidden, num_classes, dropout).to(device)

    if MLP_CKPT.exists():
        mlp.load_state_dict(torch.load(MLP_CKPT, map_location=device))
        print(f"[INFO] MLP weights loaded from {MLP_CKPT}")
    else:
        raise RuntimeError(f"MLP checkpoint not found at {MLP_CKPT}")

    mlp.eval()

    model_cache["clip_model"] = clip_model
    model_cache["processor"]  = processor
    model_cache["mlp"]        = mlp
    model_cache["classes"]    = classes

    print(f"[INFO] Models ready — classes: {classes}")


# ── Feature extraction + classification ──────────────────────────────────────
def classify_image_array(img_array: np.ndarray) -> tuple[str, float] | None:
    """
    Classify a (H, W, 3) uint8 numpy array.
    Returns (class_name, confidence) or None if models not loaded.
    """
    clip_model = model_cache["clip_model"]
    processor  = model_cache["processor"]
    mlp        = model_cache["mlp"]
    classes    = model_cache["classes"]

    if clip_model is None or mlp is None or not classes:
        return None

    try:
        # Resize to 224×224 if needed
        if img_array.shape[:2] != (224, 224):
            img_array = np.array(Image.fromarray(img_array).resize((224, 224)))

        # CLIP features
        with torch.no_grad():
            inputs = processor(images=img_array, return_tensors="pt")
            pv     = inputs["pixel_values"].to(device)
            out    = clip_model.vision_model(pixel_values=pv)
            feat   = out.pooler_output          # (1, 768)

        # MLP
        with torch.no_grad():
            logits = mlp(feat)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx   = int(probs.argmax())
        pred_class = classes[pred_idx]
        pred_conf  = float(probs[pred_idx])
        return pred_class, pred_conf

    except Exception as e:
        print(f"[WARN] Classification failed: {e}")
        return None


# ── Buildings ─────────────────────────────────────────────────────────────────
buildings: list[dict] = []

def load_buildings() -> None:
    global buildings
    if not Path(BUILDINGS_PATH).exists():
        print(f"[WARN] {BUILDINGS_PATH} not found.")
        return
    with open(BUILDINGS_PATH) as f:
        fc = json.load(f)
    buildings = fc.get("features", [])
    print(f"[INFO] Loaded {len(buildings)} buildings from {BUILDINGS_PATH}")


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_buildings()
    load_models()
    yield


app = FastAPI(title="Building Detection API — CLIP+MLP", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    area: dict
    zoom: int = ZOOM


# ── Geo helpers ───────────────────────────────────────────────────────────────
def meters_to_deg_lat(m: float) -> float:
    return m / 111_320

def meters_to_deg_lng(m: float, lat: float) -> float:
    return m / (111_320 * math.cos(math.radians(lat)))

def padded_bbox(minx, miny, maxx, maxy, pad_m=PADDING_METERS):
    clat = (miny + maxy) / 2
    dlat = meters_to_deg_lat(pad_m)
    dlng = meters_to_deg_lng(pad_m, clat)
    return (minx - dlng, miny - dlat, maxx + dlng, maxy + dlat)

def lng_lat_to_tile(lng, lat, zoom):
    n = 2 ** zoom
    xtile = int((lng + 180) / 360 * n)
    lat_r = math.radians(lat)
    ytile = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return xtile, ytile

def tile_to_lng_lat(x, y, zoom):
    n = 2 ** zoom
    lng = x / n * 360 - 180
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return lng, math.degrees(lat_r)

def fetch_tile(x, y, z) -> Image.Image | None:
    url = TILE_URL.format(x=x, y=y, z=z)
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"[WARN] Tile {z}/{x}/{y} failed: {e}")
        return None

def stitch_and_crop(bbox_padded, zoom) -> Image.Image | None:
    minx, miny, maxx, maxy = bbox_padded

    tx_min, ty_min = lng_lat_to_tile(minx, maxy, zoom)
    tx_max, ty_max = lng_lat_to_tile(maxx, miny, zoom)

    w_tiles = tx_max - tx_min + 1
    h_tiles = ty_max - ty_min + 1
    if w_tiles * h_tiles > 64:
        print(f"[WARN] Too many tiles ({w_tiles}×{h_tiles}), skipping.")
        return None

    canvas = Image.new("RGB", (w_tiles * TILE_SIZE, h_tiles * TILE_SIZE))
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            tile = fetch_tile(tx, ty, zoom)
            if tile:
                canvas.paste(tile, ((tx - tx_min) * TILE_SIZE, (ty - ty_min) * TILE_SIZE))

    def lng_to_px(lng):
        return int((lng + 180) / 360 * (2 ** zoom) * TILE_SIZE) - tx_min * TILE_SIZE

    def lat_to_py(lat):
        lat_r  = math.radians(lat)
        world_y = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) \
                  / 2 * (2 ** zoom) * TILE_SIZE
        return int(world_y) - ty_min * TILE_SIZE

    left   = max(0, lng_to_px(minx))
    right  = min(canvas.width,  lng_to_px(maxx))
    top    = max(0, lat_to_py(maxy))
    bottom = min(canvas.height, lat_to_py(miny))

    if right <= left or bottom <= top:
        return canvas
    return canvas.crop((left, top, right, bottom))


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/api/process")
def process_area(req: ProcessRequest):
    if not buildings:
        raise HTTPException(status_code=500, detail="No buildings loaded on server.")

    area_geom  = req.area.get("geometry") or req.area
    area_shape = shape(area_geom)

    results = []

    for idx, feature in enumerate(buildings):
        if not feature.get("geometry"):
            continue

        props      = dict(feature.get("properties") or {})
        feat_shape = shape(feature["geometry"])
        if not area_shape.intersects(feat_shape):
            continue

        minx, miny, maxx, maxy         = feat_shape.bounds
        pminx, pminy, pmaxx, pmaxy     = padded_bbox(minx, miny, maxx, maxy)
        name                            = props.get("name") or props.get("id") or f"building_{idx}"

        pred_class, pred_conf = None, 0.0

        img = stitch_and_crop((pminx, pminy, pmaxx, pmaxy), req.zoom)
        if img is not None:
            result = classify_image_array(np.array(img))
            if result:
                pred_class, pred_conf = result
                print(f"[INFO] {name} → {pred_class} ({pred_conf:.2%})")

        results.append({
            "type": "Feature",
            "properties": {
                **props,
                "_source_idx":  idx,
                "_bbox_padded": [pminx, pminy, pmaxx, pmaxy],
                "__bbox":       [minx, miny, maxx, maxy],
                "_crop_file":   None,
                "_class":       pred_class,
                "_confidence":  pred_conf,
            },
            "bbox":     [pminx, pminy, pmaxx, pmaxy],
            "geometry": mapping(shapely_box(pminx, pminy, pmaxx, pmaxy)),
        })

    return {
        "type": "FeatureCollection",
        "features": results,
        "meta": {
            "total_buildings": len(buildings),
            "matched":         len(results),
            "classified":      sum(1 for f in results if f["properties"]["_class"]),
            "zoom_used":       req.zoom,
            "padding_m":       PADDING_METERS,
            "classes":         model_cache["classes"],
        },
    }


@app.get("/api/health")
def health():
    return {
        "status":          "ok",
        "buildings_loaded": len(buildings),
        "classes":         model_cache["classes"],
        "clip_loaded":     model_cache["clip_model"] is not None,
        "mlp_loaded":      model_cache["mlp"] is not None,
    }
