"""
Building Detection API — Multi-mode classifier
═══════════════════════════════════════════════

Mode 1 — clip_mlp
  • Iterates GeoJSON building polygons that intersect the AOI
  • Downloads a padded satellite crop per polygon
  • Classifies each crop with CLIP vision encoder + MLP head
  • Returns one Feature per polygon

Mode 2 — rf_detr
  • Downloads the full AOI raster at zoom 19
  • Slices it into 200 m × 200 m grid cells
  • Sends each cell to the RF-DETR Roboflow workflow
  • Collects all bounding-box detections, converts pixel → geo coords
  • Returns one Feature per detected box (no GeoJSON polygon database needed)

Mode 3 — rf_detr_clip
  • Same grid raster approach as rf_detr for detection
  • After RF-DETR locates each box, crops that sub-image from the cell
  • Reclassifies the crop with CLIP+MLP
  • Returns one Feature per box, with both RF-DETR and CLIP labels

Checkpoint layout:
  clip_mlp_checkpoints/
    clip_finetuned.pt
    mlp_head.pt
    meta.json
"""

import warnings
warnings.filterwarnings("ignore")

import base64
import json
import math
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from io import BytesIO

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, CLIPConfig

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from shapely.geometry import shape, box as shapely_box, mapping
from dotenv import load_dotenv

# ── Roboflow SDK ──────────────────────────────────────────────────────────────
try:
    from inference_sdk import InferenceHTTPClient
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("[WARN] inference_sdk not installed — RF-DETR modes will be unavailable.")

load_dotenv()

token      = os.environ.get("HF_TOKEN")
RF_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "cDZ181T7dt4udXzJCIBh")
RF_WS      = os.environ.get("ROBOFLOW_WORKSPACE", "anassusmba")
RF_WF      = os.environ.get("ROBOFLOW_WORKFLOW",  "custom-workflow-2")

print(f"[INFO] HF token: {'yes' if token else 'no'}")
print(f"[INFO] Roboflow SDK available: {ROBOFLOW_AVAILABLE}")

# ── Config ────────────────────────────────────────────────────────────────────
BUILDINGS_PATH  = "buildings.geojson"
SAVE_DIR        = Path("clip_mlp_checkpoints_97")
CLIP_CKPT       = SAVE_DIR / "clip_finetuned.pt"
MLP_CKPT        = SAVE_DIR / "mlp_head.pt"
META_PATH       = SAVE_DIR / "meta.json"

PADDING_METERS  = 2
TILE_SIZE       = 256          # pixels per OSM/WMTS tile
ZOOM            = 19
CELL_METERS     = 75          # grid cell size for RF-DETR modes (meters)
MAX_CELLS       = 64           # safety: refuse if AOI produces more than this

TILE_URL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery"
    "/WMTS/1.0.0/default028mm/MapServer/tile/{z}/{y}/{x}"
)
HEADERS = {"User-Agent": "BuildingBBoxExtractor/1.0"}
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# MLP architecture (must match training notebook exactly)
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# Global caches
# ══════════════════════════════════════════════════════════════════════════════
model_cache: dict[str, Any] = {
    "clip_model": None,
    "processor":  None,
    "mlp":        None,
    "classes":    [],
}
rf_client: Any = None


# ══════════════════════════════════════════════════════════════════════════════
# Model loaders
# ══════════════════════════════════════════════════════════════════════════════
def load_clip_mlp() -> None:
    print("[INFO] Loading CLIP + MLP models…")
    if not META_PATH.exists():
        raise RuntimeError(f"meta.json not found at {META_PATH}")

    with open(META_PATH) as f:
        meta = json.load(f)

    classes     = meta["classes"]
    hidden      = meta.get("mlp_hidden",  [512, 256])
    dropout     = meta.get("mlp_dropout", 0.3)
    num_classes = len(classes)

    processor  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", token=token)
    config     = CLIPConfig()
    clip_model = CLIPModel._from_config(config).to(device)
    if CLIP_CKPT.exists():
        clip_model.vision_model.load_state_dict(
            torch.load(CLIP_CKPT, map_location=device)
        )
        print(f"[INFO] Fine-tuned CLIP weights loaded from {CLIP_CKPT}")
    else:
        print(f"[WARN] {CLIP_CKPT} not found — using vanilla CLIP weights.")
    clip_model.eval()

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
    print(f"[INFO] CLIP+MLP ready — classes: {classes}")


def load_rf_client() -> None:
    global rf_client
    if not ROBOFLOW_AVAILABLE:
        print("[WARN] Skipping RF client — inference_sdk not installed.")
        return
    rf_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=RF_API_KEY,
    )
    print("[INFO] Roboflow InferenceHTTPClient ready.")


def load_models() -> None:
    load_rf_client()
    try:
        load_clip_mlp()
    except Exception as e:
        print(f"[WARN] CLIP+MLP failed to load: {e}. RF-DETR modes still available.")


# ══════════════════════════════════════════════════════════════════════════════
# CLIP+MLP classification
# ══════════════════════════════════════════════════════════════════════════════
def classify_image_array(img_array: np.ndarray) -> tuple[str, float] | None:
    clip_model = model_cache["clip_model"]
    processor  = model_cache["processor"]
    mlp        = model_cache["mlp"]
    classes    = model_cache["classes"]

    if clip_model is None or mlp is None or not classes:
        return None

    try:
        if img_array.shape[:2] != (224, 224):
            img_array = np.array(Image.fromarray(img_array).resize((224, 224)))

        with torch.no_grad():
            inputs = processor(images=img_array, return_tensors="pt")
            pv     = inputs["pixel_values"].to(device)
            out    = clip_model.vision_model(pixel_values=pv)
            feat   = out.pooler_output

        with torch.no_grad():
            logits = mlp(feat)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx   = int(probs.argmax())
        pred_class = classes[pred_idx]
        pred_conf  = float(probs[pred_idx])
        return pred_class, pred_conf

    except Exception as e:
        print(f"[WARN] CLIP+MLP classification failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Geo / tile helpers
# ══════════════════════════════════════════════════════════════════════════════
def meters_to_deg_lat(m: float) -> float:
    return m / 111_320

def meters_to_deg_lng(m: float, lat: float) -> float:
    return m / (111_320 * math.cos(math.radians(lat)))

def padded_bbox(minx, miny, maxx, maxy, pad_m=PADDING_METERS):
    clat = (miny + maxy) / 2
    dlat = meters_to_deg_lat(pad_m)
    dlng = meters_to_deg_lng(pad_m, clat)
    return (minx - dlng, miny - dlat, maxx + dlng, maxy + dlat)

def lng_lat_to_tile(lng: float, lat: float, zoom: int) -> tuple[int, int]:
    n     = 2 ** zoom
    xtile = int((lng + 180) / 360 * n)
    lat_r = math.radians(lat)
    ytile = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return xtile, ytile

def fetch_tile(x: int, y: int, z: int) -> Image.Image | None:
    url = TILE_URL.format(x=x, y=y, z=z)
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"[WARN] Tile {z}/{x}/{y} failed: {e}")
        return None

def geo_to_pixel(
    lng: float, lat: float,
    tx_min: int, ty_min: int, zoom: int
) -> tuple[int, int]:
    """Geographic (lng, lat) → pixel (x, y) in the stitched canvas."""
    n  = 2 ** zoom
    px = int((lng + 180) / 360 * n * TILE_SIZE) - tx_min * TILE_SIZE
    lr = math.radians(lat)
    py = int(
        (1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi)
        / 2 * n * TILE_SIZE
    ) - ty_min * TILE_SIZE
    return px, py

def pixel_to_geo(
    px: int, py: int,
    tx_min: int, ty_min: int, zoom: int
) -> tuple[float, float]:
    """Pixel (x, y) in the stitched canvas → geographic (lng, lat)."""
    n   = 2 ** zoom
    lng = (px + tx_min * TILE_SIZE) / (n * TILE_SIZE) * 360 - 180
    my  = math.pi * (1 - 2 * (py + ty_min * TILE_SIZE) / (n * TILE_SIZE))
    lat = math.degrees(math.atan(math.sinh(my)))
    return lng, lat


# ══════════════════════════════════════════════════════════════════════════════
# AOI full-raster downloader
# ══════════════════════════════════════════════════════════════════════════════
def download_aoi_raster(
    minx: float, miny: float, maxx: float, maxy: float, zoom: int,
    max_workers: int = 8
) -> tuple[Image.Image, int, int] | None:
    """
    Stitch all zoom-level tiles covering the bounding box into one canvas.
    Fetches tiles in parallel using multithreading for speed.
    Returns (canvas_image, tx_min, ty_min) or None on failure.
    """
    tx_min, ty_min = lng_lat_to_tile(minx, maxy, zoom)   # NW corner
    tx_max, ty_max = lng_lat_to_tile(maxx, miny, zoom)   # SE corner

    w_tiles = tx_max - tx_min + 1
    h_tiles = ty_max - ty_min + 1
    total   = w_tiles * h_tiles

    if total > 256:
        print(f"[WARN] AOI requires {total} tiles — too large, aborting.")
        return None

    print(f"[INFO] Downloading AOI raster: {w_tiles}×{h_tiles} tiles ({total} total)…")
    canvas = Image.new("RGB", (w_tiles * TILE_SIZE, h_tiles * TILE_SIZE))

    # Collect all tile coordinates
    tile_coords: list[tuple[int, int, int, int]] = []
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            tile_coords.append((tx, ty, zoom, tx - tx_min))

    # Fetch tiles in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a mapping of future → (offset_x, offset_y)
        future_to_offset = {
            executor.submit(fetch_tile, tx, ty, z): (ox * TILE_SIZE, (ty - ty_min) * TILE_SIZE)
            for tx, ty, z, ox in tile_coords
        }
        
        for future in as_completed(future_to_offset):
            tile = future.result()
            if tile:
                offset_x, offset_y = future_to_offset[future]
                canvas.paste(tile, (offset_x, offset_y))

    return canvas, tx_min, ty_min


# def download_aoi_raster(
#     minx: float, miny: float, maxx: float, maxy: float, zoom: int
# ) -> tuple[Image.Image, int, int] | None:
#     """
#     Stitch all zoom-level tiles covering the bounding box into one canvas.
#     Returns (canvas_image, tx_min, ty_min) or None on failure.
#     """
#     tx_min, ty_min = lng_lat_to_tile(minx, maxy, zoom)   # NW corner
#     tx_max, ty_max = lng_lat_to_tile(maxx, miny, zoom)   # SE corner

#     w_tiles = tx_max - tx_min + 1
#     h_tiles = ty_max - ty_min + 1
#     total   = w_tiles * h_tiles

#     if total > 256:
#         print(f"[WARN] AOI requires {total} tiles — too large, aborting.")
#         return None

#     print(f"[INFO] Downloading AOI raster: {w_tiles}×{h_tiles} tiles ({total} total)…")
#     canvas = Image.new("RGB", (w_tiles * TILE_SIZE, h_tiles * TILE_SIZE))

#     for ty in range(ty_min, ty_max + 1):
#         for tx in range(tx_min, tx_max + 1):
#             tile = fetch_tile(tx, ty, zoom)
#             if tile:
#                 canvas.paste(
#                     tile,
#                     ((tx - tx_min) * TILE_SIZE, (ty - ty_min) * TILE_SIZE)
#                 )

#     return canvas, tx_min, ty_min

# ══════════════════════════════════════════════════════════════════════════════
# 200 m × 200 m grid slicing
# ══════════════════════════════════════════════════════════════════════════════
def compute_grid_cells(
    minx: float, miny: float, maxx: float, maxy: float,
    cell_m: float = CELL_METERS
) -> list[tuple[float, float, float, float]]:
    """
    Divide the bbox into geographic cells of ~cell_m × cell_m metres.
    Returns list of (cell_minx, cell_miny, cell_maxx, cell_maxy).
    """
    center_lat = (miny + maxy) / 2
    dlat = meters_to_deg_lat(cell_m)
    dlng = meters_to_deg_lng(cell_m, center_lat)

    cells: list[tuple[float, float, float, float]] = []
    cy = miny
    while cy < maxy:
        cx = minx
        while cx < maxx:
            cells.append((cx, cy, min(cx + dlng, maxx), min(cy + dlat, maxy)))
            cx += dlng
        cy += dlat
    return cells


# ══════════════════════════════════════════════════════════════════════════════
# RF-DETR Roboflow workflow call
# ══════════════════════════════════════════════════════════════════════════════
# def run_rf_detr(img: Image.Image) -> list[dict]:
#     """
#     Send a PIL image to the Roboflow workflow.
#     Returns list of { class, confidence, x, y, width, height } (COCO center px).
#     """
#     if rf_client is None:
#         raise RuntimeError("Roboflow client not initialised.")

#     buf = BytesIO()
#     img.save(buf, format="PNG", quality=92)
#     b64      = base64.b64encode(buf.getvalue()).decode()
#     data_uri = f"data:image/jpeg;base64,{b64}"
#     from pathlib import Path
#     import time

#     # create debug folder
#     debug_dir = Path("debug_rf_inputs")
#     debug_dir.mkdir(exist_ok=True)

#     # save image before sending
#     debug_path = debug_dir / f"cell_{int(time.time() * 1000)}.png"
#     img.save(debug_path)

#     print(f"[DEBUG] Saved RF input image to: {debug_path}")
#     result = rf_client.run_workflow(
#         workspace_name=RF_WS,
#         workflow_id=RF_WF,
#         images={"image": data_uri},
#         use_cache=True,
#     )

#     # Normalise the SDK's variable return structure
#     predictions: list[dict] = []
#     if isinstance(result, list):
#         for item in result:
#             preds = (
#                 item.get("predictions")
#                 or item.get("output")
#                 or item.get("detections")
#                 or []
#             )
#             if isinstance(preds, dict):
#                 preds = preds.get("predictions", [])
#             predictions.extend(preds if isinstance(preds, list) else [])
#     elif isinstance(result, dict):
#         preds = result.get("predictions") or result.get("output") or []
#         if isinstance(preds, dict):
#             preds = preds.get("predictions", [])
#         predictions = preds if isinstance(preds, list) else []

#     return predictions

def run_rf_detr(img: Image.Image) -> list[dict]:
    """
    Send a resized PNG image to the Roboflow workflow.
    Adds extensive debug logging to diagnose failures.
    """

    if rf_client is None:
        raise RuntimeError("Roboflow client not initialised.")

    try:
        print("\n" + "=" * 80)
        print("[RF] Starting RF-DETR inference")

        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
            print("[RF] Converted image to RGB")

        # ─────────────────────────────────────────────────────────────
        # Resize image
        # ─────────────────────────────────────────────────────────────
        resized_img = img.resize((575, 575), Image.Resampling.LANCZOS)
        # ─────────────────────────────────────────────────────────────
        # Encode as PNG in memory
        # ─────────────────────────────────────────────────────────────
        buf = BytesIO()
        resized_img.save(buf, format="PNG")

        img_bytes = buf.getvalue()
        # Base64 encode
        b64 = base64.b64encode(img_bytes).decode()

        data_uri = f"data:image/png;base64,{b64}"

        # ─────────────────────────────────────────────────────────────
        # Run workflow
        # ─────────────────────────────────────────────────────────────

        result = rf_client.run_workflow(
            workspace_name=RF_WS,
            workflow_id=RF_WF,
            images={"image": data_uri},
            use_cache=False,
        )

        # ─────────────────────────────────────────────────────────────
        # Raw response debug
        # ─────────────────────────────────────────────────────────────
       
        try:
            pretty = json.dumps(result, indent=2)
        except Exception as e:
            print(f"[RF] Failed to serialize response: {e}")
            print(result)

        # ─────────────────────────────────────────────────────────────
        # Parse predictions
        # ─────────────────────────────────────────────────────────────
        predictions: list[dict] = []

        if isinstance(result, list):
            print(f"[RF] Response is LIST with {len(result)} items")

            for idx, item in enumerate(result):
            

                if not isinstance(item, dict):
                    continue
                # ─────────────────────────────────────────────
                # NEW: handle model_output
                # ─────────────────────────────────────────────
                if "model_output" in item:
                    model_output = item["model_output"]

                    preds = model_output.get("predictions", [])

                    print(f"[RF] model_output predictions: {len(preds)}")

                    if isinstance(preds, list):
                        predictions.extend(preds)

                    continue

                # fallback parsers
                preds = (
                    item.get("predictions")
                    or item.get("output")
                    or item.get("detections")
                    or item.get("outputs")
                    or []
                )

                if isinstance(preds, dict):
                    preds = (
                        preds.get("predictions")
                        or preds.get("detections")
                        or preds.get("output")
                        or []
                    )

                if isinstance(preds, list):
                    predictions.extend(preds)

        # ─────────────────────────────────────────────────────────────
        # Final debug
        # ─────────────────────────────────────────────────────────────
        print(f"[RF] FINAL DETECTIONS: {len(predictions)}")

        # if predictions:
        #     print("[RF] First prediction example:")
        #     print(json.dumps(predictions[0], indent=2))

        print("=" * 80 + "\n")

        return predictions

    except Exception as e:
        print("\n" + "!" * 80)
        print("[RF ERROR] Exception during RF inference")
        print(type(e).__name__)
        print(str(e))
        print("!" * 80 + "\n")
        raise

# ══════════════════════════════════════════════════════════════════════════════
# Core AOI → grid → RF-DETR pipeline
# ══════════════════════════════════════════════════════════════════════════════
# def run_rf_detr_on_aoi(
#     minx: float, miny: float, maxx: float, maxy: float,
#     zoom: int = ZOOM,
#     cell_m: float = CELL_METERS,
# ) -> list[dict]:
#     """
#     1. Download the full AOI raster at `zoom`.
#     2. Slice into `cell_m` × `cell_m` metre cells.
#     3. Run RF-DETR on each cell.
#     4. Convert every detected pixel-box back to WGS-84 geographic bbox.
#     5. Attach a numpy crop of the detected box for optional CLIP reclassification.

#     Returns list of detection dicts — internal _crop_arr field removed before
#     serialisation in the endpoint.
#     """
#     raster = download_aoi_raster(minx, miny, maxx, maxy, zoom)
#     if raster is None:
#         return []

#     canvas, tx_min, ty_min = raster
#     cells = compute_grid_cells(minx, miny, maxx, maxy, cell_m)

#     if len(cells) > MAX_CELLS:
#         print(f"[WARN] {len(cells)} cells requested — capping at {MAX_CELLS}.")
#         cells = cells[:MAX_CELLS]

#     print(f"[INFO] Running RF-DETR on {len(cells)} grid cells…")
#     all_detections: list[dict] = []

#     for cell_idx, (cminx, cminy, cmaxx, cmaxy) in enumerate(cells):
#         # Pixel coords of this cell's NW (top-left) and SE (bottom-right) corners
#         px0, py0 = geo_to_pixel(cminx, cmaxy, tx_min, ty_min, zoom)   # NW
#         px1, py1 = geo_to_pixel(cmaxx, cminy, tx_min, ty_min, zoom)   # SE

#         # Clamp to canvas size
#         px0 = max(0, px0);          py0 = max(0, py0)
#         px1 = min(canvas.width, px1); py1 = min(canvas.height, py1)

#         if px1 <= px0 or py1 <= py0:
#             continue

#         cell_img = canvas.crop((px0, py0, px1, py1))

#         try:
#             preds = run_rf_detr(cell_img)
#         except Exception as e:
#             print(f"[WARN] RF-DETR failed on cell {cell_idx}: {e}")
#             continue

#         for p in preds:
#             # Box centre in cell-local pixels
#             cx_cell = float(p.get("x", 0))
#             cy_cell = float(p.get("y", 0))
#             bw      = float(p.get("width",  0))
#             bh      = float(p.get("height", 0))

#             # Translate to canvas-global pixels
#             cx_global = px0 + cx_cell
#             cy_global = py0 + cy_cell

#             # Box corners in global canvas pixels
#             box_x0 = max(0,             int(cx_global - bw / 2))
#             box_y0 = max(0,             int(cy_global - bh / 2))
#             box_x1 = min(canvas.width,  int(cx_global + bw / 2))
#             box_y1 = min(canvas.height, int(cy_global + bh / 2))

#             # Convert corners to WGS-84
#             # NW corner → (minx, maxy);  SE corner → (maxx, miny)
#             geo_lng0, geo_lat1 = pixel_to_geo(box_x0, box_y0, tx_min, ty_min, zoom)
#             geo_lng1, geo_lat0 = pixel_to_geo(box_x1, box_y1, tx_min, ty_min, zoom)

#             # Crop the detected building out of the full canvas
#             if box_x1 > box_x0 and box_y1 > box_y0:
#                 crop_img = canvas.crop((box_x0, box_y0, box_x1, box_y1))
#             else:
#                 crop_img = cell_img
#             crop_arr = np.array(crop_img)

#             all_detections.append({
#                 "rf_class":      p.get("class", "unknown"),
#                 "rf_confidence": round(float(p.get("confidence", 0)), 4),
#                 "cell_idx":      cell_idx,
#                 # cell-local box (for debug reference)
#                 "x": cx_cell, "y": cy_cell, "width": bw, "height": bh,
#                 # geographic bbox of this detected box
#                 "geo_minx": geo_lng0, "geo_miny": geo_lat0,
#                 "geo_maxx": geo_lng1, "geo_maxy": geo_lat1,
#                 # numpy crop — stripped before JSON serialisation
#                 "_crop_arr": crop_arr,
#             })

#     print(f"[INFO] RF-DETR total detections: {len(all_detections)}")
#     return all_detections

def run_rf_detr_on_aoi(
    minx: float, miny: float, maxx: float, maxy: float,
    zoom: int = ZOOM,
    cell_m: float = CELL_METERS,
) -> list[dict]:
    """
    1. Download the full AOI raster at `zoom`.
    2. Slice into `cell_m` × `cell_m` metre cells.
    3. Run RF-DETR on each cell (normalized to 575x575).
    4. Convert every detected pixel-box back to WGS-84 geographic bbox.
    """
    raster = download_aoi_raster(minx, miny, maxx, maxy, zoom)
    if raster is None:
        return []

    canvas, tx_min, ty_min = raster
    cells = compute_grid_cells(minx, miny, maxx, maxy, cell_m)

    if len(cells) > MAX_CELLS:
        print(f"[WARN] {len(cells)} cells requested — capping at {MAX_CELLS}.")
        cells = cells[:MAX_CELLS]

    print(f"[INFO] Running RF-DETR on {len(cells)} grid cells…")
    all_detections: list[dict] = []

    # The model's internal inference resolution set in run_rf_detr()
    INFERENCE_SIZE = 575.0 

    for cell_idx, (cminx, cminy, cmaxx, cmaxy) in enumerate(cells):
        # Pixel coords of this cell's NW (top-left) and SE (bottom-right) corners
        px0, py0 = geo_to_pixel(cminx, cmaxy, tx_min, ty_min, zoom)   # NW
        px1, py1 = geo_to_pixel(cmaxx, cminy, tx_min, ty_min, zoom)   # SE

        # Clamp to canvas size
        px0 = max(0, px0);          py0 = max(0, py0)
        px1 = min(canvas.width, px1); py1 = min(canvas.height, py1)

        if px1 <= px0 or py1 <= py0:
            continue

        cell_img = canvas.crop((px0, py0, px1, py1))
        cell_w, cell_h = cell_img.size

        try:
            # run_rf_detr resizes internaly to 575x575
            preds = run_rf_detr(cell_img)
        except Exception as e:
            print(f"[WARN] RF-DETR failed on cell {cell_idx}: {e}")
            continue

        for p in preds:
            # 1. Normalize predictions relative to the 575x575 resize
            norm_x = float(p.get("x", 0)) / INFERENCE_SIZE
            norm_y = float(p.get("y", 0)) / INFERENCE_SIZE
            norm_w = float(p.get("width", 0)) / INFERENCE_SIZE
            norm_h = float(p.get("height", 0)) / INFERENCE_SIZE

            # 2. Convert Center (DETR format) to Top-Left (Pixel format)
            left_norm = norm_x - (norm_w / 2.0)
            top_norm  = norm_y - (norm_h / 2.0)

            # 3. Scale normalized ratios back to the actual cell pixel dimensions
            box_x0_local = left_norm * cell_w
            box_y0_local = top_norm * cell_h
            box_w_pixel  = norm_w * cell_w
            box_h_pixel  = norm_h * cell_h

            # 4. Translate local cell pixels to global canvas pixels
            box_x0_global = px0 + box_x0_local
            box_y0_global = py0 + box_y0_local
            box_x1_global = box_x0_global + box_w_pixel
            box_y1_global = box_y0_global + box_h_pixel

            # 5. Convert Canvas Pixels to WGS-84 Geographic Coordinates
            geo_lng0, geo_lat0 = pixel_to_geo(int(box_x0_global), int(box_y0_global), tx_min, ty_min, zoom)
            geo_lng1, geo_lat1 = pixel_to_geo(int(box_x1_global), int(box_y1_global), tx_min, ty_min, zoom)

            # Ensure proper min/max ordering for Shapely/GeoJSON [minx, miny, maxx, maxy]
            final_minx = min(geo_lng0, geo_lng1)
            final_maxx = max(geo_lng0, geo_lng1)
            final_miny = min(geo_lat0, geo_lat1)
            final_maxy = max(geo_lat0, geo_lat1)

            # Crop for optional CLIP re-classification
            if box_x1_global > box_x0_global and box_y1_global > box_y0_global:
                crop_img = canvas.crop((
                    int(max(0, box_x0_global)), 
                    int(max(0, box_y0_global)), 
                    int(min(canvas.width, box_x1_global)), 
                    int(min(canvas.height, box_y1_global))
                ))
            else:
                crop_img = cell_img
            
            crop_arr = np.array(crop_img)

            all_detections.append({
                "rf_class":      p.get("class", "unknown"),
                "rf_confidence": round(float(p.get("confidence", 0)), 4),
                "cell_idx":      cell_idx,
                "x": float(p.get("x")), "y": float(p.get("y")), # original model pixels
                "width": norm_w * cell_w, "height": norm_h * cell_h,
                "geo_minx": final_minx, "geo_miny": final_miny,
                "geo_maxx": final_maxx, "geo_maxy": final_maxy,
                "_crop_arr": crop_arr,
            })

    print(f"[INFO] RF-DETR total detections: {len(all_detections)}")
    return all_detections
# ══════════════════════════════════════════════════════════════════════════════
# Stitch + crop (clip_mlp mode only — per-polygon)
# ══════════════════════════════════════════════════════════════════════════════
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
        lat_r   = math.radians(lat)
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


# ══════════════════════════════════════════════════════════════════════════════
# Buildings (clip_mlp mode only)
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# App lifespan
# ══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_buildings()
    load_models()
    yield


app = FastAPI(title="Building Detection API — Multi-mode", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic models
# ══════════════════════════════════════════════════════════════════════════════
class ProcessRequest(BaseModel):
    area: dict
    zoom: int = ZOOM
    mode: Literal["clip_mlp", "rf_detr", "rf_detr_clip"] = "clip_mlp"
    cell_meters: int = CELL_METERS   # allow per-request cell size override


# ══════════════════════════════════════════════════════════════════════════════
# /api/process
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/process")
def process_area(req: ProcessRequest):
    mode      = req.mode
    area_geom = req.area.get("geometry") or req.area
    area_shp  = shape(area_geom)
    minx, miny, maxx, maxy = area_shp.bounds

    # ── Mode 1: clip_mlp ─────────────────────────────────────────────────────
    if mode == "clip_mlp":
        if not buildings:
            raise HTTPException(status_code=500, detail="No buildings loaded on server.")

        results = []
        for idx, feature in enumerate(buildings):
            if not feature.get("geometry"):
                continue
            props      = dict(feature.get("properties") or {})
            feat_shape = shape(feature["geometry"])
            if not area_shp.intersects(feat_shape):
                continue

            fminx, fminy, fmaxx, fmaxy     = feat_shape.bounds
            pminx, pminy, pmaxx, pmaxy     = padded_bbox(fminx, fminy, fmaxx, fmaxy)
            name                            = props.get("name") or props.get("id") or f"building_{idx}"

            pred_class, pred_conf = None, 0.0
            img = stitch_and_crop((pminx, pminy, pmaxx, pmaxy), req.zoom)
            if img is not None:
                result = classify_image_array(np.array(img))
                if result:
                    pred_class, pred_conf = result
                    print(f"[clip_mlp] {name} → {pred_class} ({pred_conf:.2%})")

            results.append({
                "type": "Feature",
                "properties": {
                    **props,
                    "_source_idx":  idx,
                    "_bbox_padded": [pminx, pminy, pmaxx, pmaxy],
                    "__bbox":       [fminx, fminy, fmaxx, fmaxy],
                    "_class":       pred_class,
                    "_confidence":  pred_conf,
                    "_mode":        "clip_mlp",
                    "_detections":  [],
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
                "mode":            "clip_mlp",
            },
        }

    # ── Modes 2 & 3: AOI raster → grid → RF-DETR ─────────────────────────────
    if not ROBOFLOW_AVAILABLE or rf_client is None:
        raise HTTPException(status_code=503, detail="Roboflow client not available.")

    raw_dets = run_rf_detr_on_aoi(
        minx, miny, maxx, maxy,
        zoom=req.zoom,
        cell_m=float(req.cell_meters),
    )

    results: list[dict] = []

    for det_idx, det in enumerate(raw_dets):
        # Pop the non-serialisable numpy crop
        crop_arr   = det.pop("_crop_arr", None)
        pred_class = det["rf_class"]
        pred_conf  = det["rf_confidence"]
        clip_class = None
        clip_conf  = None

        # Mode 3: reclassify the crop with CLIP+MLP
        if mode == "rf_detr_clip" and crop_arr is not None:
            clip_result = classify_image_array(crop_arr)
            if clip_result:
                clip_class, clip_conf = clip_result
                pred_class = clip_class
                pred_conf  = clip_conf

        geo_minx = det["geo_minx"]
        geo_miny = det["geo_miny"]
        geo_maxx = det["geo_maxx"]
        geo_maxy = det["geo_maxy"]

        results.append({
            "type": "Feature",
            "properties": {
                "_det_idx":           det_idx,
                "_cell_idx":          det.get("cell_idx"),
                "_class":             pred_class,
                "_confidence":        round(pred_conf, 4),
                "_mode":              mode,
                "_rf_class":          det["rf_class"],
                "_rf_confidence":     det["rf_confidence"],
                "_clip_class":        clip_class,
                "_clip_confidence":   round(clip_conf, 4) if clip_conf is not None else None,
                "_bbox":              [geo_minx, geo_miny, geo_maxx, geo_maxy],
                "_detections": [{
                    "class":           det["rf_class"],
                    "confidence":      det["rf_confidence"],
                    "clip_class":      clip_class,
                    "clip_confidence": round(clip_conf, 4) if clip_conf is not None else None,
                    "x":     det["x"],     "y":      det["y"],
                    "width": det["width"], "height": det["height"],
                }],
            },
            "bbox":     [geo_minx, geo_miny, geo_maxx, geo_maxy],
            "geometry": mapping(shapely_box(geo_minx, geo_miny, geo_maxx, geo_maxy)),
        })

    return {
        "type": "FeatureCollection",
        "features": results,
        "meta": {
            "total_buildings": len(buildings),
            "matched":         len(results),
            "classified":      len(results),
            "zoom_used":       req.zoom,
            "cell_meters":     req.cell_meters,
            "classes":         model_cache["classes"],
            "mode":            mode,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# /api/health
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/health")
def health():
    return {
        "status":             "ok",
        "buildings_loaded":   len(buildings),
        "classes":            model_cache["classes"],
        "clip_loaded":        model_cache["clip_model"] is not None,
        "mlp_loaded":         model_cache["mlp"] is not None,
        "roboflow_available": ROBOFLOW_AVAILABLE and rf_client is not None,
        "modes_available":    _available_modes(),
        "cell_meters":        CELL_METERS,
    }


def _available_modes() -> list[str]:
    modes = []
    if model_cache["clip_model"] is not None and model_cache["mlp"] is not None:
        modes.append("clip_mlp")
    if ROBOFLOW_AVAILABLE and rf_client is not None:
        modes.append("rf_detr")
        if model_cache["clip_model"] is not None:
            modes.append("rf_detr_clip")
    return modes


# ══════════════════════════════════════════════════════════════════════════════
# Static frontend
# ══════════════════════════════════════════════════════════════════════════════
_static = Path("static")
if _static.exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")