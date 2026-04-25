import os
from pathlib import Path
import traceback
import json
import time
import sys
import threading

import numpy as np
import cv2
import open3d as o3d
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import trimesh

# Pipeline settings
DEPTH_CHECKPOINT = os.environ.get("DEPTH_CHECKPOINT", "LiheYoung/depth-anything-large-hf")
USE_GPU = int(os.environ.get("USE_GPU", "1"))
POISSON_DEPTH = int(os.environ.get("POISSON_DEPTH", "9"))
OUTLIER_NEIGHBORS = int(os.environ.get("OUTLIER_NEIGHBORS", "15"))
OUTLIER_STD_RATIO = float(os.environ.get("OUTLIER_STD_RATIO", "1.0"))
ORTHO_SCALE_FACTOR = float(os.environ.get("ORTHO_SCALE_FACTOR", "255"))
INFERENCE_RESIZE = int(os.environ.get("INFERENCE_RESIZE", "0"))
RESULT_PREFIX = os.environ.get("RESULT_PREFIX", "")

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
except Exception:
    pass

_model = None
_processor = None
_device = "cpu"

# In-memory dictionary for job status
_JOB_STATUS = {}
_STATUS_LOCK = threading.Lock()

def get_status(job_id: str):
    with _STATUS_LOCK:
        return _JOB_STATUS.get(job_id)

def set_status(job_id: str, state: str, detail: str = "", result: str = ""):
    with _STATUS_LOCK:
        _JOB_STATUS[job_id] = {"state": state, "detail": detail, "result": result}

def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()

def load_model():
    global _model, _processor, _device
    if _model is None:
        log(f"Loading model: {DEPTH_CHECKPOINT}")
        try:
            _processor = AutoImageProcessor.from_pretrained(DEPTH_CHECKPOINT)
            _model = AutoModelForDepthEstimation.from_pretrained(DEPTH_CHECKPOINT)
        except Exception as e:
            # Fallback to a valid HF model 
            fallback = "LiheYoung/depth-anything-large-hf"
            log(f"Failed to load {DEPTH_CHECKPOINT}, falling back to {fallback}. Error: {e}")
            _processor = AutoImageProcessor.from_pretrained(fallback)
            _model = AutoModelForDepthEstimation.from_pretrained(fallback)
            
        if USE_GPU and torch.cuda.is_available():
            _device = "cuda"
            _model = _model.to("cuda")
        else:
            _device = "cpu"
        _model.eval()
    return _model, _processor, _device

def normalize_depth_uint8(depth_np: np.ndarray) -> np.ndarray:
    m = np.max(depth_np)
    if m <= 0:
        return np.zeros_like(depth_np, dtype=np.uint8)
    return (depth_np * 255.0 / m).astype("uint8")

def build_orthographic_point_cloud(depth_u8: np.ndarray, color_rgb: np.ndarray) -> o3d.geometry.PointCloud:
    depth_map = depth_u8.astype(np.float32)
    h, w = depth_map.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    z = (depth_map / ORTHO_SCALE_FACTOR) * (h / 2.0)
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    mask = points[:, 2] != 0
    points = points[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = color_rgb.reshape(-1, 3)[mask] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def process_image_direct(image_path: str, result_dir: str, job_id: str):
    start = time.time()
    try:
        set_status(job_id, "RUNNING", "Loading model")
        model, processor, device = load_model()
        log(f"[{job_id}] Model loaded on {device}")

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise RuntimeError("Failed to read image")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        if INFERENCE_RESIZE and INFERENCE_RESIZE > 0:
            scale = INFERENCE_RESIZE / max(orig_h, orig_w)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img_proc = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_proc = img_rgb

        set_status(job_id, "RUNNING", "Running depth inference")
        depth_inputs = processor(images=img_proc, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**depth_inputs)
        depth = outputs.predicted_depth.squeeze().detach().cpu().numpy()

        dh, dw = depth.shape
        color_resized = cv2.resize(img_proc, (dw, dh), interpolation=cv2.INTER_LINEAR)
        depth_u8 = normalize_depth_uint8(depth)

        set_status(job_id, "RUNNING", "Building orthographic point cloud")
        pcd = build_orthographic_point_cloud(depth_u8, color_resized)

        try:
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
            pcd = pcd.select_by_index(ind)
        except Exception as e:
            log(f"[{job_id}] Outlier removal warning: {e}")

        if len(pcd.points) >= 10:
            try:
                pcd.estimate_normals()
                pcd.orient_normals_to_align_with_direction()
            except Exception as e:
                log(f"[{job_id}] Normal estimation warning: {e}")

        num_pts = np.asarray(pcd.points).shape[0]
        log(f"[{job_id}] Point cloud size after cleanup: {num_pts}")
        if num_pts == 0:
            raise RuntimeError("Empty point cloud after cleanup")

        set_status(job_id, "RUNNING", f"Poisson reconstruction depth={POISSON_DEPTH}")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=POISSON_DEPTH
        )

        try:
            mesh.compute_vertex_normals()
        except Exception:
            pass
        mesh.compute_triangle_normals()

        num_vertices = np.asarray(mesh.vertices).shape[0]
        num_tris = np.asarray(mesh.triangles).shape[0]
        log(f"[{job_id}] Mesh stats vertices={num_vertices} triangles={num_tris}")
        if num_tris == 0:
            raise RuntimeError("Poisson produced empty mesh")

        Path(result_dir).mkdir(parents=True, exist_ok=True)
        stl_path = Path(result_dir) / f"{RESULT_PREFIX}{job_id}.stl"

        set_status(job_id, "RUNNING", "Exporting STL")
        tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles), process=True)
        tm.export(str(stl_path), file_type="stl")

        total = time.time() - start
        set_status(job_id, "SUCCESS", f"Done in {total:.2f}s", str(stl_path))
        log(f"[{job_id}] SUCCESS total={total:.2f}s STL={stl_path}")
        return {
            "status": "success",
            "stl": str(stl_path),
            "mesh_stats": {"vertices": int(num_vertices), "triangles": int(num_tris)}
        }
    except Exception as e:
        traceback.print_exc()
        set_status(job_id, "FAILURE", str(e))
        log(f"[{job_id}] FAILURE: {e}")
        return {"status": "error", "error": str(e)}