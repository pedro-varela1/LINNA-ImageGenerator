"""
generate_synthetic_labels.py
============================
Gera labels sintéticos de crateras SEM imagens renderizadas.

Para cada amostra, sorteia aleatoriamente:
  - lat/lon do centro da "câmera" (cobertura total da Lua)
  - altitude em [ALT_MIN_KM, ALT_MAX_KM]

Em seguida, calcula quais crateras do catálogo caem dentro do FOV
quadrado usando projeção equiretangular local, e escreve os labels
no mesmo formato do build_random_labels.py:

  label/<id>.txt
    class_id  x_center_norm  y_center_norm  radius_norm  radius_px  radius_km

  json/<id>.json
    render_params.camera.{lat_deg, lon_deg, height_km}

Uso
---
    python generate_synthetic_labels.py \\
        --parquet craters_unified.parquet \\
        --out     synthetic_dataset/ \\
        --n       100000 \\
        --fov     120.0 \\
        --img-size 512 \\
        --min-diam 10.0 \\
        --max-diam 80.0 \\
        --alt-min  15.0 \\
        --alt-max  150.0 \\
        --workers  8 \\
        --seed     42
"""

from __future__ import annotations
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes físicas
# ---------------------------------------------------------------------------
MOON_RADIUS_KM: float = 1737.4
KM_PER_DEG_LAT: float = np.pi / 180.0 * MOON_RADIUS_KM   # ~30.34 km/°


# ---------------------------------------------------------------------------
# Projeção: lat/lon → pixel (x_norm, y_norm, radius_px, radius_norm)
# ---------------------------------------------------------------------------

def _fov_half_km(altitude_km: float, fov_deg: float) -> float:
    """Metade do lado do footprint quadrado em km."""
    return altitude_km * np.tan(np.radians(fov_deg / 2.0))


def project_craters(
    craters_lat: np.ndarray,   # (N,)
    craters_lon: np.ndarray,   # (N,)
    craters_diam_km: np.ndarray,  # (N,)
    cam_lat: float,
    cam_lon: float,
    altitude_km: float,
    fov_deg: float,
    img_size: int,
) -> list[dict]:
    """
    Projeta crateras visíveis no footprint da câmera.

    Usa projeção equiretangular local centrada em (cam_lat, cam_lon):
      x_km = (lon - cam_lon) * km_per_lon
      y_km = (lat - cam_lat) * KM_PER_DEG_LAT

    Retorna lista de dicts com campos do label.
    """
    half_km    = _fov_half_km(altitude_km, fov_deg)
    km_per_lon = KM_PER_DEG_LAT * np.cos(np.radians(cam_lat))
    if km_per_lon < 1e-6:
        km_per_lon = 1e-6   # evita divisão por zero nos polos

    # Diferença de longitude com wrap ±180°
    dlon = ((craters_lon - cam_lon + 180.0) % 360.0) - 180.0

    x_km = dlon * km_per_lon                          # east  (+direita)
    y_km = (craters_lat - cam_lat) * KM_PER_DEG_LAT  # north (+cima)

    # Filtro: centro da cratera dentro do footprint
    # (usa raio 0 para o filtro — crateras parcialmente fora são incluídas
    #  se o centro estiver dentro, consistente com build_random_labels.py)
    inside = (np.abs(x_km) <= half_km) & (np.abs(y_km) <= half_km)
    if not inside.any():
        return []

    x_km  = x_km[inside]
    y_km  = y_km[inside]
    diams = craters_diam_km[inside]

    # km/pixel
    kpp = (2.0 * half_km) / img_size

    # Coordenadas normalizadas [0, 1]
    # x_km=0 → pixel central → 0.5
    x_norm = 0.5 + x_km  / (2.0 * half_km)   # east  = direita = +x_norm
    y_norm = 0.5 - y_km  / (2.0 * half_km)   # north = cima   = -y_norm (origem top-left)

    radius_km   = diams / 2.0
    radius_px   = np.maximum(1, np.round(radius_km / kpp).astype(int))
    radius_norm = radius_px / float(img_size)

    labels = []
    for i in range(len(x_km)):
        labels.append({
            "x_center_norm": float(x_norm[i]),
            "y_center_norm": float(y_norm[i]),
            "radius_norm":   float(radius_norm[i]),
            "radius_px":     int(radius_px[i]),
            "radius_km":     float(radius_km[i]),
        })
    return labels


# ---------------------------------------------------------------------------
# Worker (uma amostra)
# Os arrays de crateras ficam no processo worker via initializer,
# evitando serialização cara a cada task.
# ---------------------------------------------------------------------------

# Variáveis globais do worker (preenchidas pelo initializer)
_W_LATS  = None
_W_LONS  = None
_W_DIAMS = None


def _worker_init(lats, lons, diams):
    global _W_LATS, _W_LONS, _W_DIAMS
    _W_LATS  = lats
    _W_LONS  = lons
    _W_DIAMS = diams


def _worker(task: dict) -> dict:
    """
    Gera label e json para uma amostra.
    task keys: sample_id, cam_lat, cam_lon, altitude_km,
               fov_deg, img_size, class_id, label_dir, json_dir
    """
    sid         = task["sample_id"]
    cam_lat     = task["cam_lat"]
    cam_lon     = task["cam_lon"]
    altitude_km = task["altitude_km"]
    fov_deg     = task["fov_deg"]
    img_size    = task["img_size"]
    class_id    = task["class_id"]

    labels = project_craters(
        _W_LATS, _W_LONS, _W_DIAMS,
        cam_lat, cam_lon, altitude_km,
        fov_deg, img_size,
    )

    # Escreve label .txt
    label_path = Path(task["label_dir"]) / f"{sid}.txt"
    with open(label_path, "w") as f:
        for obj in labels:
            f.write(
                f"{class_id} "
                f"{obj['x_center_norm']:.8f} {obj['y_center_norm']:.8f} "
                f"{obj['radius_norm']:.8f} "
                f"{obj['radius_px']} {obj['radius_km']:.6f}\n"
            )

    # Escreve json de metadados
    meta = {
        "render_params": {
            "camera": {
                "lat_deg":   cam_lat,
                "lon_deg":   cam_lon,
                "height_km": altitude_km,
            }
        },
        "fov_deg":    fov_deg,
        "img_size":   img_size,
        "n_craters":  len(labels),
    }
    json_path = Path(task["json_dir"]) / f"{sid}.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {"sid": sid, "n_craters": len(labels)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(args):
    out_dir   = Path(args.out)
    label_dir = out_dir / "label"
    json_dir  = out_dir / "json"
    label_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True,  exist_ok=True)

    # Carrega parquet e filtra diâmetros
    print(f"[1/3] Carregando parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    df.columns = [c.lower() for c in df.columns]
    df["x_coord"] = df["x_coord"].where(df["x_coord"] <= 180.0, df["x_coord"] - 360.0)

    df = df[(df.diam_km >= args.min_diam) & (df.diam_km <= args.max_diam)].reset_index(drop=True)
    print(f"  {len(df):,} crateras após filtro [{args.min_diam}, {args.max_diam}] km")

    craters_lat  = df.y_coord.values.astype(np.float32)
    craters_lon  = df.x_coord.values.astype(np.float32)
    craters_diam = df.diam_km.values.astype(np.float32)

    # Sorteia poses aleatórias
    print(f"[2/3] Sorteando {args.n} poses aleatórias (seed={args.seed}) …")
    rng = np.random.default_rng(args.seed)

    # Lat uniforme em seno para cobertura isotrópica da esfera
    sin_lat  = rng.uniform(-1.0, 1.0,   args.n)
    cam_lats = np.degrees(np.arcsin(sin_lat)).astype(np.float32)
    cam_lons = rng.uniform(-180.0, 180.0, args.n).astype(np.float32)
    altitudes = rng.uniform(args.alt_min, args.alt_max, args.n).astype(np.float32)

    # Padding do ID com zeros
    n_digits = len(str(args.n - 1))

    tasks = [
        {
            "sample_id":   str(i).zfill(n_digits),
            "cam_lat":     float(cam_lats[i]),
            "cam_lon":     float(cam_lons[i]),
            "altitude_km": float(altitudes[i]),
            "fov_deg":     args.fov,
            "img_size":    args.img_size,
            "class_id":    args.class_id,
            "label_dir":   str(label_dir),
            "json_dir":    str(json_dir),
        }
        for i in range(args.n)
    ]

    # Processamento paralelo — arrays de crateras passados via initializer
    # (evita serializar os arrays grandes a cada task)
    print(f"[3/3] Gerando labels ({args.workers} workers) …")
    total_craters = 0
    empty_count   = 0
    done          = 0

    init_args = (craters_lat, craters_lon, craters_diam)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=init_args,
    ) as pool:
        futures = {pool.submit(_worker, t): t["sample_id"] for t in tasks}
        for fut in as_completed(futures):
            res           = fut.result()
            total_craters += res["n_craters"]
            done          += 1
            if res["n_craters"] == 0:
                empty_count += 1
            if done % max(1, args.n // 20) == 0 or done == args.n:
                pct = 100 * done / args.n
                print(f"  {done}/{args.n} ({pct:.0f}%)  "
                      f"crateras totais: {total_craters:,}  "
                      f"amostras vazias: {empty_count}")

    print()
    print("=" * 55)
    print(f"  Amostras geradas   : {args.n - empty_count:,}  (com ≥1 cratera)")
    print(f"  Amostras vazias    : {empty_count:,}  (sem crateras no FOV)")
    print(f"  Total de labels    : {total_craters:,}")
    print(f"  Média por amostra  : {total_craters / max(1, args.n - empty_count):.1f}")
    print(f"  Saída              : {out_dir}/")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Gera labels sintéticos de crateras sem imagens renderizadas."
    )
    p.add_argument("--parquet",   required=True,
                   help="Caminho para o catálogo de crateras (.parquet)")
    p.add_argument("--out",       required=True,
                   help="Pasta de saída (serão criadas label/ e json/)")
    p.add_argument("--n",         type=int, required=True,
                   help="Número de amostras a gerar")
    p.add_argument("--fov",       type=float, default=120.0,
                   help="FOV quadrado da câmera em graus (default: 120.0)")
    p.add_argument("--img-size",  type=int,   default=512,
                   help="Tamanho da imagem virtual em pixels (default: 512)")
    p.add_argument("--min-diam",  type=float, default=10.0,
                   help="Diâmetro mínimo de cratera em km (default: 10.0)")
    p.add_argument("--max-diam",  type=float, default=80.0,
                   help="Diâmetro máximo de cratera em km (default: 80.0)")
    p.add_argument("--alt-min",   type=float, default=15.0,
                   help="Altitude mínima em km (default: 15.0)")
    p.add_argument("--alt-max",   type=float, default=150.0,
                   help="Altitude máxima em km (default: 150.0)")
    p.add_argument("--class-id",  type=int,   default=0,
                   help="class_id escrito nos labels (default: 0)")
    p.add_argument("--workers",   type=int,   default=os.cpu_count() or 4,
                   help="Número de processos paralelos")
    p.add_argument("--seed",      type=int,   default=42,
                   help="Semente aleatória (default: 42)")
    args = p.parse_args()
    generate(args)