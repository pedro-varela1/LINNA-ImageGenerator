import os
from PIL import Image

# Maps WAC tile suffix → (lat_min, lat_max, lon_min, lon_max)
WAC_TILES = {
    "E350N0450": (  0,  70,   0,  90),
    "E350S0450": (-70,   0,   0,  90),
    "E350N1350": (  0,  70,  90, 180),
    "E350S1350": (-70,   0,  90, 180),
    "E350N2250": (  0,  70, 180, 270),
    "E350S2250": (-70,   0, 180, 270),
    "E350N3150": (  0,  70, 270, 360),
    "E350S3150": (-70,   0, 270, 360),
}


def find_wac_tiles(lat_min, lat_max, lon_min, lon_max, wac_dir, wac_ext):
    """Return every WAC tile that overlaps the patch with its resolved file path."""
    needed = []
    for suffix, (tlat_min, tlat_max, tlon_min, tlon_max) in WAC_TILES.items():
        if tlat_max <= lat_min or tlat_min >= lat_max:
            continue
        if tlon_max <= lon_min or tlon_min >= lon_max:
            continue
        found = None
        for ext in (wac_ext, wac_ext.upper(), wac_ext.lower()):
            fpath = os.path.join(wac_dir, f"WAC_HAPKE_3BAND_{suffix}{ext}")
            if os.path.isfile(fpath):
                found = fpath
                break
        if found is None:
            raise FileNotFoundError(
                f"WAC Hapke tile not found: WAC_HAPKE_3BAND_{suffix}{wac_ext}\n"
                f"  Searched in: {wac_dir}"
            )
        needed.append((suffix, tlat_min, tlat_max, tlon_min, tlon_max, found))

    if not needed:
        raise ValueError(
            f"No WAC tiles cover lat={lat_min}..{lat_max}, lon={lon_min}..{lon_max}"
        )
    print(f"[WAC] Using {len(needed)} tile(s): {[t[0] for t in needed]}")
    return needed


def _crop_tile_to_patch(tile_path, tile_lat_min, tile_lat_max,
                         tile_lon_min, tile_lon_max,
                         patch_lat_min, patch_lat_max,
                         patch_lon_min, patch_lon_max):
    """
    Open one WAC tile, crop to the geographic intersection with the patch.
    Returns (PIL.Image RGB, (clat_min, clat_max, clon_min, clon_max)).
    """
    img = Image.open(tile_path)
    w, h = img.size

    lon_span = tile_lon_max - tile_lon_min
    lat_span = tile_lat_max - tile_lat_min

    clat_min = max(patch_lat_min, tile_lat_min)
    clat_max = min(patch_lat_max, tile_lat_max)
    clon_min = max(patch_lon_min, tile_lon_min)
    clon_max = min(patch_lon_max, tile_lon_max)

    # PIL row 0 = north edge (tile_lat_max)
    px_left  = int((clon_min - tile_lon_min) / lon_span * w)
    px_right = int((clon_max - tile_lon_min) / lon_span * w)
    px_top   = int((tile_lat_max - clat_max) / lat_span * h)
    px_bot   = int((tile_lat_max - clat_min) / lat_span * h)

    px_left  = max(0, min(px_left,  w - 1))
    px_right = max(px_left + 1, min(px_right, w))
    px_top   = max(0, min(px_top,   h - 1))
    px_bot   = max(px_top + 1, min(px_bot,   h))

    crop = img.crop((px_left, px_top, px_right, px_bot)).convert("RGB")
    return crop, (clat_min, clat_max, clon_min, clon_max)


def build_color_patch(lat_min, lat_max, lon_min, lon_max,
                       out_path, wac_dir, wac_ext, size=1024):
    """
    Stitch all required WAC tiles into a single (size × size) RGB PNG.
    Handles patches that cross tile boundaries.
    """
    tiles    = find_wac_tiles(lat_min, lat_max, lon_min, lon_max, wac_dir, wac_ext)
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    canvas   = Image.new("RGB", (size, size), color=(100, 100, 100))

    for (suffix, tlat_min, tlat_max, tlon_min, tlon_max, fpath) in tiles:
        crop, (clat_min, clat_max, clon_min, clon_max) = _crop_tile_to_patch(
            fpath, tlat_min, tlat_max, tlon_min, tlon_max,
            lat_min, lat_max, lon_min, lon_max,
        )
        cx = int((clon_min - lon_min) / lon_span * size)
        cy = int((lat_max  - clat_max) / lat_span * size)
        cw = max(1, int((clon_max - clon_min) / lon_span * size))
        ch = max(1, int((clat_max - clat_min) / lat_span * size))

        canvas.paste(crop.resize((cw, ch), resample=Image.LANCZOS), (cx, cy))
        print(f"[WAC] Pasted tile {suffix} at canvas offset ({cx}, {cy}), size ({cw}x{ch})")

    canvas.save(out_path)
    print(f"[WAC] Color patch saved → {out_path}")
    return out_path
