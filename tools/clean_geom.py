import json, sys, os, math
from pathlib import Path
from shapely.geometry import shape, Polygon, MultiPolygon, mapping

USAGE = """
Uso:
  python tools/clean_geom.py <entrada.geojson> [--out salida.geojson] [--report reporte.csv]

Qué hace:
  - Detecta features degenerados (área ~0, muy pocos vértices o forma ultra-alargada)
  - Los elimina y guarda un archivo limpio.
  - Deja respaldo: <archivo>.bak
Heurística (sin CRS proyectada, trabaja en grados):
  - area <= 0
  - num_vértices_exterior < 4
  - area / (bbox_area + 1e-12) < 5e-4  (súper alargado o casi línea)
  - (area / (length + 1e-12)) < 1e-6   (muy “línea”)
"""

def feature_suspect(geom):
    # Devuelve True si debe ser eliminado
    try:
        g = shape(geom)
        if g.is_empty or not g.is_valid:
            return True

        if g.geom_type not in ("Polygon","MultiPolygon"):
            # Si no es polígono, aquí no lo borramos (cámbialo si quieres)
            return False

        # Medidas simples (en grados)
        area   = g.area
        length = g.length

        # Número de vértices del anillo exterior (si es Polygon)
        def exterior_points_count(poly):
            try:
                return len(list(poly.exterior.coords))
            except Exception:
                return 0

        if g.geom_type == "Polygon":
            verts = exterior_points_count(g)
        else:  # MultiPolygon -> toma el más grande
            polys = list(g.geoms)
            if not polys:
                return True
            main = max(polys, key=lambda p: p.area)
            verts = exterior_points_count(main)

        # bbox area
        minx, miny, maxx, maxy = g.bounds
        bbox_area = max((maxx - minx), 0) * max((maxy - miny), 0)

        # Reglas
        if area <= 0:
            return True
        if verts < 4:
            return True
        if bbox_area > 0 and (area / (bbox_area + 1e-12)) < 5e-4:
            return True
        if (area / (length + 1e-12)) < 1e-6:
            return True

        return False
    except Exception:
        return True

def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"ERROR: No existe {in_path}")
        sys.exit(1)

    out_path = None
    report_path = None
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--out" and i+1 < len(args):
            out_path = Path(args[i+1])
        if a == "--report" and i+1 < len(args):
            report_path = Path(args[i+1])

    # Defaults
    if out_path is None:
        out_path = in_path.with_suffix(".cleaned.geojson")
    if report_path is None:
        report_path = in_path.with_suffix(".outliers.csv")

    data = json.load(open(in_path, "r", encoding="utf-8"))
    feats = data.get("features", [])

    suspects = []
    keep = []
    for idx, f in enumerate(feats):
        geom = f.get("geometry")
        if geom is None:
            suspects.append((idx, f))
            continue
        if feature_suspect(geom):
            suspects.append((idx, f))
        else:
            keep.append(f)

    # Resumen
    print(f"Archivo: {in_path}")
    print(f"Features totales: {len(feats)}")
    print(f"Marcados como degenerados: {len(suspects)}")
    print(f"Se conservarán: {len(keep)}")

    # Reporte CSV con info básica para rastrear
    try:
        import csv
        with open(report_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["index","SECCION","MUNICIPIO","DISTRITO_F","DISTRITO_L","TIPO"])
            for idx, f in suspects:
                p = f.get("properties", {}) or {}
                w.writerow([
                    idx,
                    str(p.get("SECCION","")),
                    str(p.get("MUNICIPIO","")),
                    str(p.get("DISTRITO_F","")),
                    str(p.get("DISTRITO_L","")),
                    str(p.get("TIPO","")),
                ])
        print(f"Reporte de degenerados: {report_path}")
    except Exception as e:
        print(f"(Aviso) No pude escribir reporte CSV: {e}")

    # Escribe limpio
    cleaned = {"type":"FeatureCollection","features": keep}
    # Respaldo
    bak = in_path.with_suffix(in_path.suffix + ".bak")
    try:
        if not bak.exists():
            os.replace(in_path, bak)
        else:
            # si ya existe backup, no sobrescribimos; solo leemos y escribimos salida
            pass
    except Exception as e:
        print(f"(Aviso) No pude renombrar respaldo: {e}")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(cleaned, fh, ensure_ascii=False)

    print(f"GeoJSON limpio escrito en: {out_path}")
    print("✔ Listo")

if __name__ == "__main__":
    main()
