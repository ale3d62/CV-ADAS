"""
kitti_gt_rotation.py
--------------------
Calcula el pitch y yaw ground truth de la cámara con respecto a la carretera
para la secuencia KITTI 2011_09_26_drive_0015_sync.

Basado en el procedimiento desarrollado con Gemini:
  1. Cadena de rotación IMU -> Velodyne -> Cámara (incluyendo R_rect_00)
  2. Corrección de sistema de ejes (T_CFLU2Cam)
  3. Corrección de inclinación del vehículo frame a frame (pitch/roll IMU)
  4. Media de los valores sobre todos los frames de la secuencia

Resultado esperado: pitch ≈ -0.5390°, yaw ≈ 0.2119°

Ejecutar desde kitti_dataset/:
    python kitti_gt_rotation.py
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import glob


# ─── Rutas ────────────────────────────────────────────────────────────────────
CALIB_DIR = "./2011_09_26"
OXTS_DIR  = "./2011_09_26/2011_09_26_drive_0015_sync/oxts/data"


# ─── Cargar matrices de calibración ──────────────────────────────────────────

def load_rotation(filepath, key):
    """Extrae la matriz de rotación 3x3 de un archivo de calibración KITTI."""
    with open(filepath) as f:
        for line in f:
            if line.startswith(key + ":"):
                vals = list(map(float, line.strip().split()[1:10]))
                return np.array(vals).reshape(3, 3)
    raise ValueError(f"Clave '{key}' no encontrada en {filepath}")


# calib_imu_to_velo.txt  →  R_imu_to_velo
R_imu_to_velo = load_rotation(
    os.path.join(CALIB_DIR, "calib_imu_to_velo.txt"), "R")

# calib_velo_to_cam.txt  →  R_velo_to_cam
R_velo_to_cam = load_rotation(
    os.path.join(CALIB_DIR, "calib_velo_to_cam.txt"), "R")

# calib_cam_to_cam.txt  →  R_rect_00  (rectificación estéreo de la cámara 00)
# Es necesaria porque las imágenes que usa el sistema son las RECTIFICADAS.
R_rect_00 = load_rotation(
    os.path.join(CALIB_DIR, "calib_cam_to_cam.txt"), "R_rect_00")


# ─── Cadena de rotación IMU → Cámara (incluyendo rectificación) ───────────────
# Resultado: dada una dirección en frame IMU, la expresa en frame cámara.
# El orden es derecha a izquierda: primero imu2velo, luego velo2cam, luego rect.
R_imu_to_cam = R_rect_00 @ R_velo_to_cam @ R_imu_to_velo


# ─── Matriz de corrección de ejes ─────────────────────────────────────────────
# La IMU usa FLU: X=Frente, Y=Izquierda, Z=Arriba
# La cámara usa RDF: X=Derecha, Y=Abajo, Z=Frente
# T_CFLU2Cam "renombra" los ejes de la cámara para que coincidan con FLU,
# de modo que pitch/yaw tengan el mismo significado que para el vehículo.
#
# Correspondencias:
#   Nuevo X (Frente)     ← Cámara Z
#   Nuevo Y (Izquierda)  ← -Cámara X
#   Nuevo Z (Arriba)     ← -Cámara Y
T_CFLU2Cam = np.array([
    [ 0, -1,  0],   # Nuevo X ← -Cam_Y  (según Gemini; reasigna ejes)
    [ 0,  0, -1],   # Nuevo Y ← -Cam_Z
    [ 1,  0,  0]    # Nuevo Z ←  Cam_X
], dtype=float)

# Ejes de la cámara (en formato FLU) expresados respecto al frame de la IMU:
# R_CFLU2imu = R_imu2cam.T @ T_CFLU2Cam
#            = R_cam_to_imu @ T_CFLU2Cam
R_CFLU2imu = R_imu_to_cam.T @ T_CFLU2Cam


# ─── Calcular media de pitch/yaw sobre todos los frames ──────────────────────
oxts_files = sorted(glob.glob(os.path.join(OXTS_DIR, "*.txt")))
if not oxts_files:
    print(f"ERROR: no se encontraron archivos OXTS en {OXTS_DIR}")
    exit(1)

pitches, yaws, rolls = [], [], []

for fpath in oxts_files:
    with open(fpath) as f:
        vals = list(map(float, f.read().split()))
    roll_imu  = vals[3]   # rad (eje X en frame IMU)
    pitch_imu = vals[4]   # rad (eje Y en frame IMU)
    # vals[5] = yaw heading → NO se usa (relativo al Norte, no a la carretera)

    # Rotación del chasis respecto a la carretera (plano horizontal):
    # primero roll (eje X), luego pitch (eje Y), yaw = 0
    R_imu_to_road = R.from_euler('y', pitch_imu) * R.from_euler('x', roll_imu)

    # Rotación total: cámara (en ejes FLU) → IMU → carretera
    R_cam_to_road = R_imu_to_road * R.from_matrix(R_CFLU2imu)

    # Extraer ángulos de Euler en convención ZYX: [yaw, pitch, roll]
    yaw_f, pitch_f, roll_f = R_cam_to_road.as_euler('zyx')

    pitches.append(np.degrees(pitch_f))
    yaws.append(np.degrees(yaw_f))
    rolls.append(np.degrees(roll_f))


# ─── Resultados ───────────────────────────────────────────────────────────────
print("=" * 55)
print(f"  GT orientación cámara w.r.t. carretera")
print(f"  (secuencia 2011_09_26_drive_0015, {len(oxts_files)} frames)")
print("=" * 55)
print(f"  Pitch  media : {np.mean(pitches):.4f}°")
print(f"  Pitch  std   : {np.std(pitches):.4f}°")
print(f"  Yaw    media : {np.mean(yaws):.4f}°")
print(f"  Yaw    std   : {np.std(yaws):.4f}°")
print(f"  Roll   media : {np.mean(rolls):.4f}°")
print()
print("  Primeros 5 frames:")
print(f"  {'Frame':>6}  {'Pitch':>8}  {'Yaw':>8}  {'Roll':>8}")
for i in range(min(5, len(pitches))):
    print(f"  {i:>6}  {pitches[i]:>8.4f}  {yaws[i]:>8.4f}  {rolls[i]:>8.4f}")
