
"""
Funciones numéricas para el Simulador Computacional de Trazado Vial.

Este módulo no contiene ninguna dependencia de Streamlit ni lógica de interfaz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Delaunay, cKDTree


@dataclass
class EjeVial:
    """Contenedor del eje vial calculado en la Fase 7."""
    x: np.ndarray
    y: np.ndarray
    progresiva: np.ndarray
    terreno_z: np.ndarray
    rasante_z: np.ndarray


def generar_terreno_sintetico(
    n_puntos: int = 450,
    ancho_x: float = 500.0,
    ancho_y: float = 400.0,
    semilla: int = 42,
) -> np.ndarray:
    """
    Fase 1: genera un levantamiento XYZ sintético con relieve ondulado y ruido.
    Devuelve una matriz de forma (n, 3) con columnas X, Y, Z.
    """
    rng = np.random.default_rng(semilla)
    x = rng.uniform(0, ancho_x, n_puntos)
    y = rng.uniform(0, ancho_y, n_puntos)
    z = (
        120.0
        + 0.025 * x
        - 0.015 * y
        + 9.0 * np.sin(x / 80.0)
        + 6.0 * np.cos(y / 65.0)
        + 10.0 * np.exp(-((x - 300.0) ** 2 + (y - 220.0) ** 2) / 18000.0)
        + rng.normal(0, 0.8, n_puntos)
    )
    return np.column_stack((x, y, z))


def construir_tin(xyz: np.ndarray, lado_maximo: float) -> tuple[Delaunay, np.ndarray]:
    """
    Fase 3: crea una triangulación Delaunay y filtra los triángulos cuyo lado
    más largo supera el valor indicado.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3:
        raise ValueError("Se requieren al menos tres puntos XYZ válidos.")
    tri = Delaunay(xyz[:, :2])
    simplices = tri.simplices
    p = xyz[simplices, :2]
    d01 = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
    d12 = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
    d20 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    validos = np.maximum.reduce([d01, d12, d20]) <= float(lado_maximo)
    return tri, validos


def aristas_tin(simplices: np.ndarray, validos: np.ndarray) -> np.ndarray:
    """
    Fase 3: obtiene las aristas únicas de los triángulos TIN válidos.
    """
    seleccion = simplices[validos]
    aristas = set()
    for a, b, c in seleccion:
        for i, j in ((a, b), (b, c), (c, a)):
            aristas.add(tuple(sorted((int(i), int(j)))))
    return np.asarray(sorted(aristas), dtype=int)


def crear_mde(
    xyz: np.ndarray,
    tri: Delaunay,
    triangulos_validos: np.ndarray,
    resolucion: int = 100,
    metodo: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fase 4: interpola un MDE regular y enmascara con NaN las celdas fuera de
    triángulos válidos para evitar inventar superficie en huecos reales.
    """
    resolucion = max(25, int(resolucion))
    gx = np.linspace(float(xyz[:, 0].min()), float(xyz[:, 0].max()), resolucion)
    gy = np.linspace(float(xyz[:, 1].min()), float(xyz[:, 1].max()), resolucion)
    xx, yy = np.meshgrid(gx, gy)
    zz = griddata(xyz[:, :2], xyz[:, 2], (xx, yy), method=metodo)

    consultas = np.column_stack((xx.ravel(), yy.ravel()))
    simplex_ids = tri.find_simplex(consultas)
    mascara = np.zeros(len(consultas), dtype=bool)
    dentro = simplex_ids >= 0
    mascara[dentro] = triangulos_validos[simplex_ids[dentro]]
    mascara = mascara.reshape(xx.shape)
    zz[~mascara] = np.nan
    return gx, gy, zz, mascara


def volumen_bloque_mde(gx: np.ndarray, gy: np.ndarray, zz: np.ndarray, cota_base: float) -> float:
    """
    Fase 5: aproxima el volumen entre el MDE y una cota base mediante integración
    por celdas regulares.
    """
    if len(gx) < 2 or len(gy) < 2:
        return 0.0
    dx = float(np.mean(np.diff(gx)))
    dy = float(np.mean(np.diff(gy)))
    alturas = np.clip(zz - cota_base, 0.0, None)
    return float(np.nansum(alturas) * dx * dy)


def construir_maqueta_malla(
    gx: np.ndarray,
    gy: np.ndarray,
    zz: np.ndarray,
    cota_base: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fase 5: crea vértices y caras triangulares para una maqueta sólida con tapa
    superior, base y paredes alrededor del contorno real de celdas válidas.
    """
    ny, nx = zz.shape
    vertices: list[list[float]] = []
    caras: list[list[int]] = []
    indice_sup: dict[tuple[int, int], int] = {}
    indice_base: dict[tuple[int, int], int] = {}

    def v_sup(i: int, j: int) -> int:
        key = (i, j)
        if key not in indice_sup:
            indice_sup[key] = len(vertices)
            vertices.append([float(gx[j]), float(gy[i]), float(zz[i, j])])
        return indice_sup[key]

    def v_base(i: int, j: int) -> int:
        key = (i, j)
        if key not in indice_base:
            indice_base[key] = len(vertices)
            vertices.append([float(gx[j]), float(gy[i]), float(cota_base)])
        return indice_base[key]

    valid = np.isfinite(zz)

    # Tapas superior e inferior.
    for i in range(ny - 1):
        for j in range(nx - 1):
            corners = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
            if all(valid[a, b] for a, b in corners):
                s = [v_sup(*c) for c in corners]
                b = [v_base(*c) for c in corners]
                caras.extend([[s[0], s[1], s[2]], [s[0], s[2], s[3]]])
                caras.extend([[b[0], b[2], b[1]], [b[0], b[3], b[2]]])

    # Paredes por cada lado de celda válida que limita con vacío o borde.
    for i in range(ny):
        for j in range(nx):
            if not valid[i, j]:
                continue
            vecinos = [
                ((i, j), (i, j + 1), i == 0 or (i - 1 < ny and not valid[i - 1, j]) if j + 1 < nx else False),
                ((i, j + 1), (i + 1, j + 1), j + 1 == nx - 1 or not valid[i, j + 1] if i + 1 < ny and j + 1 < nx else False),
                ((i + 1, j + 1), (i + 1, j), i + 1 == ny - 1 or not valid[i + 1, j] if i + 1 < ny and j + 1 < nx else False),
                ((i + 1, j), (i, j), j == 0 or not valid[i, j - 1] if i + 1 < ny else False),
            ]
            for a, b, frontera in vecinos:
                ia, ja = a
                ib, jb = b
                if not frontera or ia >= ny or ib >= ny or ja >= nx or jb >= nx:
                    continue
                if not (valid[ia, ja] and valid[ib, jb]):
                    continue
                sa, sb = v_sup(ia, ja), v_sup(ib, jb)
                ba, bb = v_base(ia, ja), v_base(ib, jb)
                caras.extend([[sa, sb, bb], [sa, bb, ba]])

    return np.asarray(vertices, dtype=float), np.asarray(caras, dtype=int)


def interpolar_eje(
    controles: Sequence[Sequence[float]],
    espaciamiento: float = 5.0,
    sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fase 7: densifica puntos de control por longitud de arco y suaviza X/Y con
    filtro gaussiano, preservando exactamente los extremos.
    """
    p = np.asarray(controles, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        raise ValueError("Defina al menos dos puntos de control.")
    dist = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s_control = np.concatenate(([0.0], np.cumsum(dist)))
    if s_control[-1] <= 0:
        raise ValueError("Los puntos de control no pueden coincidir.")
    n = max(2, int(math.ceil(s_control[-1] / max(espaciamiento, 0.5))) + 1)
    s = np.linspace(0.0, s_control[-1], n)
    x = np.interp(s, s_control, p[:, 0])
    y = np.interp(s, s_control, p[:, 1])
    if sigma > 0:
        x = gaussian_filter1d(x, sigma=float(sigma), mode="nearest")
        y = gaussian_filter1d(y, sigma=float(sigma), mode="nearest")
        x[0], y[0] = p[0]
        x[-1], y[-1] = p[-1]
    progresiva = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    return x, y, progresiva


def interpolar_cotas_eje(
    xyz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Fase 7: interpola la cota del terreno natural bajo cada estación. Los NaN de
    la interpolación lineal se sustituyen mediante vecino más cercano.
    """
    consultas = np.column_stack((x, y))
    z = griddata(xyz[:, :2], xyz[:, 2], consultas, method="linear")
    faltan = ~np.isfinite(z)
    if np.any(faltan):
        arbol = cKDTree(xyz[:, :2])
        _, idx = arbol.query(consultas[faltan])
        z[faltan] = xyz[idx, 2]
    return z


def calcular_rasante(
    progresiva: np.ndarray,
    cota_inicial: float,
    pendientes_por_tramo: Sequence[float],
    longitud_tramo: float = 100.0,
) -> np.ndarray:
    """
    Fase 7: calcula la rasante tramo por tramo, usando pendientes porcentuales
    constantes en segmentos de longitud configurable.
    """
    pendientes = np.asarray(pendientes_por_tramo, dtype=float)
    if pendientes.size == 0:
        pendientes = np.array([0.0])
    z = np.empty_like(progresiva, dtype=float)
    z[0] = float(cota_inicial)
    for i in range(1, len(progresiva)):
        ds = progresiva[i] - progresiva[i - 1]
        mid = 0.5 * (progresiva[i] + progresiva[i - 1])
        tramo = min(int(mid // longitud_tramo), len(pendientes) - 1)
        z[i] = z[i - 1] + ds * pendientes[tramo] / 100.0
    return z


def areas_movimiento_tierras(
    terreno_z: np.ndarray,
    rasante_z: np.ndarray,
    ancho: float,
    talud_corte: float,
    talud_relleno: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fase 8: calcula áreas de corte y relleno con sección trapezoidal.
    """
    diferencia = terreno_z - rasante_z
    h_c = np.clip(diferencia, 0.0, None)
    h_r = np.clip(-diferencia, 0.0, None)
    area_c = (ancho + talud_corte * h_c) * h_c
    area_r = (ancho + talud_relleno * h_r) * h_r
    return area_c, area_r


def integrar_volumenes(
    progresiva: np.ndarray,
    area_corte: np.ndarray,
    area_relleno: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fase 8: integra volúmenes acumulados por áreas extremas promedio.
    """
    ds = np.diff(progresiva)
    vc_seg = 0.5 * (area_corte[:-1] + area_corte[1:]) * ds
    vr_seg = 0.5 * (area_relleno[:-1] + area_relleno[1:]) * ds
    vc = np.concatenate(([0.0], np.cumsum(vc_seg)))
    vr = np.concatenate(([0.0], np.cumsum(vr_seg)))
    return vc, vr, vc + vr


def pk_por_presupuesto(progresiva: np.ndarray, volumen_total: np.ndarray, presupuesto: float) -> float:
    """
    Fase 8: localiza por interpolación la progresiva donde el movimiento total
    alcanza el presupuesto disponible.
    """
    if presupuesto <= 0:
        return 0.0
    if volumen_total[-1] <= presupuesto:
        return float(progresiva[-1])
    idx = int(np.searchsorted(volumen_total, presupuesto))
    x0, x1 = volumen_total[idx - 1], volumen_total[idx]
    s0, s1 = progresiva[idx - 1], progresiva[idx]
    if x1 == x0:
        return float(s1)
    return float(s0 + (presupuesto - x0) * (s1 - s0) / (x1 - x0))


def normales_planta(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fase 8: calcula vectores normales unitarios en planta para formar la calzada.
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    norma = np.hypot(dx, dy)
    norma[norma == 0] = 1.0
    return -dy / norma, dx / norma


def geometria_carretera(
    eje: EjeVial,
    ancho: float,
    talud_corte: float,
    talud_relleno: float,
    pk_limite: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Fase 8: crea bordes de calzada y extremos simplificados de taludes de corte y
    relleno para construir mallas Plotly.
    """
    mask = eje.progresiva <= (pk_limite if pk_limite is not None else eje.progresiva[-1])
    x, y = eje.x[mask], eje.y[mask]
    z, tn = eje.rasante_z[mask], eje.terreno_z[mask]
    nx, ny = normales_planta(x, y)
    mitad = ancho / 2.0
    izq = np.column_stack((x + nx * mitad, y + ny * mitad, z))
    der = np.column_stack((x - nx * mitad, y - ny * mitad, z))

    delta = tn - z
    extension = np.where(delta >= 0, talud_corte * delta, talud_relleno * (-delta))
    ext_izq = np.column_stack((izq[:, 0] + nx * extension, izq[:, 1] + ny * extension, tn))
    ext_der = np.column_stack((der[:, 0] - nx * extension, der[:, 1] - ny * extension, tn))
    return {
        "izq": izq,
        "der": der,
        "ext_izq": ext_izq,
        "ext_der": ext_der,
        "tipo": np.sign(delta),
    }


def tira_a_malla(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fase 8: convierte dos polilíneas paralelas en una malla triangular tipo tira.
    """
    if len(a) != len(b) or len(a) < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    vertices = np.vstack((a, b))
    n = len(a)
    caras = []
    for i in range(n - 1):
        caras.append([i, i + 1, n + i + 1])
        caras.append([i, n + i + 1, n + i])
    return vertices, np.asarray(caras, dtype=int)
