from __future__ import annotations

import io
import json
import math
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

import topografiasextras as topo


st.set_page_config(
    page_title="Simulador Computacional de Trazado Vial",
    page_icon="🛣️",
    layout="wide",
)

CONFIG_PLOTLY = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "modelo_vial"},
}

FASES = {
    1: "Abrir la libreta topográfica",
    2: "Clavar los palillos: nube 3D",
    3: "Armar el esqueleto: TIN",
    4: "Poner arcilla al esqueleto: MDE",
    5: "Convertir el terreno en maqueta",
    6: "Definir las reglas del camino",
    7: "Dibujar el eje y la rasante",
    8: "Meter el tractor: corte y relleno",
    9: "Guardar el diseño en el archivero",
    10: "Emitir la memoria de cálculo",
}


def iniciar_estado() -> None:
    """Inicializa todas las variables persistentes de la aplicación."""
    valores = {
        "fase_actual": 1,
        "completadas": set(),
        "xyz": None,
        "df": None,
        "tri": None,
        "tri_validos": None,
        "aristas": None,
        "gx": None,
        "gy": None,
        "zz": None,
        "mascara": None,
        "eje": None,
        "parametros": {"ancho": 8.0, "presupuesto": 250000.0},
        "pendientes": [],
        "movimiento": None,
        "pdf_bytes": None,
        "controles": [],
        "db_path": None,
    }
    for clave, valor in valores.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor
    if st.session_state.db_path is None:
        st.session_state.db_path = str(Path(tempfile.gettempdir()) / f"trazado_vial_{id(st.session_state)}.sqlite")
        crear_bd(st.session_state.db_path)


def crear_bd(ruta: str) -> None:
    """Crea el archivo SQLite temporal del archivero."""
    with sqlite3.connect(ruta) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS disenos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha TEXT NOT NULL,
                nombre TEXT NOT NULL,
                parametros TEXT NOT NULL,
                longitud_proyectada REAL,
                longitud_construida REAL,
                volumen_corte REAL,
                volumen_relleno REAL,
                volumen_total REAL,
                cumple_norma INTEGER
            )
            """
        )


def marcar_completada(fase: int) -> None:
    """Marca una fase como completada y avanza secuencialmente."""
    st.session_state.completadas.add(fase)
    if fase < 10:
        st.session_state.fase_actual = fase + 1
    st.rerun()


def fase_habilitada(numero: int) -> bool:
    return numero == 1 or (numero - 1) in st.session_state.completadas


def menu_lateral() -> None:
    st.sidebar.title("🛣️ Ruta del proyecto")
    progreso = len(st.session_state.completadas) / 10
    st.sidebar.progress(progreso, text=f"{len(st.session_state.completadas)}/10 fases completadas")
    for numero, titulo in FASES.items():
        if numero in st.session_state.completadas:
            icono = "✅"
        elif fase_habilitada(numero):
            icono = "🔓"
        else:
            icono = "🔒"
        if st.sidebar.button(
            f"{icono} {numero}. {titulo}",
            key=f"nav_{numero}",
            disabled=not fase_habilitada(numero),
            use_container_width=True,
        ):
            st.session_state.fase_actual = numero
            st.rerun()
    st.sidebar.caption("Las fases se habilitan en orden para conservar la trazabilidad del diseño.")


def leer_archivo_topografico(archivo, separador: str, encabezado: bool) -> pd.DataFrame:
    """Lee CSV/TXT y normaliza las columnas X/Y/Z."""
    sep_map = {"Coma": ",", "Punto y coma": ";", "Tabulación": "\t", "Espacio": r"\s+"}
    sep = sep_map[separador]
    header = 0 if encabezado else None
    df = pd.read_csv(archivo, sep=sep, header=header, engine="python")
    if not encabezado:
        if df.shape[1] < 3:
            raise ValueError("El archivo debe contener al menos tres columnas numéricas.")
        nombres = ["PUNTO", "X", "Y", "Z", "CODIGO"][: df.shape[1]]
        df.columns = nombres

    alias = {
        "x": {"x", "este", "easting", "coordx", "coordenadax"},
        "y": {"y", "norte", "northing", "coordy", "coordenaday"},
        "z": {"z", "cota", "elevacion", "elevación", "height", "altura"},
    }
    normalizadas = {str(c).strip().lower().replace(" ", ""): c for c in df.columns}
    encontradas = {}
    for eje, opciones in alias.items():
        for opcion in opciones:
            if opcion.replace(" ", "") in normalizadas:
                encontradas[eje.upper()] = normalizadas[opcion.replace(" ", "")]
                break

    if len(encontradas) < 3:
        # Respaldo para archivos sin encabezados formales.
        numericas = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
        if len(numericas) >= 3:
            candidatas = numericas[-3:] if len(numericas) >= 4 else numericas[:3]
            encontradas = {"X": candidatas[0], "Y": candidatas[1], "Z": candidatas[2]}
        else:
            raise ValueError("No se encontraron columnas X, Y y Z. Revise los nombres o el separador.")

    salida = pd.DataFrame({
        "X": pd.to_numeric(df[encontradas["X"]], errors="coerce"),
        "Y": pd.to_numeric(df[encontradas["Y"]], errors="coerce"),
        "Z": pd.to_numeric(df[encontradas["Z"]], errors="coerce"),
    }).dropna()
    if len(salida) < 3:
        raise ValueError("Después de limpiar datos inválidos quedaron menos de tres puntos.")
    salida.insert(0, "PUNTO", np.arange(1, len(salida) + 1))
    return salida


def fig_terreno_base() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=st.session_state.gx,
        y=st.session_state.gy,
        z=st.session_state.zz,
        colorscale="Earth",
        opacity=0.85,
        showscale=True,
        colorbar_title="Cota (m)",
    ))
    fig.update_layout(scene={"aspectmode": "data"}, margin=dict(l=0, r=0, t=35, b=0))
    return fig


def mesh_trace(vertices: np.ndarray, caras: np.ndarray, nombre: str, color: str, opacity: float = 1.0):
    if len(vertices) == 0 or len(caras) == 0:
        return None
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=caras[:, 0], j=caras[:, 1], k=caras[:, 2],
        name=nombre, color=color, opacity=opacity, flatshading=True,
    )


def formato_pk(pk: float) -> str:
    km = int(pk // 1000)
    metros = pk - km * 1000
    return f"{km}+{metros:06.2f}"


def render_fase_1():
    st.header("📒 Fase 1 — Abrir la libreta topográfica")
    st.write("Cargue el levantamiento o genere un terreno sintético para recorrer el simulador completo.")
    c1, c2 = st.columns(2)
    separador = c1.selectbox("Separador", ["Coma", "Punto y coma", "Tabulación", "Espacio"])
    encabezado = c2.checkbox("El archivo contiene encabezado", value=True)
    archivo = st.file_uploader("Archivo CSV o TXT", type=["csv", "txt"])

    col_a, col_b = st.columns(2)
    if col_a.button("🧪 Generar datos de ejemplo", use_container_width=True):
        xyz = topo.generar_terreno_sintetico()
        df = pd.DataFrame(xyz, columns=["X", "Y", "Z"])
        df.insert(0, "PUNTO", np.arange(1, len(df) + 1))
        st.session_state.xyz = xyz
        st.session_state.df = df
        st.session_state.pdf_bytes = None
        st.success("Terreno sintético generado correctamente.")

    if archivo is not None and col_b.button("📥 Procesar archivo", use_container_width=True):
        try:
            df = leer_archivo_topografico(archivo, separador, encabezado)
            st.session_state.df = df
            st.session_state.xyz = df[["X", "Y", "Z"]].to_numpy(float)
            st.session_state.pdf_bytes = None
            st.success("Levantamiento importado correctamente.")
        except Exception as exc:
            st.error(f"No fue posible cargar el archivo: {exc}")

    if st.session_state.xyz is not None:
        xyz = st.session_state.xyz
        a, b, c = st.columns(3)
        a.metric("Puntos", f"{len(xyz):,}")
        b.metric("Cota mín./máx.", f"{xyz[:,2].min():.2f} / {xyz[:,2].max():.2f} m")
        c.metric("Desnivel", f"{np.ptp(xyz[:,2]):.2f} m")
        st.dataframe(st.session_state.df.head(30), use_container_width=True)
        if st.button("Confirmar y pasar a la siguiente fase", type="primary"):
            marcar_completada(1)


def render_fase_2():
    st.header("📍 Fase 2 — Clavar los palillos: nube de puntos 3D")
    xyz = st.session_state.xyz
    fig = go.Figure(go.Scatter3d(
        x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
        mode="markers",
        marker=dict(size=3, color=xyz[:,2], colorscale="Earth", colorbar=dict(title="Cota")),
        text=[f"X={x:.2f}<br>Y={y:.2f}<br>Z={z:.2f}" for x,y,z in xyz],
        hoverinfo="text",
    ))
    fig.update_layout(scene={"aspectmode": "data"}, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True, config=CONFIG_PLOTLY)
    if st.button("Confirmar y pasar a la siguiente fase", type="primary"):
        marcar_completada(2)


def render_fase_3():
    st.header("🕸️ Fase 3 — Armar el esqueleto: triangulación TIN")
    xyz = st.session_state.xyz
    extension = max(np.ptp(xyz[:,0]), np.ptp(xyz[:,1]))
    lado = st.slider("Lado máximo admisible del triángulo (m)", 5.0, float(max(20.0, extension/2)), float(max(25.0, extension/12)), 1.0)
    if st.button("Calcular o actualizar TIN"):
        try:
            tri, validos = topo.construir_tin(xyz, lado)
            st.session_state.tri = tri
            st.session_state.tri_validos = validos
            st.session_state.aristas = topo.aristas_tin(tri.simplices, validos)
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.aristas is not None:
        xe, ye, ze = [], [], []
        for i, j in st.session_state.aristas:
            xe += [xyz[i,0], xyz[j,0], None]
            ye += [xyz[i,1], xyz[j,1], None]
            ze += [xyz[i,2], xyz[j,2], None]
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=xe,y=ye,z=ze,mode="lines",line=dict(width=2),name="TIN"))
        fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode="markers",marker=dict(size=2,color=xyz[:,2],colorscale="Earth"),name="Puntos"))
        fig.update_layout(scene={"aspectmode":"data"}, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True, config=CONFIG_PLOTLY)
        st.metric("Triángulos válidos", int(np.sum(st.session_state.tri_validos)))
        if st.button("Confirmar y pasar a la siguiente fase", type="primary"):
            marcar_completada(3)


def render_fase_4():
    st.header("🏔️ Fase 4 — Poner arcilla al esqueleto: MDE y curvas")
    resolucion = st.slider("Resolución de malla", 40, 180, 100, 10)
    n_curvas = st.slider("Número aproximado de curvas", 5, 40, 15)
    if st.button("Interpolar MDE"):
        gx, gy, zz, mask = topo.crear_mde(
            st.session_state.xyz,
            st.session_state.tri,
            st.session_state.tri_validos,
            resolucion,
        )
        st.session_state.gx, st.session_state.gy = gx, gy
        st.session_state.zz, st.session_state.mascara = zz, mask

    if st.session_state.zz is not None:
        zmin, zmax = np.nanmin(st.session_state.zz), np.nanmax(st.session_state.zz)
        paso = max((zmax-zmin)/n_curvas, 0.1)
        fig = go.Figure(go.Surface(
            x=st.session_state.gx, y=st.session_state.gy, z=st.session_state.zz,
            colorscale="Earth",
            contours={"z":{"show":True,"start":zmin,"end":zmax,"size":paso,"project_z":True}},
        ))
        fig.update_layout(scene={"aspectmode":"data"}, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True, config=CONFIG_PLOTLY)
        if st.button("Confirmar y pasar a la siguiente fase", type="primary"):
            marcar_completada(4)


def render_fase_5():
    st.header("🧱 Fase 5 — Convertir el terreno en una maqueta sólida")
    base = float(np.nanmin(st.session_state.zz))
    volumen = topo.volumen_bloque_mde(st.session_state.gx, st.session_state.gy, st.session_state.zz, base)
    vertices, caras = topo.construir_maqueta_malla(st.session_state.gx, st.session_state.gy, st.session_state.zz, base)
    fig = go.Figure(mesh_trace(vertices, caras, "Maqueta", "sienna", 1.0))
    fig.update_layout(scene={"aspectmode":"data"}, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True, config=CONFIG_PLOTLY)
    st.metric("Volumen aproximado del bloque", f"{volumen:,.0f} m³")
    if st.button("Confirmar y pasar a la siguiente fase", type="primary"):
        marcar_completada(5)


def render_fase_6():
    st.header("📐 Fase 6 — Definir las reglas del camino")
    with st.form("parametros_viales"):
        ancho = st.number_input("Ancho de calzada (m)", 3.0, 30.0, float(st.session_state.parametros["ancho"]), 0.5)
        presupuesto = st.number_input("Presupuesto máximo de movimiento de tierras (m³)", 100.0, 1e9, float(st.session_state.parametros["presupuesto"]), 1000.0)
        guardar = st.form_submit_button("Guardar parámetros")
    if guardar:
        st.session_state.parametros = {"ancho": ancho, "presupuesto": presupuesto}
        st.success("Parámetros guardados.")
    st.info("El presupuesto se interpreta como volumen máximo acumulado de corte más relleno.")
    if st.button("Confirmar y pasar a la siguiente fase", type="primary"):
        marcar_completada(6)


def punto_extremo_desde_minimo(direccion: str) -> list[list[float]]:
    xyz = st.session_state.xyz
    inicio = xyz[np.argmin(xyz[:,2]), :2]
    if direccion == "Norte":
        idx = np.argmax(xyz[:,1])
    elif direccion == "Sur":
        idx = np.argmin(xyz[:,1])
    elif direccion == "Este":
        idx = np.argmax(xyz[:,0])
    else:
        idx = np.argmin(xyz[:,0])
    return [inicio.tolist(), xyz[idx,:2].tolist()]


def render_fase_7():
    st.header("🧭 Fase 7 — Dibujar el eje y levantar la rasante")
    st.caption("Ingrese los puntos de control en orden. El mapa sirve como referencia interactiva.")
    fig2d = go.Figure(go.Heatmap(
        x=st.session_state.gx, y=st.session_state.gy, z=st.session_state.zz,
        colorscale="Earth", colorbar=dict(title="Cota"),
    ))
    if st.session_state.controles:
        p = np.asarray(st.session_state.controles)
        fig2d.add_trace(go.Scatter(x=p[:,0], y=p[:,1], mode="lines+markers", name="Controles"))
    fig2d.update_layout(xaxis_title="X", yaxis_title="Y", yaxis_scaleanchor="x")
    st.plotly_chart(fig2d, use_container_width=True, config=CONFIG_PLOTLY)

    dirs = st.columns(4)
    for col, d in zip(dirs, ["Norte","Sur","Este","Oeste"]):
        if col.button(f"Inicio rápido: {d}", use_container_width=True):
            st.session_state.controles = punto_extremo_desde_minimo(d)
            st.rerun()

    controles_df = pd.DataFrame(st.session_state.controles or [[float(st.session_state.xyz[:,0].min()), float(st.session_state.xyz[:,1].min())],
                                                               [float(st.session_state.xyz[:,0].max()), float(st.session_state.xyz[:,1].max())]],
                                columns=["X","Y"])
    editado = st.data_editor(controles_df, num_rows="dynamic", use_container_width=True)
    sigma = st.slider("Suavizado gaussiano sigma", 0.0, 10.0, 2.0, 0.5)
    max_pend = st.number_input("Pendiente normativa máxima (%)", 1.0, 30.0, 12.0, 0.5)

    if st.button("Calcular eje y preparar pendientes"):
        controles = editado[["X","Y"]].dropna().to_numpy(float).tolist()
        st.session_state.controles = controles
        try:
            x, y, s = topo.interpolar_eje(controles, espaciamiento=5.0, sigma=sigma)
            tn = topo.interpolar_cotas_eje(st.session_state.xyz, x, y)
            n_tramos = max(1, int(math.ceil(s[-1]/100.0)))
            if len(st.session_state.pendientes) != n_tramos:
                st.session_state.pendientes = [0.0]*n_tramos
            st.session_state.eje_previo = {"x":x,"y":y,"s":s,"tn":tn,"max_pend":max_pend}
        except Exception as exc:
            st.error(str(exc))

    if "eje_previo" in st.session_state:
        previo = st.session_state.eje_previo
        st.subheader("Pendientes por tramo de 100 m")
        pd_pend = pd.DataFrame({
            "Tramo": [f"{i*100:.0f}-{min((i+1)*100, previo['s'][-1]):.0f} m" for i in range(len(st.session_state.pendientes))],
            "Pendiente (%)": st.session_state.pendientes,
        })
        pd_edit = st.data_editor(pd_pend, disabled=["Tramo"], use_container_width=True)
        st.session_state.pendientes = pd.to_numeric(pd_edit["Pendiente (%)"], errors="coerce").fillna(0).tolist()
        excede = np.abs(st.session_state.pendientes) > previo["max_pend"]
        if np.any(excede):
            st.error(f"Hay {int(np.sum(excede))} tramo(s) que exceden el máximo de {previo['max_pend']:.1f}%.")
        else:
            st.success("Todas las pendientes cumplen el límite configurado.")

        if st.button("Aplicar rasante"):
            rasante = topo.calcular_rasante(previo["s"], previo["tn"][0], st.session_state.pendientes)
            st.session_state.eje = topo.EjeVial(previo["x"], previo["y"], previo["s"], previo["tn"], rasante)

    if st.session_state.eje is not None:
        eje = st.session_state.eje
        fig = fig_terreno_base()
        fig.add_trace(go.Scatter3d(x=eje.x,y=eje.y,z=eje.terreno_z,mode="lines",line=dict(width=6),name="Terreno bajo eje"))
        fig.add_trace(go.Scatter3d(x=eje.x,y=eje.y,z=eje.rasante_z,mode="lines",line=dict(width=8),name="Rasante"))
        st.plotly_chart(fig, use_container_width=True, config=CONFIG_PLOTLY)
        st.metric("Longitud proyectada", f"{eje.progresiva[-1]:,.2f} m")
        if st.button("Confirmar y pasar a la siguiente fase", type="primary"):
            marcar_completada(7)


def render_fase_8():
    st.header("🚜 Fase 8 — Meter el tractor: corte y relleno")
    eje = st.session_state.eje
    c1, c2 = st.columns(2)
    talud_c = c1.number_input("Talud de corte (H:V)", 0.1, 5.0, 1.0, 0.1)
    talud_r = c2.number_input("Talud de relleno (H:V)", 0.1, 5.0, 1.5, 0.1)
    truncar = st.checkbox("Truncar construcción cuando se agote el presupuesto", value=True)

    area_c, area_r = topo.areas_movimiento_tierras(
        eje.terreno_z, eje.rasante_z,
        st.session_state.parametros["ancho"], talud_c, talud_r,
    )
    vc, vr, vt = topo.integrar_volumenes(eje.progresiva, area_c, area_r)
    pk_lim = topo.pk_por_presupuesto(eje.progresiva, vt, st.session_state.parametros["presupuesto"])
    pk_construido = pk_lim if truncar else float(eje.progresiva[-1])

    st.session_state.movimiento = {
        "area_corte": area_c, "area_relleno": area_r,
        "vol_corte": vc, "vol_relleno": vr, "vol_total": vt,
        "pk_limite": pk_lim, "pk_construido": pk_construido,
        "talud_corte": talud_c, "talud_relleno": talud_r,
    }

    a,b,c,d = st.columns(4)
    a.metric("Corte total", f"{vc[-1]:,.0f} m³")
    b.metric("Relleno total", f"{vr[-1]:,.0f} m³")
    c.metric("Movimiento total", f"{vt[-1]:,.0f} m³")
    d.metric("Alcance del presupuesto", f"PK {formato_pk(pk_lim)}")

    geom = topo.geometria_carretera(eje, st.session_state.parametros["ancho"], talud_c, talud_r, pk_construido)
    fig = fig_terreno_base()
    v_cal, f_cal = topo.tira_a_malla(geom["izq"], geom["der"])
    tr = mesh_trace(v_cal, f_cal, "Calzada", "dimgray", 1.0)
    if tr: fig.add_trace(tr)
    v_i, f_i = topo.tira_a_malla(geom["izq"], geom["ext_izq"])
    v_d, f_d = topo.tira_a_malla(geom["der"], geom["ext_der"])
    tri = mesh_trace(v_i, f_i, "Talud izquierdo", "peru", 0.85)
    trd = mesh_trace(v_d, f_d, "Talud derecho", "olivedrab", 0.85)
    if tri: fig.add_trace(tri)
    if trd: fig.add_trace(trd)
    st.plotly_chart(fig, use_container_width=True, config=CONFIG_PLOTLY)

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=eje.progresiva, y=vc, name="Corte acumulado"))
    fig_vol.add_trace(go.Scatter(x=eje.progresiva, y=vr, name="Relleno acumulado"))
    fig_vol.add_trace(go.Scatter(x=eje.progresiva, y=vt, name="Total acumulado"))
    fig_vol.add_hline(y=st.session_state.parametros["presupuesto"], line_dash="dash", annotation_text="Presupuesto")
    fig_vol.update_layout(xaxis_title="Progresiva (m)", yaxis_title="Volumen (m³)")
    st.plotly_chart(fig_vol, use_container_width=True, config=CONFIG_PLOTLY)
    if st.button("Confirmar y pasar a la siguiente fase", type="primary"):
        marcar_completada(8)


def render_fase_9():
    st.header("🗄️ Fase 9 — Guardar el diseño en el archivero")
    nombre = st.text_input("Nombre del diseño", value=f"Diseño {datetime.now():%Y-%m-%d %H:%M}")
    max_pend = float(st.session_state.get("eje_previo", {}).get("max_pend", 12.0))
    cumple = bool(np.all(np.abs(st.session_state.pendientes) <= max_pend))
    mov = st.session_state.movimiento
    eje = st.session_state.eje

    if st.button("Guardar diseño confirmado", type="primary"):
        with sqlite3.connect(st.session_state.db_path) as con:
            con.execute(
                """INSERT INTO disenos
                (fecha,nombre,parametros,longitud_proyectada,longitud_construida,
                volumen_corte,volumen_relleno,volumen_total,cumple_norma)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now().isoformat(timespec="seconds"), nombre,
                    json.dumps({
                        **st.session_state.parametros,
                        "talud_corte": mov["talud_corte"],
                        "talud_relleno": mov["talud_relleno"],
                        "pendientes": st.session_state.pendientes,
                    }),
                    float(eje.progresiva[-1]), float(mov["pk_construido"]),
                    float(np.interp(mov["pk_construido"], eje.progresiva, mov["vol_corte"])),
                    float(np.interp(mov["pk_construido"], eje.progresiva, mov["vol_relleno"])),
                    float(np.interp(mov["pk_construido"], eje.progresiva, mov["vol_total"])),
                    int(cumple),
                ),
            )
        st.success("Diseño archivado en SQLite.")

    with sqlite3.connect(st.session_state.db_path) as con:
        registros = pd.read_sql_query("SELECT * FROM disenos ORDER BY id DESC", con)
    st.dataframe(registros, use_container_width=True)
    st.download_button(
        "Descargar base SQLite",
        data=Path(st.session_state.db_path).read_bytes(),
        file_name="archivero_trazados.sqlite",
        mime="application/vnd.sqlite3",
    )
    if st.button("Confirmar y pasar a la siguiente fase"):
        marcar_completada(9)


class PDFMemoria(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", size=8)
        self.cell(0, 10, f"Simulador Computacional de Trazado Vial | Página {self.page_no()}", align="C")


def generar_pdf(nombre_proyecto: str, responsable: str, fiscalizador: str, imagen_bytes: bytes | None) -> bytes:
    eje = st.session_state.eje
    mov = st.session_state.movimiento
    params = st.session_state.parametros
    max_pend = float(st.session_state.get("eje_previo", {}).get("max_pend", 12.0))
    cumple = bool(np.all(np.abs(st.session_state.pendientes) <= max_pend))
    pk = mov["pk_construido"]
    vc = float(np.interp(pk, eje.progresiva, mov["vol_corte"]))
    vr = float(np.interp(pk, eje.progresiva, mov["vol_relleno"]))
    vt = vc + vr

    pdf = PDFMemoria()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "MEMORIA DE CÁLCULO DE TRAZADO VIAL", ln=1, align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 7, f"Proyecto: {nombre_proyecto}", ln=1)
    pdf.cell(0, 7, f"Fecha: {datetime.now():%d/%m/%Y %H:%M}", ln=1)
    pdf.ln(3)

    def titulo(txt):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, txt, ln=1)
        pdf.set_font("Helvetica", size=10)

    titulo("1. Parámetros de diseño")
    for linea in [
        f"Ancho de calzada: {params['ancho']:.2f} m",
        f"Presupuesto de movimiento de tierras: {params['presupuesto']:,.2f} m³",
        f"Talud de corte: {mov['talud_corte']:.2f} H:1V",
        f"Talud de relleno: {mov['talud_relleno']:.2f} H:1V",
    ]:
        pdf.cell(0, 6, linea, ln=1)

    titulo("2. Resultados del trazado")
    for linea in [
        f"Longitud proyectada: {eje.progresiva[-1]:,.2f} m",
        f"Longitud construida: {pk:,.2f} m (PK {formato_pk(pk)})",
        f"Cota inicial de rasante: {eje.rasante_z[0]:.2f} m",
        f"Cota final construida: {np.interp(pk, eje.progresiva, eje.rasante_z):.2f} m",
    ]:
        pdf.cell(0, 6, linea, ln=1)

    titulo("3. Movimiento de tierras")
    uso = 100.0 * vt / params["presupuesto"] if params["presupuesto"] else 0.0
    for linea in [
        f"Volumen de corte construido: {vc:,.2f} m³",
        f"Volumen de relleno construido: {vr:,.2f} m³",
        f"Movimiento total construido: {vt:,.2f} m³",
        f"Porcentaje del presupuesto utilizado: {uso:.2f} %",
    ]:
        pdf.cell(0, 6, linea, ln=1)

    titulo("4. Pendientes por tramo de 100 m")
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(45, 7, "Tramo", border=1)
    pdf.cell(40, 7, "Pendiente (%)", border=1)
    pdf.cell(45, 7, "Verificación", border=1, ln=1)
    pdf.set_font("Helvetica", size=9)
    for i, pendiente in enumerate(st.session_state.pendientes):
        fin = min((i+1)*100, eje.progresiva[-1])
        estado = "CUMPLE" if abs(pendiente) <= max_pend else "NO CUMPLE"
        pdf.cell(45, 7, f"{i*100:.0f}-{fin:.0f} m", border=1)
        pdf.cell(40, 7, f"{pendiente:.2f}", border=1)
        pdf.cell(45, 7, estado, border=1, ln=1)

    titulo("5. Verificación normativa")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"Resultado general: {'CUMPLE' if cumple else 'NO CUMPLE'}", ln=1)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 6, f"Límite de pendiente adoptado para esta simulación: {max_pend:.2f} %. La memoria tiene finalidad educativa y debe ser revisada y firmada por un profesional competente antes de cualquier uso contractual o legal.")

    if imagen_bytes:
        ruta = Path(tempfile.gettempdir()) / f"captura_modelo_{id(imagen_bytes)}.png"
        ruta.write_bytes(imagen_bytes)
        pdf.add_page()
        titulo("6. Captura adjunta del modelo 3D")
        try:
            pdf.image(str(ruta), x=15, w=180)
        except Exception:
            pdf.multi_cell(0, 6, "La imagen adjunta no pudo incorporarse al PDF. Use PNG o JPG.")

    pdf.ln(15)
    pdf.cell(85, 8, "_______________________________", align="C")
    pdf.cell(20, 8, "")
    pdf.cell(85, 8, "_______________________________", align="C", ln=1)
    pdf.cell(85, 6, responsable or "Responsable de diseño", align="C")
    pdf.cell(20, 6, "")
    pdf.cell(85, 6, fiscalizador or "Fiscalizador", align="C", ln=1)

    contenido = pdf.output()
    if isinstance(contenido, str):
        return contenido.encode("latin1")
    return bytes(contenido)


def render_fase_10():
    st.header("📄 Fase 10 — Emitir la memoria de cálculo")
    nombre = st.text_input("Nombre del proyecto", "Proyecto vial académico")
    c1,c2 = st.columns(2)
    responsable = c1.text_input("Responsable de diseño")
    fiscalizador = c2.text_input("Fiscalizador")
    captura = st.file_uploader("Captura opcional del modelo 3D (PNG/JPG)", type=["png","jpg","jpeg"])
    if st.button("Generar memoria PDF", type="primary"):
        try:
            st.session_state.pdf_bytes = generar_pdf(
                nombre, responsable, fiscalizador,
                captura.getvalue() if captura else None,
            )
            st.success("Memoria generada y guardada en la sesión.")
        except Exception as exc:
            st.error(f"No se pudo generar el PDF: {exc}")

    if st.session_state.pdf_bytes:
        st.download_button(
            "⬇️ Descargar memoria de cálculo",
            data=st.session_state.pdf_bytes,
            file_name="memoria_calculo_trazado_vial.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        if st.button("Marcar proyecto como completado"):
            marcar_completada(10)


def main():
    iniciar_estado()
    menu_lateral()
    st.title("Simulador Computacional de Trazado Vial")
    st.caption("Del punto topográfico a la memoria de cálculo, fase por fase.")
    fase = st.session_state.fase_actual
    if not fase_habilitada(fase):
        st.warning("Complete primero la fase anterior.")
        return
    renderizadores = {
        1: render_fase_1, 2: render_fase_2, 3: render_fase_3, 4: render_fase_4,
        5: render_fase_5, 6: render_fase_6, 7: render_fase_7, 8: render_fase_8,
        9: render_fase_9, 10: render_fase_10,
    }
    try:
        renderizadores[fase]()
    except Exception as exc:
        st.error(f"Se produjo un error en la fase {fase}: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()