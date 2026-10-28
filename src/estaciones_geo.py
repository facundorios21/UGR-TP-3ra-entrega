"""
Información espacial y geolocalizada de las 44 estaciones activas
de las Líneas A, B y C de la Red de Subterráneos de CABA.
Incluye latitud, longitud, posición ordinal en la traza (desde cabecera centro),
indicadores de cabecera, combinación/transbordo y distancia al Obelisco (Microcentro).
"""
import numpy as np
import pandas as pd

# Coordenadas de referencia del centro neurálgico (Obelisco / Diagonal Norte)
OBELISCO_LAT = -34.6037389
OBELISCO_LON = -58.3815704

# Diccionario de las 44 estaciones normalizadas:
# (latitud, longitud, orden_desde_centro, es_cabecera, es_combinacion)
ESTACIONES_GEO = {
    # --- LÍNEA A (18 estaciones: Plaza de Mayo -> San Pedrito) ---
    'Plaza de Mayo':       (-34.608810, -58.370968, 1,  1, 0),
    'Perú':                (-34.608559, -58.374268, 2,  0, 1),
    'Piedras':             (-34.608882, -58.379088, 3,  0, 0),
    'Lima':                (-34.609099, -58.382547, 4,  0, 1),
    'Sáenz Peña':          (-34.609413, -58.386777, 5,  0, 0),
    'Congreso':            (-34.609226, -58.392669, 6,  0, 0),
    'Pasco':               (-34.609646, -58.398427, 7,  0, 0),
    'Alberti':             (-34.609834, -58.401204, 8,  0, 0),
    'Plaza Miserere':      (-34.609817, -58.406709, 9,  0, 1),
    'Loria':               (-34.610782, -58.415186, 10, 0, 0),
    'Castro Barros':       (-34.611770, -58.421816, 11, 0, 0),
    'Río de Janeiro':      (-34.615206, -58.429500, 12, 0, 0),
    'Acoyte':              (-34.618280, -58.436429, 13, 0, 0),
    'Primera Junta':       (-34.620405, -58.441178, 14, 0, 0),
    'Puán':                (-34.623529, -58.448648, 15, 0, 0),
    'Carabobo':            (-34.626667, -58.456710, 16, 0, 0),
    'San José de Flores':  (-34.629087, -58.463541, 17, 0, 1),
    'San Pedrito':         (-34.630707, -58.469640, 18, 1, 0),

    # --- LÍNEA B (17 estaciones: Leandro N. Alem -> Juan Manuel de Rosas) ---
    'Leandro N. Alem':      (-34.602989, -58.369929, 1,  1, 1),
    'Florida':              (-34.603297, -58.375072, 2,  0, 0),
    'Carlos Pellegrini':    (-34.603637, -58.380715, 3,  0, 1),
    'Uruguay':              (-34.604094, -58.387296, 4,  0, 0),
    'Callao (B)':           (-34.604420, -58.392314, 5,  0, 0),
    'Pasteur':              (-34.604643, -58.399474, 6,  0, 0),
    'Pueyrredón':           (-34.604581, -58.405399, 7,  0, 1),
    'Carlos Gardel':        (-34.604080, -58.411763, 8,  0, 0),
    'Medrano':              (-34.603165, -58.420962, 9,  0, 0),
    'Ángel Gallardo':       (-34.602162, -58.431274, 10, 0, 0),
    'Malabia':              (-34.598967, -58.439771, 11, 0, 0),
    'Dorrego':              (-34.591718, -58.447573, 12, 0, 1),
    'Federico Lacroze':     (-34.587028, -58.455029, 13, 0, 1),
    'Tronador':             (-34.584095, -58.465472, 14, 0, 0),
    'De los Incas':         (-34.581249, -58.474241, 15, 0, 0),
    'Echeverría':           (-34.577797, -58.481014, 16, 0, 0),
    'Juan Manuel de Rosas': (-34.574319, -58.486385, 17, 1, 1),

    # --- LÍNEA C (9 estaciones: Retiro -> Constitución) ---
    'Retiro':              (-34.591194, -58.374018, 1, 1, 1),
    'General San Martín':  (-34.595057, -58.377819, 2, 0, 0),
    'Lavalle':             (-34.601770, -58.378156, 3, 0, 0),
    'Diagonal Norte':      (-34.604844, -58.379530, 4, 0, 1),
    'Avenida de Mayo':     (-34.608983, -58.380611, 5, 0, 1),
    'Moreno':              (-34.612617, -58.380444, 6, 0, 0),
    'Independencia':       (-34.618126, -58.380174, 7, 0, 1),
    'San Juan':            (-34.621917, -58.379921, 8, 0, 0),
    'Constitución':        (-34.627619, -58.381434, 9, 1, 1),
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Calcula la distancia ortodrómica (Haversine) en kilómetros."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def get_estaciones_metadata_df() -> pd.DataFrame:
    """Retorna un DataFrame con las variables espaciales de las 44 estaciones."""
    rows = []
    for est, (lat, lon, orden, cabecera, combinacion) in ESTACIONES_GEO.items():
        dist_centro = float(haversine_km(lat, lon, OBELISCO_LAT, OBELISCO_LON))
        rows.append({
            'estacion': est,
            'latitud': lat,
            'longitud': lon,
            'orden_linea': orden,
            'es_cabecera': cabecera,
            'es_combinacion': combinacion,
            'distancia_centro_km': round(dist_centro, 4)
        })
    return pd.DataFrame(rows)
