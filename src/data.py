"""
Carga y limpieza de los conjuntos de datos abiertos del proyecto.

Dos casos:
  - German Credit (UCI / Statlog): caso guía transversal (scoring crediticio)
    que recorre las tres secciones (XAI -> causalidad -> integración).
  - NSW / Lalonde (NBER, Dehejia-Wahba): benchmark causal clásico para
    estimar el efecto de un programa de empleo sobre los ingresos.

Los archivos crudos se esperan en ``data/raw/`` (ver ``data/README.md``).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

# Raíz del repo = carpeta padre de src/
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# --------------------------------------------------------------------------- #
# German Credit (Statlog)
# --------------------------------------------------------------------------- #
GERMAN_COLUMNS = [
    "estado_cuenta",          # A11-A14  cualitativa: saldo cuenta corriente
    "duracion_meses",         # numérica
    "historial_credito",      # A30-A34
    "proposito",              # A40-A410
    "monto_credito",          # numérica
    "ahorros",                # A61-A65
    "empleo_desde",           # A71-A75
    "tasa_cuota_pct",         # numérica (% del ingreso disponible)
    "estado_personal_sexo",   # A91-A95
    "otros_deudores",         # A101-A103
    "residencia_desde",       # numérica
    "propiedad",              # A121-A124
    "edad",                   # numérica
    "otros_planes_pago",      # A141-A143
    "vivienda",               # A151-A153
    "n_creditos_banco",       # numérica
    "trabajo",                # A171-A174
    "n_dependientes",         # numérica
    "telefono",               # A191-A192
    "trabajador_extranjero",  # A201-A202
    "riesgo",                 # clase: 1=bueno, 2=malo
]

# Decodificación legible de los códigos categóricos relevantes.
GERMAN_DECODE = {
    "estado_cuenta": {
        "A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM",
        "A14": "sin cuenta",
    },
    "historial_credito": {
        "A30": "sin créditos/todos pagados", "A31": "pagados en este banco",
        "A32": "al día", "A33": "atrasos previos", "A34": "crítica/otros créditos",
    },
    "ahorros": {
        "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
        "A64": ">= 1000 DM", "A65": "desconocido/sin ahorros",
    },
    "empleo_desde": {
        "A71": "desempleado", "A72": "< 1 año", "A73": "1-4 años",
        "A74": "4-7 años", "A75": ">= 7 años",
    },
    "proposito": {
        "A40": "carro nuevo", "A41": "carro usado", "A42": "mobiliario",
        "A43": "radio/TV", "A44": "electrodomésticos", "A45": "reparaciones",
        "A46": "educación", "A47": "vacaciones", "A48": "reentrenamiento",
        "A49": "negocio", "A410": "otros",
    },
    "propiedad": {
        "A121": "inmueble", "A122": "seguro/ahorro pensional",
        "A123": "carro u otro", "A124": "sin propiedad/desconocido",
    },
    "vivienda": {"A151": "alquiler", "A152": "propia", "A153": "gratuita"},
    "trabajo": {
        "A171": "no calificado no residente", "A172": "no calificado residente",
        "A173": "calificado/empleado", "A174": "directivo/autónomo",
    },
}


def load_german_credit(decode: bool = True, raw_dir: Path | str = RAW) -> pd.DataFrame:
    """Devuelve el German Credit como DataFrame limpio.

    Parameters
    ----------
    decode : bool
        Si es True, traduce los códigos A1x a etiquetas legibles y crea las
        columnas derivadas ``sexo`` (para auditorías de equidad) y
        ``riesgo_bueno`` (1 = buen pagador, 0 = mal pagador).
    """
    path = Path(raw_dir) / "german.data"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=GERMAN_COLUMNS)

    # Objetivo: en el original 1=bueno, 2=malo. Codificamos 1=bueno, 0=malo.
    df["riesgo_bueno"] = (df["riesgo"] == 1).astype(int)

    # Variable sensible 'sexo' a partir del estado personal (para fairness).
    sexo_map = {
        "A91": "hombre", "A93": "hombre", "A94": "hombre",
        "A92": "mujer", "A95": "mujer",
    }
    df["sexo"] = df["estado_personal_sexo"].map(sexo_map)

    if decode:
        for col, mapping in GERMAN_DECODE.items():
            df[col] = df[col].map(lambda v, m=mapping: m.get(v, v))

    return df


# --------------------------------------------------------------------------- #
# NSW / Lalonde (programa de empleo) — benchmark causal
# --------------------------------------------------------------------------- #
NSW_COLUMNS = [
    "tratamiento",   # 1 = participó en el programa de empleo, 0 = control
    "edad",
    "educacion",     # años de educación
    "afroamericano", # 1/0
    "hispano",       # 1/0
    "casado",        # 1/0
    "sin_grado",     # 1 = sin diploma de secundaria
    "re75",          # ingresos reales 1975 (antes del tratamiento)
    "re78",          # ingresos reales 1978 (resultado)
]


def load_nsw(raw_dir: Path | str = RAW) -> pd.DataFrame:
    """Devuelve el experimento NSW (tratados + controles) en un DataFrame.

    El ingreso de 1978 (``re78``) es el resultado; ``tratamiento`` es la
    intervención. Al ser un experimento aleatorizado, el efecto promedio del
    tratamiento (ATE) se puede estimar con una simple diferencia de medias,
    lo que sirve de 'verdad de referencia' frente a estimadores observacionales.
    """
    raw_dir = Path(raw_dir)
    tratados = pd.read_csv(raw_dir / "nsw_dw_treated.txt", sep=r"\s+",
                           header=None, names=NSW_COLUMNS)
    control = pd.read_csv(raw_dir / "nsw_dw_control.txt", sep=r"\s+",
                          header=None, names=NSW_COLUMNS)
    df = pd.concat([tratados, control], ignore_index=True)
    for c in ["tratamiento", "afroamericano", "hispano", "casado", "sin_grado"]:
        df[c] = df[c].astype(int)
    return df


# Controles observacionales (CPS/PSID) traen además re74 (10 columnas).
_OBS_COLUMNS = ["tratamiento", "edad", "educacion", "afroamericano", "hispano",
                "casado", "sin_grado", "re74", "re75", "re78"]


def load_lalonde_obs(control="cps", raw_dir: Path | str = RAW) -> pd.DataFrame:
    """Muestra observacional de LaLonde: tratados NSW + controles no experimentales.

    Reemplaza el grupo de control aleatorizado por una muestra de población general
    (CPS o PSID). El estimador ingenuo se vuelve fuertemente sesgado: es el desafío
    clásico de LaLonde (1986). Se alinea a las 9 columnas comunes con NSW (se
    descarta ``re74``, ausente en la muestra experimental).
    """
    raw_dir = Path(raw_dir)
    tratados = pd.read_csv(raw_dir / "nsw_dw_treated.txt", sep=r"\s+",
                           header=None, names=NSW_COLUMNS)
    fname = {"cps": "cps_controls.txt", "psid": "psid_controls.txt"}[control]
    obs = pd.read_csv(raw_dir / fname, sep=r"\s+", header=None, names=_OBS_COLUMNS)
    df = pd.concat([tratados, obs[NSW_COLUMNS]], ignore_index=True)
    for c in ["tratamiento", "afroamericano", "hispano", "casado", "sin_grado"]:
        df[c] = df[c].astype(int)
    return df


# --------------------------------------------------------------------------- #
# Datos abiertos colombianos (cuaderno 11)
#   - ICFES Saber 11 (datos.gov.co, Socrata): caso observacional (XAI + equidad).
#   - Vouchers PACES (Angrist et al. 2002, Harvard Dataverse): experimento
#     aleatorizado (lotería) como verdad causal de referencia.
# Ambos se descargan bajo demanda y se cachean en data/raw/colombia/.
# --------------------------------------------------------------------------- #
import ssl
import urllib.parse
import urllib.request

COL_RAW = ROOT / "data" / "raw" / "colombia"

# Contexto SSL tolerante (algunos portales gubernamentales tienen cadenas de
# certificado incompletas en Windows); solo se usa para descargar datos públicos.
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE


def _download(url: str, dest: Path, force: bool = False) -> Path:
    """Descarga ``url`` a ``dest`` (cachea; no re-descarga si ya existe)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180, context=_SSL) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


# --- ICFES Saber 11 -------------------------------------------------------- #
SABER11_ID = "kgxf-xxbe"          # "Resultados únicos Saber 11" en datos.gov.co
SABER11_COLS = [
    "punt_global", "punt_matematicas", "punt_lectura_critica",
    "punt_c_naturales", "punt_sociales_ciudadanas", "punt_ingles",
    "cole_naturaleza", "cole_bilingue", "cole_jornada", "cole_calendario",
    "cole_caracter", "cole_area_ubicacion", "cole_depto_ubicacion", "cole_genero",
    "estu_genero", "estu_depto_reside",
    "fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre",
    "fami_tieneinternet", "fami_tienecomputador", "fami_tieneautomovil",
    "fami_tienelavadora", "fami_cuartoshogar", "fami_personashogar",
]
_SABER11_PUNT = [c for c in SABER11_COLS if c.startswith("punt_")]


def load_saber11(periodo: str = "20224", n: int = 60000,
                 force: bool = False) -> pd.DataFrame:
    """Muestra reproducible de Saber 11 (un periodo) desde datos.gov.co.

    Descarga por la API de Socrata las ``SABER11_COLS`` para un ``periodo``
    (por defecto ``20224`` = calendario A 2022, escala 0-500 post-reforma 2014),
    ordenada por ``estu_consecutivo`` y limitada a ``n`` filas (muestra fija,
    reproducible). Cachea en ``data/raw/colombia/``.
    """
    dest = COL_RAW / f"saber11_{periodo}_n{n}.csv"
    qs = urllib.parse.urlencode({
        "$select": ",".join(SABER11_COLS + ["estu_consecutivo"]),
        "$where": f"periodo='{periodo}'",
        "$order": "estu_consecutivo",
        "$limit": n,
    })
    url = f"https://www.datos.gov.co/resource/{SABER11_ID}.csv?{qs}"
    _download(url, dest, force=force)
    df = pd.read_csv(dest)
    for c in _SABER11_PUNT:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["punt_global"]).reset_index(drop=True)
    return df


# --- Vouchers PACES (Angrist et al. 2002) ---------------------------------- #
PACES_FILE_ID = 1232224           # aerdat4.tab en Harvard Dataverse (hdl:1902.1/11298)


def load_paces(force: bool = False) -> pd.DataFrame:
    """Datos de réplica del experimento de vouchers PACES (Angrist et al., 2002).

    Lotería aleatoria de becas para colegio privado en Colombia. Variables clave:
    ``vouch0`` (ganó la lotería = tratamiento aleatorizado), ``scyfnsh`` (años de
    escolaridad terminados), ``finish8`` (terminó 8.º grado), ``rept`` (repitió),
    ``usngsch`` (usó beca = variable endógena), ``prsch_c`` (en colegio privado);
    covariables ``sex``, ``age``, ``mom_sch``, ``dad_sch``, ``strata1..6``,
    ``dbogota``, ``d1995``/``d1997``. Cachea en ``data/raw/colombia/``.
    """
    dest = COL_RAW / "paces_aerdat4.tab"
    url = f"https://dataverse.harvard.edu/api/access/datafile/{PACES_FILE_ID}"
    _download(url, dest, force=force)
    df = pd.read_csv(dest, sep="\t", na_values=["."])
    return df


if __name__ == "__main__":  # diagnóstico rápido
    g = load_german_credit()
    n = load_nsw()
    print("German Credit:", g.shape, "| % buen pagador:", round(g.riesgo_bueno.mean(), 3))
    print("NSW:", n.shape, "| tratados:", int(n.tratamiento.sum()),
          "| ATE crudo (re78):",
          round(n.loc[n.tratamiento == 1, "re78"].mean()
                - n.loc[n.tratamiento == 0, "re78"].mean(), 1))
