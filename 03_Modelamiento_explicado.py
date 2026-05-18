# -*- coding: utf-8 -*-
"""
03_Modelamiento — VERSIÓN EXPLICADA
===================================
Modelo de credit scoring (regresión logística regularizada).

Este script es la versión COMENTADA de `03_Modelamiento.ipynb`: cada bloque lleva
arriba una guía breve de qué hace y por qué. Sirve para leer y exponer el flujo.

Flujo general:
  cargar datos -> EDA -> partición DEV/TEST_OOT -> discretización ->
  feature selection (filtro IV + wrapper RFECV) -> optimización de lambda ->
  comparación de técnicas -> pipeline -> t-SNE -> evaluación en TEST_OOT.

Sugerencia: ejecutarlo por celdas (marcadores '# %%') en VSCode / Jupyter.
"""

# %% 1 — Cargar datos
# Se carga la base ya consolidada (salida de 02_VariableDependiente).
# 'info' tiene una fila por crédito, las variables predictoras y el target 'VarDep'.
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

info = pd.read_pickle("BDD/InfoModelamiento.pkl")
print(info.shape)

# Las 5 fechas de corte (cohortes trimestrales) que contiene la base.
print(info.FECHA_CORTE.unique())


# ============================================================================
# EDA — ANÁLISIS EXPLORATORIO
# ============================================================================

# %% 2 — Librerías y variables del EDA
# Se elige a mano un subconjunto de 14 variables con sentido de riesgo (no se
# analizan las ~2500). Se fuerzan a tipo numérico por si vinieran como texto.
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 50)

VARS_NUM = ["edad", "INGRESOS", "CARGAS", "ANTIG_LABORAL", "score419",
            "MAX_DVEN_SCE_6M", "numOpsVencidas101", "NOPE_VENC_OP_3M",
            "DEUDA_TOTAL_SCE_12M", "SALDO_PROMEDIO_AHORRO", "ANTIGUEDAD_SCE"]
VARS_CAT = ["ESTADO_CIVIL", "INSTRUCCION", "PERFIL_CLIENTE"]

info[VARS_NUM] = info[VARS_NUM].apply(pd.to_numeric, errors="coerce")
print("Dimensiones de la base:", info.shape)
print(info[["FECHA_CORTE", "VarDep"] + VARS_NUM[:4] + VARS_CAT].head())

# %% 3 — Población y variable objetivo
# Distribución de las 6 categorías de VarDep (0/1 se modelan; 2-5 son exclusiones).
tab = info["VarDep"].value_counts(dropna=False).sort_index().rename("N").reset_index()
tab.columns = ["VarDep", "N"]
tab["pct"] = (tab["N"] / tab["N"].sum() * 100).round(2)
print(tab)

# Población de scoring = solo bueno (0) / malo (1); se reporta la tasa de malos.
pob = info[info["VarDep"].isin([0, 1])]
print("Población total:           ", len(info))
print("Población de scoring (0/1):", len(pob),
      f"({len(pob) / len(info) * 100:.1f}%)")
print("Tasa de malos:             ", round(pob["VarDep"].mean(), 4))

plt.figure(figsize=(5, 4))
pob["VarDep"].map({0: "Bueno", 1: "Malo"}).value_counts().plot.bar(
    color=["#4c72b0", "#c44e52"])
plt.title("Distribución bueno / malo (población de scoring)")
plt.ylabel("N")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% 4 — Calidad de datos
# Tipo, nº de nulos, % de nulos y nº de valores únicos de cada variable elegida.
vars_eda = VARS_NUM + VARS_CAT + ["VarDep"]
calidad = pd.DataFrame({
    "tipo": info[vars_eda].dtypes.astype(str),
    "n_nulos": info[vars_eda].isna().sum(),
    "pct_nulos": (info[vars_eda].isna().mean() * 100).round(2),
    "n_unicos": info[vars_eda].nunique(),
})
print(calidad)

# %% 5 — Análisis univariado
# Estadísticos descriptivos de las variables numéricas.
print(info[VARS_NUM].describe().T)

# Histograma de cada variable numérica (forma de la distribución, colas).
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
for ax, v in zip(axes.flat, VARS_NUM):
    info[v].plot.hist(bins=40, ax=ax, color="#4c72b0")
    ax.set_title(v)
for ax in axes.flat[len(VARS_NUM):]:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Frecuencia de categorías de las variables categóricas.
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, v in zip(axes, VARS_CAT):
    info[v].value_counts(dropna=False).plot.bar(ax=ax, color="#4c72b0")
    ax.set_title(v)
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# %% 6 — Análisis bivariado con el target
# Tasa de malos por decil de cada variable numérica: si sube/baja de forma
# monótona, la variable discrimina riesgo.
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
for ax, v in zip(axes.flat, VARS_NUM):
    grp = pd.qcut(pob[v], 10, duplicates="drop")
    tasa = pob.groupby(grp, observed=True)["VarDep"].mean()
    tasa.plot.bar(ax=ax, color="#c44e52")
    ax.set_title(f"Tasa de malos por decil — {v}")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
for ax in axes.flat[len(VARS_NUM):]:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Tasa de malos por categoría (la línea negra es la tasa global de referencia).
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, v in zip(axes, VARS_CAT):
    tasa = pob.groupby(v, observed=True)["VarDep"].mean().sort_values()
    tasa.plot.bar(ax=ax, color="#c44e52")
    ax.axhline(pob["VarDep"].mean(), color="black", ls="--", lw=1, label="tasa global")
    ax.set_title(f"Tasa de malos — {v}")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
plt.tight_layout()
plt.show()

# %% 7 — WoE e Information Value
# woe_iv(): agrupa la variable (deciles si es continua, categorías si no) y
# calcula el WoE por grupo y el IV total (poder predictivo de la variable).
def woe_iv(var, bins=10):
    d = info.loc[info["VarDep"].isin([0, 1]), [var, "VarDep"]].copy()
    if var in VARS_CAT:
        d["grupo"] = d[var].astype("object").fillna("(NULO)")
    else:
        d["grupo"] = pd.qcut(d[var], bins, duplicates="drop")
        d["grupo"] = d["grupo"].cat.add_categories(["(NULO)"]).fillna("(NULO)")

    g = d.groupby("grupo", observed=True)["VarDep"].agg(N="count", malos="sum")
    g["buenos"] = g["N"] - g["malos"]
    g["tasa_malos"] = g["malos"] / g["N"]
    g["pct_malos"] = g["malos"] / g["malos"].sum()
    g["pct_buenos"] = g["buenos"] / g["buenos"].sum()
    eps = 1e-6
    g["WoE"] = np.log((g["pct_buenos"] + eps) / (g["pct_malos"] + eps))
    g["IV_parcial"] = (g["pct_buenos"] - g["pct_malos"]) * g["WoE"]
    return g, g["IV_parcial"].sum()


# fuerza(): traduce el valor de IV a una etiqueta de poder predictivo.
def fuerza(iv):
    if iv < 0.02:
        return "inútil"
    if iv < 0.1:
        return "débil"
    if iv < 0.3:
        return "media"
    if iv < 0.5:
        return "fuerte"
    return "sospechosa (>0.5)"


# Ranking de IV de las 14 variables del EDA.
filas = []
for v in VARS_NUM + VARS_CAT:
    _, iv = woe_iv(v)
    filas.append((v, iv))
iv_rank = pd.DataFrame(filas, columns=["variable", "IV"]).sort_values("IV", ascending=False)
iv_rank["poder_predictivo"] = iv_rank["IV"].apply(fuerza)
print(iv_rank.reset_index(drop=True))

# Detalle del WoE de la variable más predictiva.
top_var = iv_rank.iloc[0]["variable"]
print("Variable con mayor IV:", top_var)
g, iv = woe_iv(top_var)
print("IV =", round(iv, 4))
print(g.round(4))

# Gráfico del WoE por grupo de esa variable.
plt.figure(figsize=(9, 4))
g["WoE"].plot.bar(color="#4c72b0")
plt.axhline(0, color="black", lw=1)
plt.title(f"WoE por grupo — {top_var}")
plt.ylabel("WoE")
plt.tight_layout()
plt.show()

# %% 8 — Estabilidad temporal
# Tasa de malos por cohorte: muestra si el riesgo se mueve en el tiempo.
est = (pob.groupby("FECHA_CORTE")["VarDep"]
          .agg(N="count", tasa_malos="mean"))
print(est)

est_plot = est.copy()
est_plot.index = est_plot.index.astype(str)   # índice a texto: evita el bug del bar plot con fechas

plt.figure(figsize=(7, 4))
est_plot["tasa_malos"].plot.bar(color="#c44e52")
plt.axhline(pob["VarDep"].mean(), color="black", ls="--", lw=1, label="tasa global")
plt.title("Tasa de malos por cohorte (FECHA_CORTE)")
plt.ylabel("Tasa de malos")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Media de cada variable por cohorte (chequeo grueso de estabilidad).
print(info.groupby("FECHA_CORTE")[VARS_NUM].mean().T.round(2))


# psi(): Population Stability Index — compara dos distribuciones; mide si una
# variable cambió de forma entre dos momentos (<0.1 estable).
def psi(base, actual, bins=10):
    base = pd.Series(base).dropna()
    actual = pd.Series(actual).dropna()
    cortes = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    e = pd.cut(base, cortes, include_lowest=True).value_counts(normalize=True).sort_index()
    a = pd.cut(actual, cortes, include_lowest=True).value_counts(normalize=True).sort_index()
    e, a = e.align(a, fill_value=0)
    e = e.replace(0, 1e-6)
    a = a.replace(0, 1e-6)
    return float(((a - e) * np.log(a / e)).sum())


# PSI de score419 de cada cohorte contra la primera (referencia).
cohortes = sorted(info["FECHA_CORTE"].unique())
base = info.loc[info["FECHA_CORTE"] == cohortes[0], "score419"]
print("PSI de score419 (referencia = cohorte", str(cohortes[0])[:10], "):")
for ch in cohortes[1:]:
    act = info.loc[info["FECHA_CORTE"] == ch, "score419"]
    print(f"  {str(ch)[:10]}:  PSI = {psi(base, act):.4f}")

# %% 9 — Correlaciones y outliers
# Matriz de correlación entre las variables numéricas.
corr_eda = info[VARS_NUM].corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr_eda, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(VARS_NUM)))
ax.set_xticklabels(VARS_NUM, rotation=90)
ax.set_yticks(range(len(VARS_NUM)))
ax.set_yticklabels(VARS_NUM)
for i in range(len(VARS_NUM)):
    for j in range(len(VARS_NUM)):
        ax.text(j, i, f"{corr_eda.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, fraction=0.046)
plt.title("Correlación entre variables numéricas")
plt.tight_layout()
plt.show()

# Boxplots para ver outliers (colas largas).
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
for ax, v in zip(axes.flat, VARS_NUM):
    info.boxplot(column=v, ax=ax)
    ax.set_title(v)
for ax in axes.flat[len(VARS_NUM):]:
    ax.axis("off")
plt.tight_layout()
plt.show()


# ============================================================================
# PARTICIÓN DESARROLLO / TEST OOT
# ============================================================================

# %% 10 — Partición DEV / TEST_OOT
# DEV  = cohortes 2021-09 a 2022-06 -> todo el desarrollo.
# TEST_OOT = cohorte más reciente (2022-09) -> prueba fuera de tiempo, intacta.
fecha_oot = [pd.Timestamp("2022-09-30")]
es_oot = info["FECHA_CORTE"].isin(fecha_oot)
info["MUESTRA"] = np.where(es_oot, "TEST_OOT", "DEV")

chk = info[info["VarDep"].isin([0, 1])]
resumen = chk.groupby("MUESTRA").agg(N=("VarDep", "size"), tasa_malos=("VarDep", "mean"))
resumen["pct"] = resumen["N"] / resumen["N"].sum()
print(resumen)
print(pd.crosstab(info["FECHA_CORTE"], info["MUESTRA"]))

# Muestra de desarrollo (TEST_OOT se deja para el final).
dev = info[info["MUESTRA"] == "DEV"].copy()
print("DEV:", dev.shape)


# ============================================================================
# CREACIÓN DE VARIABLES — DISCRETIZACIÓN
# ============================================================================

# %% 11 — Framework de binning con árboles de decisión
# fit_arbol():   ajusta un árbol SOLO con DEV -> aprende los cortes (sin fuga).
# aplicar_arbol(): aplica una regla ya ajustada a cualquier muestra; devuelve el
#                  bin y prbm (= P(malo) por hoja). Los nulos van a un bin aparte.
# cortes():      devuelve los umbrales que aprendió el árbol.
def fit_arbol(cols, max_leaf_nodes=5, min_frac=0.05):
    # 'cols' = 1 o 2 columnas fuente (univariada o bivariada)
    base = dev[dev["VarDep"].isin([0, 1])]
    completos = base[cols].notna().all(axis=1)
    d = base.loc[completos]
    tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,
                                  min_samples_leaf=min_frac, random_state=13579)
    tree.fit(d[cols], d["VarDep"])
    faltantes = base.loc[~completos, "VarDep"]          # tasa de malos de los nulos
    p_na = faltantes.mean() if len(faltantes) else base["VarDep"].mean()
    return {"cols": cols, "tree": tree, "p_na": float(p_na),
            "tasa_global": float(base["VarDep"].mean())}


def aplicar_arbol(df, regla):
    cols = regla["cols"]
    completos = df[cols].notna().all(axis=1)
    bin_ = pd.Series(-1, index=df.index, dtype="int64")          # -1 = bin de nulos
    prbm = pd.Series(regla["p_na"], index=df.index, dtype="float64")
    if completos.any():
        X = df.loc[completos, cols]
        bin_.loc[completos] = regla["tree"].apply(X)
        prbm.loc[completos] = regla["tree"].predict_proba(X)[:, 1]
    prbm = prbm.fillna(regla["tasa_global"])
    return bin_, prbm


def cortes(nombre):
    t = reglas[nombre]["tree"].tree_
    es_hoja = t.children_left == t.children_right
    return sorted(np.round(t.threshold[~es_hoja], 4).tolist())


# %% 12 — Variables a discretizar y ajuste de las reglas
# Diccionario nombre -> columnas fuente. Las 4 últimas son bivariadas: el árbol
# modela la interacción entre las dos variables.
VARIABLES = {
    "MAX_DVEN_SCE_6M":      ["MAX_DVEN_SCE_6M"],
    "PROM_VEN_SCE_6M":      ["PROM_VEN_SCE_6M"],
    "maySalVen24M269":      ["maySalVen24M269"],
    "NENT_VEN_SCE_24M":     ["NENT_VEN_SCE_24M"],
    "maySalVenD3M227":      ["maySalVenD3M227"],
    "califHisTitularY361":  ["califHisTitularY361"],
    "NOPE_XVEN_OP_3M":      ["NOPE_XVEN_OP_3M"],
    "NOPE_APERT_SCE_OP_3M": ["NOPE_APERT_SCE_OP_3M"],
    "PROM_NDI_SCE_36M":     ["PROM_NDI_SCE_36M"],
    "NOPE_VENC_OP_3M":      ["NOPE_VENC_OP_3M"],
    "MAX_DVEN_SCE_6M_y_SALDO_PROMEDIO_AHORRO": ["MAX_DVEN_SCE_6M", "SALDO_PROMEDIO_AHORRO"],
    "NOPE_VENC_OP_12_y_INGRESOS":              ["NOPE_VENC_OP_12M", "INGRESOS"],
    "edad_y_INGRESOS":                         ["edad", "INGRESOS"],
    "NOPE_NDI_OP_3MySalTotOpD383":             ["NOPE_NDI_OP_3M", "SalTotOpD383"],
}

# Se ajusta un árbol por variable, SOLO con DEV.
reglas = {nombre: fit_arbol(cols) for nombre, cols in VARIABLES.items()}
for nombre, r in reglas.items():
    print(f"{nombre:42s} bins={r['tree'].get_n_leaves()}  cortes={cortes(nombre)}")

# %% 13 — Construcción de features
# construir_features(): aplica a una muestra TODO el feature engineering:
#   - binning con árboles  -> bin_*, prbm_* (P malo), prbb_* (P bueno)
#   - variables dummy (reglas fijas fila a fila)
#   - capeo de atípicos (_c, umbrales fijos)
def construir_features(df):
    df = df.copy()

    # --- Binning con árboles (reglas ajustadas en DEV) ---
    for nombre, regla in reglas.items():
        b, prbm = aplicar_arbol(df, regla)
        df[f"bin_{nombre}"] = b
        df[f"prbm_{nombre}"] = prbm
        df[f"prbb_{nombre}"] = 1 - prbm

    # --- Variables dummy (deterministas) ---
    df["d_numOpsVencidas3M102_cast"] = np.where(df["numOpsVencidas3M102"] > 0, 1, 0)
    df["d_numOpsVencidas3M102_prem"] = np.where(df["numOpsVencidas3M102"] == 0, 1, 0)
    df["d_numOpsVencidas101_cast"] = np.where(df["numOpsVencidas101"] > 0, 1, 0)
    df["d_numOpsVencidas101_prem"] = np.where(df["numOpsVencidas101"] == 0, 1, 0)
    df["d_numOpeCarteraCastigadaTitular376_cast"] = np.where(
        df["numOpeCarteraCastigadaTitular376"] > 0, 1, 0)
    df["d_numOpeCarteraCastigadaTitular376_prem"] = np.where(
        df["numOpeCarteraCastigadaTitular376"] == 0, 1, 0)
    df["d_peorNivelRiesgoValorOpBanCooComD415_prem"] = np.where(
        df["peorNivelRiesgoValorOpBanCooComD415"] <= 1, 1, 0)
    df["d_peorNivelRiesgoValorOpBanCooComD415_cast"] = np.where(
        df["peorNivelRiesgoValorOpBanCooComD415"] > 1, 1, 0)
    df["d_NOPE_NDI_OP_3M_cast"] = np.where(df["NOPE_NDI_OP_3M"] > 0, 1, 0)

    # --- Capeo de atípicos (umbrales fijos) ---
    df["califHisTitularV360_c"] = np.where(df["califHisTitularV360"] > 23, 23, df["califHisTitularV360"])
    df["MAX_DVEN_SF_OP_12M_c"] = np.where(df["MAX_DVEN_SF_OP_12M"] > 270, 270, df["MAX_DVEN_SF_OP_12M"])
    df["numOpsVig092_c"] = np.where(df["numOpsVig092"] > 59, 59, df["numOpsVig092"])
    df["MAX_DVEN_SCE_36M_c"] = np.where(df["MAX_DVEN_SCE_36M"] > 1583.54, 1583.54, df["MAX_DVEN_SCE_36M"])
    df["MAX_DVEN_SCE_6M_c"] = np.where(df["MAX_DVEN_SCE_6M"] > 360, 360, df["MAX_DVEN_SCE_6M"])
    df["SALDO_PROMEDIO_AHORRO_c"] = np.where(
        df["SALDO_PROMEDIO_AHORRO"] > 10500.9854, 10500.9854, df["SALDO_PROMEDIO_AHORRO"])
    df["NOPE_APERT_SCE_OP_24M_c"] = np.where(
        df["NOPE_APERT_SCE_OP_24M"] > 25, 25, df["NOPE_APERT_SCE_OP_24M"])
    df["NOPE_NDI_OP_24M_c"] = np.where(df["NOPE_NDI_OP_24M"] > 2, 2, df["NOPE_NDI_OP_24M"])
    df["SalTotOpD383_c"] = np.where(df["SalTotOpD383"] > 141419.8156, 2, df["SalTotOpD383"])
    df["NOPE_APERT_SF_OP_6M_c"] = np.where(df["NOPE_APERT_SF_OP_6M"] > 7, 7, df["NOPE_APERT_SF_OP_6M"])
    df["NOPE_APERT_SCE_OP_12M_c"] = np.where(
        df["NOPE_APERT_SCE_OP_12M"] > 15, 15, df["NOPE_APERT_SCE_OP_12M"])
    df["salOpVen014_c"] = np.where(df["salOpVen014"] > 8723.8756, 8723.8756, df["salOpVen014"])
    df["DEUDA_TOTAL_SCE_12M_c"] = np.where(
        df["DEUDA_TOTAL_SCE_12M"] > 1745712.081, 1745712.081, df["DEUDA_TOTAL_SCE_12M"])
    df["NOPE_REFIN_OP_12M_c"] = np.where(df["NOPE_REFIN_OP_12M"] >= 2, 2, df["NOPE_REFIN_OP_12M"])
    df["numOpsVencidas101_c"] = np.where(df["numOpsVencidas101"] >= 2, 2, df["numOpsVencidas101"])
    df["numOpsVencidas3M102_c"] = np.where(df["numOpsVencidas3M102"] >= 3, 3, df["numOpsVencidas3M102"])
    df["maySalVen24M269_c"] = np.where(
        df["maySalVen24M269"] >= 1881.4544, 1881.4544, df["maySalVen24M269"])
    return df.copy()


# Se aplica a DEV. TEST_OOT se discretiza luego, tras el feature selection.
dev = construir_features(dev)
print("DEV:", dev.shape)

# %% 14 — Diagnóstico de los bins
# reporte_bin(): por cada bin, N y tasa de malos en DEV -> sirve para revisar
# que la relación sea monótona (control de calidad de la discretización).
def reporte_bin(nombre):
    tr = dev[dev["VarDep"].isin([0, 1])]
    g = tr.groupby(f"bin_{nombre}").agg(N=("VarDep", "size"), tasa_malos=("VarDep", "mean"))
    g["prbm"] = tr.groupby(f"bin_{nombre}")[f"prbm_{nombre}"].first()
    return g


for nombre in reglas:
    print("===", nombre, "| cortes:", cortes(nombre))
    print(reporte_bin(nombre))

# Se guardan las reglas para aplicarlas idénticas a TEST_OOT más adelante.
with open("BDD/reglas_binning.pkl", "wb") as f:
    pickle.dump(reglas, f)
print("Reglas guardadas en BDD/reglas_binning.pkl")

# Lista de variables nuevas creadas.
nuevas = [c for c in dev.columns
          if c.startswith(("bin_", "prbm_", "prbb_", "d_")) or c.endswith("_c")]
print(len(nuevas), "variables creadas")


# ============================================================================
# FEATURE SELECTION  (esquema híbrido: filtro IV + wrapper RFECV)
# ============================================================================

# %% 15 — Pool de candidatas e Information Value
# Candidatas = las features construidas (prbm_, dummies, capadas _c).
feat_prbm = sorted(c for c in dev.columns if c.startswith("prbm_"))
feat_dummy = sorted(c for c in dev.columns if c.startswith("d_"))
feat_c = sorted(c for c in dev.columns if c.endswith("_c"))
candidatas = feat_prbm + feat_dummy + feat_c
print(f"Candidatas: {len(candidatas)}")

# iv_de(): IV de una columna cualquiera, calculado SOLO en DEV.
tr = dev[dev["VarDep"].isin([0, 1])]


def iv_de(col, bins=10):
    d = tr[[col, "VarDep"]].copy()
    if d[col].nunique(dropna=True) <= 10:
        d["g"] = d[col].astype("object").fillna("(NULO)")
    else:
        d["g"] = pd.qcut(d[col], bins, duplicates="drop")
        d["g"] = d["g"].cat.add_categories(["(NULO)"]).fillna("(NULO)")
    g = d.groupby("g", observed=True)["VarDep"].agg(N="count", malos="sum")
    g["buenos"] = g["N"] - g["malos"]
    pm = g["malos"] / g["malos"].sum()
    pb = g["buenos"] / g["buenos"].sum()
    eps = 1e-6
    return float(((pb - pm) * np.log((pb + eps) / (pm + eps))).sum())


# IV de todas las candidatas, ordenado de mayor a menor.
iv_tab = (pd.Series({c: iv_de(c) for c in candidatas}, name="IV")
            .sort_values(ascending=False))
print(iv_tab)

# %% 16 — Validación de cobertura
# Escaneo del IV de TODAS las variables crudas: detecta si alguna con IV alto
# quedó fuera de la discretización (lista curada de 14).
no_predictoras = {"VarDep", "ModVal", "MUESTRA", "DESEMPENO", "SIN_DESEMPENO",
                  "MARCA_BANCARIZADO", "MARCA_VENCIDO", "MAX_DIAS_MOROSIDAD"}
crudas = [c for c in dev.columns
          if pd.api.types.is_numeric_dtype(dev[c])
          and not c.startswith(("bin_", "prbm_", "prbb_", "d_"))
          and not c.endswith("_c") and c not in no_predictoras]

iv_crudas = {}
for c in crudas:
    try:
        if dev[c].nunique(dropna=True) > 1:
            iv_crudas[c] = iv_de(c)
    except Exception:
        pass
iv_crudas = pd.Series(iv_crudas, name="IV").sort_values(ascending=False)

ya_usadas = set(sum(VARIABLES.values(), []))      # crudas que ya alimentan features
faltantes = [c for c in iv_crudas[iv_crudas >= 0.10].index if c not in ya_usadas]
print(f"Variables crudas escaneadas: {len(iv_crudas)}")
print(f"Con IV >= 0.10 NO discretizadas: {len(faltantes)}")
print(iv_crudas.head(20))

# %% 17 — Filtro 1: Information Value
# Se descartan IV < 0.02 (sin poder) y se marcan IV > 0.5 (sospechosas de fuga).
IV_MIN, IV_MAX = 0.02, 0.50
filtro_iv = iv_tab[iv_tab >= IV_MIN].index.tolist()
sospechosas = iv_tab[iv_tab > IV_MAX].index.tolist()
print(f"Pasan el filtro IV: {len(filtro_iv)}")
print(f"Marcadas sospechosas (IV > {IV_MAX}): {sospechosas}")

# %% 18 — Filtro 2: Redundancia por correlación
# Entre dos variables con |corr| > 0.70 se conserva la de mayor IV.
UMBRAL_CORR = 0.70
corr = dev[filtro_iv].astype(float).corr().abs()
orden = [c for c in iv_tab.index if c in filtro_iv]    # de mayor a menor IV
descartar = set()
for i, a in enumerate(orden):
    if a in descartar:
        continue
    for b in orden[i + 1:]:
        if b in descartar:
            continue
        if corr.loc[a, b] > UMBRAL_CORR:
            descartar.add(b)
filtro_corr = [c for c in orden if c not in descartar]
print(f"Tras los filtros (IV + correlación): {len(filtro_corr)} variables")

# %% 19 — Wrapper: RFECV
# Recursive Feature Elimination con CV: entrena la logística, elimina la variable
# más débil, repite, y elige por validación cruzada el nº óptimo de variables.
# El CV se agrupa por cliente (StratifiedGroupKFold) para no tener fuga.
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

tr01 = dev[dev["VarDep"].isin([0, 1])]
y_w = tr01["VarDep"].astype(int)
grupos_w = tr01["IDENTIFICACION"].to_numpy()

# RFE elimina por magnitud del coeficiente -> primero imputar y escalar.
X_w = pd.DataFrame(
    StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(tr01[filtro_corr].astype(float))),
    columns=filtro_corr, index=tr01.index,
)

rfecv = RFECV(
    estimator=LogisticRegression(max_iter=1000, random_state=13579),
    step=1,
    cv=StratifiedGroupKFold(5, shuffle=True, random_state=13579),
    scoring="roc_auc",
    min_features_to_select=5,
    n_jobs=-1,
)
rfecv.fit(X_w, y_w, groups=grupos_w)

seleccion_final = [c for c, keep in zip(filtro_corr, rfecv.support_) if keep]
print(f"RFECV — nº óptimo de variables: {rfecv.n_features_}")
print("SELECCIÓN FINAL:", seleccion_final)

# Curva AUC (CV) vs número de variables.
scores = rfecv.cv_results_["mean_test_score"]
xs = list(range(rfecv.min_features_to_select, rfecv.min_features_to_select + len(scores)))
plt.figure(figsize=(8, 4))
plt.errorbar(xs, scores, yerr=rfecv.cv_results_["std_test_score"], marker="o")
plt.axvline(rfecv.n_features_, color="red", ls="--", label=f"óptimo = {rfecv.n_features_}")
plt.xlabel("Número de variables")
plt.ylabel("AUC (CV)")
plt.title("RFECV — AUC vs número de variables")
plt.legend()
plt.tight_layout()
plt.show()

# Ranking de IV resaltando en azul las variables finalmente seleccionadas.
colores = ["#4c72b0" if c in seleccion_final else "#cccccc" for c in iv_tab.index]
plt.figure(figsize=(8, 6))
iv_tab[::-1].plot.barh(color=colores[::-1])
plt.axvline(IV_MIN, color="red", ls="--", lw=1, label=f"IV mínimo = {IV_MIN}")
plt.xlabel("Information Value")
plt.title("Ranking de IV  (azul = seleccionada)")
plt.legend()
plt.tight_layout()
plt.show()

# Se guarda la selección final.
with open("BDD/features_seleccionadas.pkl", "wb") as f:
    pickle.dump(seleccion_final, f)
print("Selección guardada:", len(seleccion_final), "variables")


# ============================================================================
# OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================================================

# %% 20 — Datos de modelado, pipeline y CV agrupado
# X/y = población de modelado (DEV, bueno/malo, con las variables seleccionadas).
# Pipeline = imputación -> escalado -> logística (se reajusta dentro de cada fold).
# cv_agrupado(): repeated stratified GROUP k-fold -> cada cliente en un solo fold.
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve, learning_curve

mod = dev[dev["VarDep"].isin([0, 1])]
X = mod[seleccion_final].astype(float)
y = mod["VarDep"].astype(int)
groups = mod["IDENTIFICACION"].to_numpy()
print("X:", X.shape, "| tasa de malos:", round(y.mean(), 4))

modelo = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("logit", LogisticRegression(max_iter=1000, random_state=13579)),
])


def cv_agrupado(X, y, groups, n_splits=10, n_repeats=5, semilla=13579):
    rng = np.random.RandomState(semilla)
    splits = []
    for _ in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=rng.randint(1_000_000))
        splits.extend(sgkf.split(X, y, groups))
    return splits


cv = cv_agrupado(X, y, groups)        # 10 folds x 5 repeticiones = 50 particiones
print("Folds de CV:", len(cv))

# %% 21 — Optimización de lambda
# Se prueba una rejilla de C (= 1/lambda); validation_curve da el AUC de CV de
# cada valor. Se elige el lambda con mayor AUC de validación.
Cs = np.logspace(-3, 2, 20)
lambdas = 1 / Cs
train_sc, val_sc = validation_curve(
    modelo, X, y, param_name="logit__C", param_range=Cs,
    scoring="roc_auc", cv=cv, n_jobs=-1)
train_m, val_m = train_sc.mean(axis=1), val_sc.mean(axis=1)
val_s = val_sc.std(axis=1)

idx = val_m.argmax()
mejor_C, mejor_lambda = Cs[idx], lambdas[idx]
print(f"λ óptimo = {mejor_lambda:.4g}  | AUC CV = {val_m[idx]:.4f} ± {val_s[idx]:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(lambdas, train_m, "o-", color="#4c72b0", label="AUC entrenamiento")
plt.plot(lambdas, val_m, "o-", color="#c44e52", label="AUC validación (CV)")
plt.fill_between(lambdas, val_m - val_s, val_m + val_s, color="#c44e52", alpha=0.2)
plt.axvline(mejor_lambda, color="black", ls="--", lw=1, label=f"λ óptimo = {mejor_lambda:.3g}")
plt.xscale("log")
plt.xlabel("λ  (fuerza de regularización)")
plt.ylabel("AUC")
plt.title("Optimización de λ — Regresión Logística L2")
plt.legend()
plt.tight_layout()
plt.show()

# %% 22 — Curva de aprendizaje y ajuste del modelo óptimo
# Curva de aprendizaje: AUC vs tamaño de entrenamiento (diagnóstico de sesgo/varianza).
modelo_opt = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("logit", LogisticRegression(C=mejor_C, max_iter=1000, random_state=13579)),
])
sizes, lc_tr, lc_val = learning_curve(
    modelo_opt, X, y, groups=groups,
    cv=StratifiedGroupKFold(5, shuffle=True, random_state=13579),
    scoring="roc_auc", train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)

plt.figure(figsize=(8, 5))
plt.plot(sizes, lc_tr.mean(axis=1), "o-", color="#4c72b0", label="AUC entrenamiento")
plt.plot(sizes, lc_val.mean(axis=1), "o-", color="#c44e52", label="AUC validación (CV)")
plt.fill_between(sizes, lc_val.mean(1) - lc_val.std(1),
                 lc_val.mean(1) + lc_val.std(1), color="#c44e52", alpha=0.2)
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("AUC")
plt.title("Curva de aprendizaje — Regresión Logística L2")
plt.legend()
plt.tight_layout()
plt.show()

# Modelo final ajustado en todo DEV con el lambda óptimo.
modelo_opt.fit(X, y)
print(f"Modelo final — λ = {mejor_lambda:.4g}, variables = {len(seleccion_final)}")

# %% 23 — Métricas por validación cruzada
# ks_statistic(): KS = máxima separación entre acumuladas de buenos y malos.
# cross_validate calcula varias métricas a la vez sobre el CV agrupado.
from sklearn.model_selection import cross_validate
from scipy.stats import ks_2samp


def ks_statistic(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    return ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic


def ks_scorer(estimator, X, y):       # scorer KS para cross_validate
    return ks_statistic(np.asarray(y), estimator.predict_proba(X)[:, 1])


scorers = {"AUC": "roc_auc", "Accuracy": "accuracy", "Precision": "precision",
           "Recall": "recall", "F1": "f1", "KS": ks_scorer}
res_cv = cross_validate(modelo_opt, X, y, cv=cv, scoring=scorers, n_jobs=-1)

tabla_cv = pd.DataFrame(
    {m: [res_cv[f"test_{m}"].mean(), res_cv[f"test_{m}"].std()] for m in scorers},
    index=["media", "desv"],
).T
tabla_cv.loc["Gini"] = [2 * tabla_cv.loc["AUC", "media"] - 1, 2 * tabla_cv.loc["AUC", "desv"]]
print(tabla_cv.round(4))


# ============================================================================
# COMPARACIÓN ESTADÍSTICA DE DOS TÉCNICAS
# ============================================================================

# %% 24 — Logística L2 vs Random Forest
# Se evalúan ambos modelos con LOS MISMOS folds y se compara el AUC por fold con
# un test de Wilcoxon pareado (H0: mismo desempeño).
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import wilcoxon

modelo_A = modelo_opt                                    # logística regularizada
modelo_B = Pipeline([                                    # Random Forest
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                  random_state=13579, n_jobs=-1)),
])
auc_A = cross_val_score(modelo_A, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
auc_B = cross_val_score(modelo_B, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"Logística L2 : AUC = {auc_A.mean():.4f} ± {auc_A.std():.4f}")
print(f"Random Forest: AUC = {auc_B.mean():.4f} ± {auc_B.std():.4f}")

# Test de Wilcoxon: si p < 0.05 la diferencia es estadísticamente significativa.
stat, p = wilcoxon(auc_A, auc_B)
print(f"Wilcoxon: p-valor = {p:.4g}")

plt.figure(figsize=(6, 4))
plt.boxplot([auc_A, auc_B])
plt.xticks([1, 2], ["Logística L2", "Random Forest"])
plt.ylabel("AUC por fold (CV)")
plt.title("Comparación de técnicas — AUC en validación cruzada")
plt.tight_layout()
plt.show()


# ============================================================================
# PIPELINES DE SKLEARN
# ============================================================================

# %% 25 — Transformer propio + pipeline completo
# FiltroIV: implementa el feature selection (filtro IV) como un transformer de
# sklearn, para poder meterlo DENTRO del pipeline (se reajusta por fold).
from sklearn.base import BaseEstimator, TransformerMixin


class FiltroIV(BaseEstimator, TransformerMixin):
    """Filtro de feature selection: conserva las columnas con IV >= umbral."""

    def __init__(self, umbral=0.02, bins=10):
        self.umbral = umbral
        self.bins = bins

    @staticmethod
    def _iv(x, y, bins):
        d = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)})
        if d["x"].nunique(dropna=True) <= 10:
            d["g"] = d["x"]
        else:
            d["g"] = pd.qcut(d["x"], bins, duplicates="drop")
        g = d.groupby("g", observed=True)["y"].agg(["count", "sum"])
        malos = g["sum"]
        buenos = g["count"] - malos
        pm, pb = malos / malos.sum(), buenos / buenos.sum()
        eps = 1e-6
        return float(((pb - pm) * np.log((pb + eps) / (pm + eps))).sum())

    def fit(self, X, y):                       # calcula el IV con el train del fold
        X = pd.DataFrame(X)
        self.iv_ = {c: self._iv(X[c], y, self.bins) for c in X.columns}
        self.seleccion_ = [c for c, v in self.iv_.items() if v >= self.umbral]
        return self

    def transform(self, X):                    # conserva solo las columnas elegidas
        return pd.DataFrame(X)[self.seleccion_]


# Pipeline completo: filtro IV -> imputación -> escalado -> logística.
pipe_final = Pipeline([
    ("filtro_iv", FiltroIV(umbral=0.02)),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("logit", LogisticRegression(C=mejor_C, max_iter=1000, random_state=13579)),
])

# Se parte de TODAS las candidatas: el filtro se aplica dentro del pipeline.
X_full = mod[candidatas].astype(float)
auc_pipe = cross_val_score(pipe_final, X_full, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"Pipeline completo — AUC CV = {auc_pipe.mean():.4f} ± {auc_pipe.std():.4f}")
pipe_final.fit(X_full, y)
print("Variables que pasaron el filtro IV:",
      len(pipe_final.named_steps["filtro_iv"].seleccion_))


# ============================================================================
# VISUALIZACIÓN CON t-SNE
# ============================================================================

# %% 26 — t-SNE del espacio de características
# Proyecta las variables seleccionadas a 2D (sobre una muestra) y colorea por
# clase: permite ver si buenos y malos forman regiones separables.
from sklearn.manifold import TSNE

n_muestra = 4000
rng = np.random.RandomState(13579)
idx = rng.choice(len(X), size=min(n_muestra, len(X)), replace=False)
X_s, y_s = X.iloc[idx], y.iloc[idx]

X_prep = StandardScaler().fit_transform(
    SimpleImputer(strategy="median").fit_transform(X_s))
emb = TSNE(n_components=2, perplexity=30, init="pca",
           random_state=13579).fit_transform(X_prep)

plt.figure(figsize=(7, 6))
for clase, color, lab in [(0, "#4c72b0", "Bueno"), (1, "#c44e52", "Malo")]:
    m = y_s.to_numpy() == clase
    plt.scatter(emb[m, 0], emb[m, 1], s=6, c=color, alpha=0.5, label=lab)
plt.legend()
plt.title("t-SNE del espacio de características (color = clase)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()


# ============================================================================
# EVALUACIÓN DEL MODELO — PERFORMANCE Y TEST OOT
# ============================================================================

# %% 27 — Preparar el TEST_OOT
# Se aplican a la cohorte 2022-09 las MISMAS reglas de binning (ajustadas en DEV);
# no se reajusta nada. Es la prueba fuera de tiempo.
with open("BDD/reglas_binning.pkl", "rb") as f:
    reglas = pickle.load(f)

test = info[info["MUESTRA"] == "TEST_OOT"].copy()
test = construir_features(test)
test01 = test[test["VarDep"].isin([0, 1])]
X_test = test01[seleccion_final].astype(float)
y_test = test01["VarDep"].astype(int)
print("TEST_OOT:", X_test.shape, "| tasa de malos:", round(y_test.mean(), 4))

# %% 28 — Métricas DEV (CV) vs TEST_OOT
# metricas(): calcula AUC, Gini, KS y métricas de umbral sobre una muestra.
# La tabla compara el desempeño de CV (DEV) contra el de TEST_OOT.
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, auc as _auc)


def metricas(modelo, X, y, umbral=0.5):
    y = np.asarray(y).astype(int)
    proba = modelo.predict_proba(X)[:, 1]
    pred = (proba >= umbral).astype(int)
    a = roc_auc_score(y, proba)
    return pd.Series({
        "AUC": a, "Gini": 2 * a - 1, "KS": ks_statistic(y, proba),
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
    })


comparacion = pd.DataFrame({
    "DEV (CV)": tabla_cv["media"],
    "TEST_OOT": metricas(modelo_opt, X_test, y_test),
})
comparacion["Δ (OOT − DEV)"] = comparacion["TEST_OOT"] - comparacion["DEV (CV)"]
print(comparacion.round(4))

# %% 29 — Curva ROC y curva KS en TEST_OOT
proba_dev = modelo_opt.predict_proba(X)[:, 1]
proba_oot = modelo_opt.predict_proba(X_test)[:, 1]

# ROC: DEV vs TEST_OOT superpuestas.
fpr_d, tpr_d, _ = roc_curve(y, proba_dev)
fpr_o, tpr_o, _ = roc_curve(y_test, proba_oot)
plt.figure(figsize=(6, 6))
plt.plot(fpr_d, tpr_d, color="#4c72b0", label=f"DEV  (AUC = {_auc(fpr_d, tpr_d):.3f})")
plt.plot(fpr_o, tpr_o, color="#c44e52", label=f"TEST_OOT  (AUC = {_auc(fpr_o, tpr_o):.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.title("Curva ROC — DEV vs TEST_OOT")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# KS: separación de las acumuladas de buenos y malos en TEST_OOT.
orden = np.argsort(proba_oot)
yb = (y_test.to_numpy()[orden] == 0)
ym = (y_test.to_numpy()[orden] == 1)
cum_b = np.cumsum(yb) / yb.sum()
cum_m = np.cumsum(ym) / ym.sum()
dif = np.abs(cum_m - cum_b)
ks_i = int(dif.argmax())
eje = np.arange(1, len(proba_oot) + 1) / len(proba_oot)
plt.figure(figsize=(7, 4))
plt.plot(eje, cum_b, color="#4c72b0", label="Buenos")
plt.plot(eje, cum_m, color="#c44e52", label="Malos")
plt.vlines(eje[ks_i], cum_b[ks_i], cum_m[ks_i], color="black", label=f"KS = {dif[ks_i]:.3f}")
plt.xlabel("Población ordenada por score")
plt.ylabel("Proporción acumulada")
plt.title("Curva KS — TEST_OOT")
plt.legend()
plt.tight_layout()
plt.show()

# %% 30 — Tabla de performance del scorecard
# tabla_performance(): divide la población en 10 bandas de score.
# Los cortes se DERIVAN en DEV (cortes=None) y esos MISMOS cortes se aplican a
# TEST_OOT (cortes=cortes_dev) -> como un scorecard real con grados fijos.
def tabla_performance(modelo, X, y, n_bandas=10, cortes=None):
    proba = modelo.predict_proba(X)[:, 1]            # P(malo)
    score = (1 - proba) * 1000                       # score: mayor = menor riesgo
    df = pd.DataFrame({"score": score, "y": np.asarray(y).astype(int)})

    if cortes is None:                               # DEV: se derivan los cortes
        _, cortes = pd.qcut(df["score"], n_bandas, retbins=True, duplicates="drop")
        cortes = np.array(cortes, dtype=float)
        cortes[0], cortes[-1] = -np.inf, np.inf      # cubrir extremos de TEST

    df["banda"] = pd.cut(df["score"], bins=cortes,
                         labels=range(1, len(cortes)), include_lowest=True)
    t = df.groupby("banda", observed=True).agg(
        score_min=("score", "min"), score_max=("score", "max"),
        N=("y", "size"), malos=("y", "sum"),
    ).sort_index()
    t["buenos"] = t["N"] - t["malos"]
    t["tasa_malos"] = t["malos"] / t["N"]
    t["pct_poblacion"] = t["N"] / t["N"].sum()
    t["malos_acum_%"] = t["malos"].cumsum() / t["malos"].sum()
    t["buenos_acum_%"] = t["buenos"].cumsum() / t["buenos"].sum()
    t["KS"] = (t["malos_acum_%"] - t["buenos_acum_%"]).abs()
    cols = ["score_min", "score_max", "N", "pct_poblacion", "buenos", "malos",
            "tasa_malos", "malos_acum_%", "buenos_acum_%", "KS"]
    return t[cols], cortes


# DEV: deriva los cortes; debe rank-ordenar (tasa de malos monótona).
perf, cortes_dev = tabla_performance(modelo_opt, X, y, n_bandas=10)
print("DEV — rank-ordering:", perf["tasa_malos"].is_monotonic_decreasing)
print(perf.round(4))

# TEST_OOT: usa los MISMOS cortes de DEV (en TEST, pct_poblacion != 10% = drift).
perf_oot, _ = tabla_performance(modelo_opt, X_test, y_test, cortes=cortes_dev)
print("TEST_OOT — rank-ordering:", perf_oot["tasa_malos"].is_monotonic_decreasing)
print(perf_oot.round(4))

# %% 31 — PSI del score (DEV vs TEST_OOT)
# Mide si la distribución del puntaje se mantiene estable fuera de tiempo
# (< 0.10 = estable).
psi_score = psi(proba_dev, proba_oot)
print(f"PSI del score (DEV vs TEST_OOT): {psi_score:.4f}")
