> **🤖 Prompt de asistencia de IA**
>
> *"Crea un archivo markdown con un resumen general del proyecto, pensado para
> introducir al profesor: una visión global de alto nivel de la que puedan surgir
> preguntas, de modo que, si quiere ver el detalle explícito, se remita a los
> notebooks"*

---

# Proyecto Final — Modelo de Credit Scoring

*Aprendizaje Supervisado · Regresión Logística Regularizada*

Resumen ejecutivo del proyecto. Es una **visión general**; el detalle, el código y los
resultados completos están en los notebooks (ver sección 8).

---

## 1. El problema

El **riesgo de crédito** —la posibilidad de que un cliente no pague— es la principal
fuente de pérdidas de una institución financiera. Este proyecto construye y valida un
**modelo de credit scoring**: estima la probabilidad de incumplimiento de cada
solicitante a partir de su historial de buró y lo clasifica como **bueno / malo**.

**Importancia:** reduce las pérdidas esperadas, vuelve la decisión **objetiva y
auditable** (requisito regulatorio) y habilita la **inclusión financiera**.

## 2. Los datos

- **~79.591 operaciones** de crédito, en **5 cohortes trimestrales** (2021-09 a 2022-09).
- Fuente: **buró de crédito** — comportamiento crediticio + perfil del cliente.
- Variable objetivo `VarDep`: **bueno (0) / malo (1)**. Se modela solo esa población —
  **41.393 registros (52%)**; el 48% restante son exclusiones de diseño
  (indeterminados, vencidos, sin desempeño, no bancarizados).
- Tasa de malos: **40,8%**.

## 3. Estructura del proyecto

| Notebook | Contenido |
|---|---|
| `01_Creación Base de datos` | Integración de fuentes del buró y creación de ~2.500 variables |
| `02_VariableDependiente` | Definición de la etiqueta bueno/malo según la ventana de desempeño |
| `03_Modelamiento` | EDA, feature selection, modelo y validación — **cubre la rúbrica** |

## 4. Pasos de la rúbrica (todos en `03_Modelamiento`)

| Paso | Qué se hizo |
|---|---|
| Introducción | Motivación del problema de riesgo de crédito |
| **EDA** | Población, calidad, WoE / Information Value, estabilidad temporal |
| Feature Extraction | *No aplica* — datos tabulares estructurados |
| **Feature Selection** | Esquema **híbrido**: filtro por IV + correlación, y wrapper RFECV |
| **Hiperparámetros** | Optimización de λ con repeated k-fold CV; curva de aprendizaje |
| **Comparación** | Logística regularizada vs Random Forest, test de Wilcoxon |
| **Pipelines** | `Pipeline` de sklearn con el filtro de selección embebido |
| **t-SNE** | Visualización 2D del espacio de características |
| **Evaluación final** | Tabla de performance y prueba *out-of-time* sobre `TEST_OOT` |
| Conclusiones / Referencias | Cierre y bibliografía en formato IEEE |

## 5. Decisiones metodológicas clave

- **Partición temporal:** `DEV` (cohortes 2021-09 a 2022-06) para todo el desarrollo y
  `TEST_OOT` (cohorte 2022-09) reservado como prueba fuera de tiempo.
- **Sin fuga de datos:** la discretización, la selección y los hiperparámetros se
  estiman **solo en `DEV`**; la validación cruzada se **agrupa por cliente**
  (`StratifiedGroupKFold`), así un mismo cliente nunca entrena y valida a la vez.
- **Discretización con árboles de decisión:** cada variable se convierte en bins con
  su probabilidad de malo (estilo WoE), tratando los nulos como grupo propio.
- **Feature selection híbrido:** filtro rápido por *Information Value* (estándar en
  scoring) seguido de un *wrapper* (RFECV) que considera el modelo completo.

## 6. Resultados principales

- De **14 variables** discretizadas se llegó a **18 features** seleccionadas.
- Modelo final: **regresión logística regularizada L2**, λ óptimo ≈ 162.
- Desempeño en validación cruzada: **AUC 0,891 · Gini 0,782 · KS 0,637**
  (Accuracy 0,83 · Recall 0,72 · Precision 0,83) — cifras **fuertes** para scoring.
- **Comparación:** Random Forest (AUC 0,8947) supera a la logística (0,8908) de forma
  *estadísticamente significativa* pero con una **diferencia mínima** (+0,004); se
  conserva la logística por su **interpretabilidad**.
- **Estabilidad temporal:** el PSI entre cohortes es < 0,03 → distribuciones estables.
- La **tabla de performance** del scorecard **rank-ordena** correctamente: la tasa de
  malos decrece de forma monótona entre las bandas de score.
- **Validación out-of-time:** en `TEST_OOT` (cohorte 2022-09, nunca vista) el modelo
  **mantiene el desempeño** — AUC 0,889 (vs 0,891 en CV), KS 0,621, **PSI del score
  0,018** — y la tabla de performance sigue rank-ordenando: el modelo es **estable
  fuera de tiempo**.

## 7. Puntos abiertos para la discusión

- Algunas variables de mora alcanzan un IV muy alto (> 0,5, hasta ≈ 2): conviene
  confirmar que son estrictamente **previas** al punto de observación (riesgo de fuga).
- Trade-off interpretabilidad vs. desempeño: logística regularizada frente a modelos
  de ensamble (Random Forest gana, pero por un margen mínimo).

## 8. Dónde ver el detalle

- **Construcción de datos y variables:** `01_Creación Base de datos.ipynb`
- **Definición de bueno/malo:** `02_VariableDependiente.ipynb`
- **EDA, modelo y validación:** `03_Modelamiento.ipynb`
- **Rúbrica del proyecto:** `Libro1.xlsx`

> Las consultas de asistencia de IA usadas en cada paso están documentadas dentro de
> `03_Modelamiento.ipynb`, en las celdas marcadas con 🤖.
