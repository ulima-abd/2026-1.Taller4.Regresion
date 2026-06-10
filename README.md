# Taller: Regresión con Apache Spark MLlib
## Curso: Analítica con Big Data

**Dataset:** `merged_players.csv` — Base de datos de jugadores de fútbol (Football Manager)

---

## Descripción del Dataset

El archivo `merged_players.csv` contiene información de más de 90,000 jugadores de fútbol con los siguientes tipos de atributos:

| Categoría | Columnas destacadas |
|-----------|-------------------|
| **Información general** | `Name`, `Age`, `Nat` (nacionalidad), `Position`, `Club` |
| **Físicos** | `Height`, `Weight`, `Pac` (velocidad), `Str` (fuerza), `Sta` (resistencia), `Agi` (agilidad) |
| **Técnicos** | `Tec`, `Dri` (regate), `Fin` (finalización), `Pas` (pase), `Fir` (control), `Cro` (centro) |
| **Mentales** | `Vis` (visión), `Dec` (decisión), `Det` (determinación), `Ldr` (liderazgo), `Bra` (valentía) |
| **Económicos** | `Transfer Value` (valor de transferencia) |

> **Nota:** Los atributos numéricos de habilidad tienen valores en el rango 1–20. Las columnas `Height` y `Weight` requieren limpieza de texto antes de usarse.

---

## Instrucciones Generales

1. Utiliza **PySpark** con el módulo `pyspark.ml` para todos los ejercicios.
2. Carga el dataset con `spark.read.csv(..., header=True, inferSchema=True)`.
3. Para cada ejercicio aplica la siguiente estructura mínima:
   - Carga y limpieza de datos
   - Construcción del pipeline (`VectorAssembler` + modelo)
   - División train/test (80/20)
   - Entrenamiento y predicción
   - Evaluación con `RegressionEvaluator` (RMSE y R²)
4. Comenta tu código explicando cada etapa.
5. Responde las **preguntas de reflexión** al final de cada ejercicio.

---

## Funciones necesarias

```python
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# Funciones de limpieza reutilizables
def parse_height(col_name):
    """Convierte altura de formato 5'9" a centímetros."""
    feet = F.regexp_extract(col_name, r"(\d+)'", 1).cast(DoubleType())
    inches = F.regexp_extract(col_name, r"'(\d+)", 1).cast(DoubleType())
    return (feet * 30.48 + inches * 2.54).alias("Height_cm")

def parse_weight(col_name):
    """Extrae el valor numérico del peso en kg."""
    return F.regexp_extract(col_name, r"(\d+)", 1).cast(DoubleType()).alias("Weight_kg")
```

---

## Ejercicios

---

### Ejercicio 1 — Regresión Lineal Simple: Predicción de Peso por Altura 🟢

**Contexto:** En el análisis de rendimiento deportivo, existe una relación conocida entre la altura y el peso de un atleta. Queremos construir un modelo básico que prediga el peso de un jugador a partir de su altura.

**Preprocesamiento requerido:**
- La columna `Height` viene en formato `"5'9""` — debes convertirla a centímetros (cm).
- La columna `Weight` viene en formato `"65 kg"` — debes extraer el valor numérico.

**Tareas:**
1. Limpia y convierte las columnas `Height` y `Weight` a valores numéricos.
2. Construye un modelo de regresión lineal simple usando `Height_cm` como única feature para predecir `Weight_kg`.
3. Evalúa el modelo con RMSE y R².
4. Grafica los valores reales vs. predichos (opcional, con matplotlib).

**Preguntas de reflexión:**
- ¿Qué valor de R² obtuviste? ¿Es un modelo útil con una sola variable?
- ¿Qué otras variables físicas podrían mejorar la predicción del peso?

---

### Ejercicio 2 — Regresión Lineal Múltiple: Predicción de Velocidad (`Pac`) 🟢

**Contexto:** La velocidad (`Pac`) es uno de los atributos más valorados en el fútbol moderno. Quieres predecir qué tan rápido es un jugador basándote en sus características físicas y de condición.

**Features:** `Age`, `Weight_kg`, `Acc` (aceleración), `Sta` (resistencia), `Agi` (agilidad)
**Target:** `Pac` (velocidad)

**Tareas:**
1. Selecciona y limpia las columnas necesarias. Elimina filas con valores nulos.
2. Usa `VectorAssembler` para combinar las features.
3. Aplica `StandardScaler` para normalizar las features antes del modelo.
4. Entrena un modelo `LinearRegression` y evalúa con RMSE y R².
5. Muestra los coeficientes del modelo e identifica la feature más influyente.

**Preguntas de reflexión:**
- ¿Cuál es el coeficiente más alto? ¿Tiene sentido futbolísticamente?
- ¿Mejora el R² con respecto al Ejercicio 1? ¿Por qué?

---

### Ejercicio 3 — Análisis de Correlación y Regresión: Predicción de Fuerza (`Str`) 🟢

**Contexto:** Antes de entrenar un modelo, es buena práctica analizar la correlación entre las features y el target. En este ejercicio explorarás qué variables se correlacionan más con la fuerza física de un jugador.

**Features candidatas:** `Height_cm`, `Weight_kg`, `Age`, `Jum` (salto), `Bal` (balance)
**Target:** `Str` (fuerza)

**Tareas:**
1. Calcula la correlación de Pearson entre cada feature candidata y `Str` usando `df.stat.corr()`.
2. Selecciona las **3 features con mayor correlación** para el modelo.
3. Entrena un modelo `LinearRegression` y evalúa con RMSE y R².
4. Compara brevemente los resultados con un modelo que usa las 5 features originales.

**Preguntas de reflexión:**
- ¿Cuáles fueron las 3 features más correlacionadas con `Str`?
- ¿El modelo con 3 features tiene un rendimiento comparable al de 5 features?

---

### Ejercicio 4 — Regresión con Variables Categóricas: Predicción de Técnica (`Tec`) 🟡

**Contexto:** La posición de un jugador influye en su desarrollo técnico. Un mediocampista central típicamente tiene más técnica que un defensa central. Queremos incorporar esta variable categórica a nuestro modelo.

**Features:** `Position` (categórica), `Age`, `Dri` (regate), `Fir` (primer toque), `Pas` (pase)
**Target:** `Tec` (técnica)

**Tareas:**
1. Limpia la columna `Position` eliminando caracteres especiales si es necesario.
2. Aplica `StringIndexer` para convertir `Position` en índice numérico.
3. Aplica `OneHotEncoder` al índice generado.
4. Construye el pipeline completo: `StringIndexer` → `OneHotEncoder` → `VectorAssembler` → `LinearRegression`.
5. Evalúa el modelo. Identifica qué posición (dummy variable) tiene el coeficiente más alto.

**Preguntas de reflexión:**
- ¿Cuál es la diferencia entre `StringIndexer` y `OneHotEncoder`? ¿Por qué no se puede usar directamente el índice numérico?
- ¿Qué posición predice mayor técnica en tu modelo?

---

### Ejercicio 5 — Regularización: Ridge vs. Lasso para Predecir Finalización (`Fin`) 🟡

**Contexto:** Cuando se tienen muchas features, los modelos pueden sobreajustarse. La regularización penaliza la complejidad del modelo. Lasso (L1) puede incluso eliminar features irrelevantes poniendo sus coeficientes en cero.

**Features:** Todos los atributos técnicos y mentales disponibles (`Tec`, `Dri`, `Pas`, `Dec`, `Vis`, `Ant`, `Cmp`, `OtB`, `Tea`, `Wor`, `Det`, `Bra`, `Bal`, `Agi`, `Str`, `Sta`, `Pac`, `Acc`, `Jum`, `Hea`, `Lon`, `Fre`, `Cor`)
**Target:** `Fin` (finalización)

**Tareas:**
1. Selecciona y limpia todas las features numéricas listadas.
2. Normaliza con `StandardScaler`.
3. Entrena **dos modelos**:
   - **Ridge:** `LinearRegression(regParam=0.1, elasticNetParam=0.0)`
   - **Lasso:** `LinearRegression(regParam=0.1, elasticNetParam=1.0)`
4. Compara RMSE y R² de ambos modelos.
5. Lista los coeficientes del modelo Lasso e identifica cuáles quedaron en 0.

**Preguntas de reflexión:**
- ¿Cuántas features eliminó Lasso (coeficiente = 0)?
- ¿Cuál modelo tiene mejor RMSE? ¿Cuál elegirías en producción y por qué?

---

### Ejercicio 6 — Árbol de Decisión: Predicción de Visión (`Vis`) 🟡

**Contexto:** Los modelos lineales asumen relaciones continuas entre variables. Un árbol de decisión puede capturar relaciones no lineales y umbrales discretos, lo que puede ser más adecuado para algunos atributos.

**Features:** `Pas` (pase), `Dec` (decisión), `Ant` (anticipación), `Tea` (trabajo en equipo), `Cmp` (compostura), `OtB` (movimiento sin balón), `Age`
**Target:** `Vis` (visión)

**Tareas:**
1. Construye el pipeline con `VectorAssembler` → `DecisionTreeRegressor`.
2. Entrena el modelo con `maxDepth=5`.
3. Evalúa con RMSE y R².
4. Repite el entrenamiento variando `maxDepth` en [3, 5, 8, 10] y compara los resultados en una tabla.
5. Compara el mejor árbol contra un modelo de regresión lineal con las mismas features.

**Preguntas de reflexión:**
- ¿Qué ocurre con el RMSE de entrenamiento vs. test cuando aumenta `maxDepth`?
- ¿En qué escenarios preferirías un árbol de decisión sobre una regresión lineal?

---

### Ejercicio 7 — Random Forest: Predicción de Determinación (`Det`) 🟡

**Contexto:** La determinación de un jugador es difícil de predecir porque depende de múltiples factores psicológicos interrelacionados. Un Random Forest, al combinar múltiples árboles, puede capturar mejor estas relaciones complejas.

**Features:** `Bra` (valentía), `Ldr` (liderazgo), `Wor` (capacidad de trabajo), `Prof` (profesionalismo), `Amb` (ambición), `Pres` (presión), `Cons` (consistencia), `Age`
**Target:** `Det` (determinación)

**Tareas:**
1. Construye el pipeline con `VectorAssembler` → `RandomForestRegressor(numTrees=50)`.
2. Entrena y evalúa con RMSE y R².
3. Extrae e imprime el vector `featureImportances`.
4. Muestra las features ordenadas de mayor a menor importancia.
5. Compara el resultado con un árbol de decisión simple.

**Preguntas de reflexión:**
- ¿Cuál es la feature más importante para predecir `Det`? ¿Tiene sentido intuitivamente?
- ¿Por qué Random Forest suele superar a un solo árbol de decisión?

---

### Ejercicio 8 — Gradient Boosting con Cross-Validation: Predicción de Liderazgo (`Ldr`) 🔴

**Contexto:** El Gradient Boosting entrena árboles de forma secuencial, donde cada árbol corrige los errores del anterior. Es uno de los algoritmos más potentes para regresión. Combinado con validación cruzada, podemos encontrar los mejores hiperparámetros de forma sistemática.

**Features:** `Det` (determinación), `Bra` (valentía), `Age`, `Caps` (partidos internacionales), `Wor`, `Prof`, `Amb`, `Vis`, `Dec`
**Target:** `Ldr` (liderazgo)

**Tareas:**
1. Construye el pipeline base: `VectorAssembler` → `GBTRegressor`.
2. Define una grilla de hiperparámetros con `ParamGridBuilder`:
   - `maxIter`: [10, 20, 50]
   - `maxDepth`: [3, 5]
3. Usa `CrossValidator` con 3 folds y RMSE como métrica.
4. Entrena y selecciona el mejor modelo.
5. Reporta los mejores hiperparámetros y el RMSE final en el conjunto de test.

**Preguntas de reflexión:**
- ¿Qué combinación de `maxIter` y `maxDepth` dio el mejor resultado?
- ¿Cuánto tiempo tardó la validación cruzada? ¿Cómo afecta el número de folds al tiempo de cómputo?

---

### Ejercicio 9 — Pipeline Completo con Preprocesamiento Real: Predicción de Valor de Transferencia 🔴

**Contexto:** El valor de mercado de un jugador depende de múltiples factores: sus habilidades técnicas, su edad, su posición y sus atributos físicos. Este ejercicio simula un caso de uso real donde el preprocesamiento de datos es tan importante como el modelo.

**Features:** `Age`, `Position` (categórica), `Pac`, `Tec`, `Fin`, `Pas`, `Str`, `Height_cm`, `Weight_kg`, `Det`, `Vis`, `Dec`
**Target:** `Transfer Value` (valor de transferencia en USD)

**Tareas:**
1. Parsea la columna `Transfer Value`:
   - Elimina el símbolo `$` y las comas.
   - Convierte sufijos: `"1.5M"` → `1500000`, `"500K"` → `500000`, `"0"` → `0`.
2. Filtra jugadores con `Transfer Value > 0` (los valores en `0$` son jugadores sin valoración de mercado).
3. Aplica transformación logarítmica al target: `log_value = log(Transfer Value + 1)`.
4. Construye el pipeline completo incluyendo encoding de `Position`.
5. Entrena un `RandomForestRegressor` y evalúa en escala logarítmica y original.

**Preguntas de reflexión:**
- ¿Por qué se aplica transformación logarítmica al target? ¿Qué problema resuelve?
- ¿Qué ocurre si no filtramos los jugadores con valor `$0`?

---

### Ejercicio 10 — Comparación Sistemática de Modelos: Predicción de Agilidad (`Agi`) 🔴

**Contexto:** En la práctica del Machine Learning, raramente se trabaja con un solo modelo. El objetivo de este ejercicio es evaluar y comparar cuatro modelos distintos de forma estructurada para seleccionar el mejor en función de múltiples métricas.

**Features:** `Bal` (balance), `Pac` (velocidad), `Acc` (aceleración), `Weight_kg`, `Height_cm`, `Age`, `Dri` (regate), `Fla` (creatividad)
**Target:** `Agi` (agilidad)

**Tareas:**
1. Preprocesa las features necesarias y normaliza con `StandardScaler`.
2. Entrena los siguientes cuatro modelos con los mismos datos de entrenamiento:
   - `LinearRegression`
   - `DecisionTreeRegressor(maxDepth=5)`
   - `RandomForestRegressor(numTrees=50)`
   - `GBTRegressor(maxIter=20)`
3. Evalúa todos los modelos en el conjunto de test calculando: **RMSE**, **MAE** y **R²**.
4. Construye una tabla comparativa con los resultados.
5. Selecciona el mejor modelo y justifica tu elección considerando tanto las métricas como el tiempo de entrenamiento.

**Tabla de resultados esperada:**

| Modelo | RMSE | MAE | R² | Tiempo (aprox.) |
|--------|------|-----|----|-----------------|
| Regresión Lineal | | | | |
| Árbol de Decisión | | | | |
| Random Forest | | | | |
| Gradient Boosting | | | | |

**Preguntas de reflexión:**
- ¿El modelo con mejor RMSE es siempre la mejor elección? ¿Qué otros factores considerar?
- ¿En qué casos preferirías un modelo más simple (regresión lineal) aunque tenga peor métrica?

---

*Curso Analítica con Big Data — ULima | Semestre 2026-1*
