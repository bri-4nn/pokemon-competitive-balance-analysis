# Pokémon Analytics: Balanceo Competitivo y Descubrimiento de Roles

Análisis de datos y modelado predictivo sobre el [dataset de Pokémon de Rounak Banik](https://www.kaggle.com/datasets/rounakbanik/pokemon), simulando el rol de Data Analyst / Data Scientist para el equipo de balanceo de un videojuego competitivo.

## 🎯 Objetivo de negocio

> Optimizar el balanceo del juego y descubrir ventajas estadísticas para la composición de equipos competitivos mediante el análisis de atributos y el modelado de datos.

## 📊 Dataset

- **Fuente:** [Kaggle - Pokemon Dataset (Rounak Banik)](https://www.kaggle.com/datasets/rounakbanik/pokemon)
- **Cobertura:** Generaciones 1 a 7 (~800 Pokémon)
- **Variables utilizadas:** estadísticas base de combate (HP, Attack, Defense, Sp. Attack, Sp. Defense, Speed), tipos elementales, tasa de captura, pasos de incubación, altura, peso, generación y estatus legendario.

## ❓ Preguntas de negocio (EDA)

| # | Pregunta | Hallazgo clave |
|---|---|---|
| 1 | **Power Creep:** ¿Aumenta el poder promedio (Base Stat Total) con las generaciones? | Salto abrupto entre la 3ra y 4ta generación, ligado a la introducción de legendarios y megaevoluciones — no es un incremento lineal en el tiempo. |
| 2 | **Sinergia y Tipología:** ¿Qué combinaciones de tipos ofrecen mejor viabilidad y balance defensivo? | Se identificaron combinaciones de tipos con alto promedio y baja dispersión entre HP/Defense/Sp.Defense, filtrando por un mínimo de 3 Pokémon por combinación para evitar ruido estadístico. |
| 3 | **Economía del juego:** ¿Correlacionan `capture_rate`, `base_egg_steps` y `base_total`? | Sí, fuertemente: a mayor poder, mayor dificultad de captura y crianza — el sistema de captura funciona como mecanismo de balanceo económico. |
| 4 | **Roles ocultos:** ¿Qué tipos dominan como atacantes rápidos vs. defensores lentos? | Clasificación por cuadrantes (mediana de Speed vs. máximo de Attack/Sp.Attack) reveló qué tipos elementales tienden a definir el ritmo del meta-juego. |

## 🤖 Modelado

### Modelo 1 — Clasificación: ¿Es legendario? (Random Forest)

Se auditó si el diseño estadístico de una criatura la acerca al perfil de un Pokémon legendario.

**Iteración 1 — Detección de Data Leakage:**
Un primer modelo con `capture_rate` y `base_egg_steps` como features arrojó accuracy = 1.0 en todas las métricas. Se identificó que estas variables son casi una definición directa de "legendario" por diseño del juego (capture_rate mínimo y base_egg_steps máximo fijos para legendarios), no atributos predictivos genuinos.

**Iteración 2 — Modelo corregido:**
Reentrenado solo con `base_total`, `height_m` y `weight_kg`:

| Métrica | Valor |
|---|---|
| Accuracy | 0.957 |
| Precision | 0.824 |
| Recall | 0.778 |
| F1 Score | 0.800 |
| ROC AUC | 0.957 |

Dado el objetivo de negocio, el **recall** es la métrica prioritaria (un legendario "roto" mal clasificado es más costoso que una falsa alarma).

### Modelo 2 — Clustering: Arquetipos competitivos (K-Means)

Segmentación no supervisada usando únicamente las 6 stats de combate (estandarizadas), sin depender de tipos elementales.

- **K óptimo = 5**, seleccionado con método del codo + Silhouette Score.
- Arquetipos identificados: **Débiles/early-game**, **Wall físico**, **Sweeper rápido**, **Tanque bulky (HP)**, **Élite/pseudo-legendario**.
- Validado cualitativamente con Pokémon icónicos (Mewtwo, Ninetales, Slaking, Shuckle, Blissey).
- **Limitación identificada:** K-Means agrupa mejor perfiles "balanceadamente extremos" que "especialistas puros" (ej. Blissey, con HP altísimo pero Attack/Defense muy bajos) — una limitación esperada del algoritmo por su supuesto de clusters esféricos.

## 🔑 Lección metodológica transversal

Ambos modelos compartieron un mismo principio: **los resultados "demasiado perfectos" no deben tomarse a valor nominal.** Un accuracy de 1.0 resultó ser leakage; el mejor Silhouette Score (en K=2) resultó ser una simplificación excesiva del problema. Detectar y corregir ambos casos fue tan valioso como los resultados finales.

## 🛠️ Stack técnico

- **Python:** `pandas`, `numpy`
- **Visualización:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (RandomForestClassifier, KMeans, StandardScaler, PCA)

## 📁 Estructura del proyecto

```
├── pokemon_analysis.ipynb   # Notebook completo: limpieza, EDA, modelado
└── README.md                # Este archivo
```

## 🚀 Cómo ejecutar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
jupyter notebook pokemon_analysis.ipynb
```

## 📈 Próximos pasos

- Aplicar `SMOTE` u otras técnicas de balanceo de clases para mejorar el recall del modelo de clasificación.
- Probar `DBSCAN` o `Gaussian Mixture Models` para capturar mejor a los "especialistas puros" en el clustering.
- Incorporar generaciones más recientes (el dataset actual llega hasta la Gen 7) para validar si la tendencia de Power Creep se mantiene.

## 👤 Autor

*Tu nombre aquí* — [LinkedIn](#) · [Portafolio](#)
