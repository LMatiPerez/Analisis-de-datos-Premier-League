# ⚽ Trabajo de Simulación de la Premier League (2020-2023)

## Licenciatura en Análisis y Gestión de Datos (Trabajo Final Integrador)

Este proyecto aplica técnicas de **Modelos y Simulación** para analizar la variabilidad inherente a los resultados de la Premier League inglesa. A partir de datos históricos, se construye un modelo estocástico de goles por partido y se simula la temporada mediante el método de Monte Carlo para cuantificar la incertidumbre en la tabla de posiciones final.

---

## 🎯 Objetivos del Proyecto

- Construir y calibrar un **modelo estocástico** de goles por partido basado en la Distribución de Poisson ($\lambda$).
- Cuantificar la **ventaja de localía** y la **capacidad goleadora ($\lambda$)** de cada equipo a partir de los datos históricos.
- Implementar una **Simulación de Monte Carlo** (con $N \geq 1.000$ réplicas) para generar distribuciones de puntos finales para cada equipo.
- Analizar la **variabilidad** de los resultados y comparar las distribuciones simuladas con el desenlace real de la temporada de validación.

---

## 🛠️ Metodología y Herramientas

La simulación se basa en un enfoque de modelos de conteo (Poisson) integrado con un procedimiento Monte Carlo.

### 1. Modelo de Goles (Poisson)

La cantidad de goles anotados por un equipo en un partido se modela como una **variable aleatoria discreta** con distribución **Poisson ($\lambda$)**.

El parámetro de intensidad **$\lambda$ (lambda)**, que es la media de goles esperada, se estima a partir de los datos históricos de la liga, separando la **condición de local y visitante**.

### 2. Simulación de Monte Carlo

El modelo se implementa en un bucle de Monte Carlo:

- Se utiliza el _fixture_ real de la temporada de validación.
- En cada partido, los goles son generados por el generador de números pseudoaleatorios utilizando $\text{Poisson}(\lambda_{equipo})$.
- Se repite el proceso $N$ veces (réplicas) para obtener la **distribución de puntos** y la **probabilidad** de eventos (ej., superar 70 puntos).

### 3. Herramientas y Librerías

| Herramienta            | Función en el Proyecto                                                                               |
| :--------------------- | :--------------------------------------------------------------------------------------------------- |
| **Python**             | Lenguaje de programación principal.                                                                  |
| **Pandas**             | Carga, limpieza y manipulación de datos (`df_matches_limpio`).                                       |
| **NumPy**              | Generación eficiente de números aleatorios (muestreo de Poisson) y operaciones numéricas.            |
| **Statsmodels**        | (Opcional) Uso de **Modelos Lineales Generalizados (GLM)** para la estimación avanzada de $\lambda$. |
| **Matplotlib/Seaborn** | Visualización de resultados (Box Plots, Histogramas de Goles).                                       |

---

## 💾 Datos de Entrada (Fuente y Descripción)

Los datos históricos se obtuvieron de la plataforma Kaggle, recopilando resultados de la Premier League en el período 2020-2023.

### Enlace de la Fuente de Datos

[https://www.kaggle.com/datasets/sajkazmi/premier-league-matches/data](https://www.kaggle.com/datasets/sajkazmi/premier-league-matches/data)

### Descripción de las Variables Clave

| Variable    | Descripción                                                                                |
| :---------- | :----------------------------------------------------------------------------------------- |
| `date`      | La **fecha** del partido.                                                                  |
| `time`      | La **hora** del partido.                                                                   |
| `comp`      | La **competición** o torneo del partido.                                                   |
| `round`     | La **jornada o ronda** del partido.                                                        |
| `day`       | El **día de la semana** en que se jugó el partido.                                         |
| `venue`     | El **estadio** o sede del partido.                                                         |
| `result`    | El **resultado** del partido (ganó, perdió o empató).                                      |
| `gf`        | **Goles a favor** del equipo local.                                                        |
| `ga`        | **Goles en contra** del equipo local (goles anotados por el equipo visitante).             |
| `opponent`  | El **oponente** del equipo local.                                                          |
| `xg`        | **Goles Esperados (Expected Goals)** a favor del equipo local.                             |
| `xga`       | **Goles Esperados (Expected Goals)** en contra del equipo local (XG del equipo visitante). |
| `poss`      | La **posesión de balón** del equipo local.                                                 |
| `captain`   | El **capitán** del equipo local.                                                           |
| `formation` | La **formación táctica** del equipo local.                                                 |
| `referee`   | El **árbitro** del partido.                                                                |
| `sh`        | Los **disparos o tiros totales** del equipo local.                                         |
| `sot`       | Los **disparos al arco** o tiros a puerta del equipo local.                                |
| `dist`      | La **distancia promedio** de los disparos del equipo local.                                |
| `fk`        | Los **tiros libres (free kicks)** del equipo local.                                        |
| `pk`        | Los **penaltis convertidos** del equipo local.                                             |
| `pka`       | Los **penaltis intentados** por el equipo local.                                           |
| `season`    | El **año de la temporada** del partido.                                                    |
| `team`      | El **equipo local**.                                                                       |
