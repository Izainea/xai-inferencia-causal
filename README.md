# Inferencia Causal + IA Explicativa (XAI)
### Cuadernos experimentales reproducibles con datos abiertos

Material de apoyo del artículo **«De la correlación a la causa: inteligencia artificial explicable e inferencia causal»** (*Revista Comunicaciones en Estadística*). Reúne diez cuadernos reproducibles que desarrollan, con experimentos sobre datos abiertos, la relación entre la explicabilidad de modelos y la inferencia causal. El material se emplea además como apoyo de un espacio académico de posgrado homónimo.

> **Tesis.** Un modelo puede ser explicable y, aun así, no informar sobre las causas del fenómeno. La explicabilidad hace transparente la **correlación** aprendida por el modelo; responder preguntas de intervención (*¿qué ocurriría si…?*) y contrafactuales requiere el formalismo de la **inferencia causal**. La serie desarrolla esa distinción de manera sistemática.

---

## Hilo conductor

La serie emplea **un mismo caso** —un modelo de *scoring* crediticio (German Credit)— de forma transversal:

1. **Sección 1 · Explicar (cuadernos 01–04).** Entrenamos una caja negra y la explicamos con SHAP/LIME y contrafactuales. Descubrimos que la "variable más importante" del modelo no es necesariamente una *causa*.
2. **Sección 2 · Entender (cuadernos 05–08).** Cambiamos de lente: resultados potenciales, DAGs y estimación causal. Usamos además el experimento **NSW/Lalonde** como "verdad de referencia" para ver cómo la confusión engaña a la intuición.
3. **Sección 3 · Actuar (cuadernos 09–10).** Integramos ambos mundos: efectos heterogéneos (CATE) explicados con SHAP, descubrimiento causal, recurso accionable y auditoría ética del modelo crediticio.

Cada cuaderno concluye con la cuestión que motiva el siguiente y dispone de su **diagrama de flujo** en `img/NN_flujo.svg`.

---

## Los diez cuadernos

### 🟦 Sección 1 — IA Explicativa (XAI) y sus límites

**`01_fundamentos_explicabilidad_y_causalidad.ipynb`**
*Pregunta:* ¿qué significa "explicar" una predicción y por qué no es lo mismo que "entender"?
Historia y filosofía de la explicación (Hempel, Salmon, Pearl). Taxonomía de XAI: *ante-hoc* vs *post-hoc*, local vs global, específico vs agnóstico. **Experimentos:** simulamos correlaciones espurias, **confusión** y la **paradoja de Simpson** para sentir en los datos por qué "importante para el modelo" ≠ "causa". Primer contacto con German Credit. *Apoyo:* Moraffah et al. (2020); Molnar (2022); Carloni et al. (2023). *Puente:* si el modelo solo ve correlaciones, ¿qué tan lejos llega su explicación?

**`02_explicaciones_locales_shap_lime.ipynb`**
*Pregunta:* ¿por qué este solicitante fue rechazado?
Entrenamos una caja negra (Gradient Boosting/XGBoost) sobre German Credit. **Experimentos:** KernelSHAP y TreeSHAP, valores de Shapley desde la teoría de juegos, *force plots*, *waterfall*, *beeswarm*; LIME y sus vecindarios locales; comparación SHAP vs LIME y análisis de **estabilidad** de las explicaciones. Discusión de la matriz de costos asimétrica. *Apoyo:* Molnar (2022); Lundberg & Lee (2017). *Puente:* explicaciones locales coherentes… pero, ¿reflejan el mundo o solo el modelo?

**`03_explicaciones_globales_y_contrafactuales.ipynb`**
*Pregunta:* ¿qué tendría que cambiar para que la decisión fuera distinta?
Importancia por permutación, **PDP/ICE/ALE**, modelos sustitutos (*surrogate*), una mirada a los **mecanismos de atención**. Luego, **explicaciones contrafactuales** con DiCE: el menor cambio que voltea la predicción, con nociones de validez, dispersión y *accionabilidad*. *Apoyo:* Verma et al. (2021); Chou et al. (2022). *Puente:* un contrafactual sin modelo causal puede sugerir cambios imposibles o engañosos.

**`04_limites_causales_de_la_explicabilidad.ipynb`**  ⟵ *el puente S1→S2*
*Pregunta:* ¿dónde y por qué falla SHAP cuando hay dependencia entre variables?
**Experimentos:** construimos un proceso generador de datos con estructura causal conocida y mostramos cómo SHAP **reparte mal** la importancia bajo correlación; comparamos con **Causal SHAP** y **do-SHAP** (operador *do* de Pearl). Visualizamos la diferencia entre atribución observacional e intervencional. *Apoyo:* Ng et al. (2025); Parafita et al. (2025). *Puente:* necesitamos formalizar la causalidad → Sección 2.

### 🟩 Sección 2 — Fundamentos de la inferencia causal

**`05_resultados_potenciales_y_contrafactuales.ipynb`**
*Pregunta:* ¿cuál es el efecto de una intervención si solo observamos uno de los dos mundos posibles?
Marco de **Rubin**: resultados potenciales, el "problema fundamental" de la inferencia causal, ATE/ATT/ATC, sesgo de selección, *ignorabilidad* y *overlap*. **Experimentos:** sobre **NSW** estimamos el ATE experimental (≈ \$886) y mostramos por qué la aleatorización lo hace válido. *Apoyo:* Pearl, Glymour & Jewell (2016); Weinberg et al. (2024). *Puente:* sin aleatorización, ¿cómo decidimos qué ajustar?

**`06_dags_y_razonamiento_grafico.ipynb`**
*Pregunta:* ¿qué variables debo (y no debo) controlar?
DAGs con `networkx`: caminos, **d-separación**, **confusores**, **mediadores**, **colisionadores** y el peligroso *sesgo del colisionador*; criterios **backdoor** y **frontdoor**; variables latentes. **Experimentos:** simulaciones donde "controlar de más" introduce sesgo (¡condicionar en un collider!). *Apoyo:* Pearl & Mackenzie (2018); Pearl et al. (2016). *Puente:* con el DAG correcto, ¿cómo estimamos el efecto en la práctica?

**`07_estimacion_causal_con_dowhy.ipynb`**
*Pregunta:* ¿cómo paso de un DAG a un número con garantías?
El flujo de **DoWhy** en cuatro pasos: *modelar → identificar → estimar → refutar*. **Experimentos:** sobre NSW (y su versión observacional con controles CPS/PSID) estimamos el efecto y aplicamos **pruebas de refutación** (placebo, causa común aleatoria, subconjuntos, *unobserved confounder*). *Apoyo:* Sharma & Kiciman (DoWhy); Pearl et al. (2016). *Puente:* ¿y si un solo estimador no basta?

**`08_metodos_de_estimacion_psm_ipw_iv_did.ipynb`**
*Pregunta:* ¿qué herramienta uso según el diseño de los datos?
Recorrido profundo por **emparejamiento por puntaje de propensión (PSM)**, **ponderación inversa (IPW)**, **variables instrumentales (IV)** y **diferencias en diferencias (DiD)**. **Experimentos:** replicamos la famosa comparación de LaLonde (experimental vs observacional) y comparamos estimadores. *Apoyo:* Pearl et al. (2016); Dehejia & Wahba (1999). *Puente:* hasta aquí, efectos *promedio*; ¿y la heterogeneidad entre individuos?

### 🟨 Sección 3 — Integración, herramientas y ética

**`09_efectos_heterogeneos_econml_y_xai.ipynb`**  ⟵ *donde XAI y causalidad se encuentran*
*Pregunta:* ¿a quién beneficia más la intervención, y por qué?
**Efectos heterogéneos (CATE)** con **EconML**: *meta-learners* (S/T/X-learner), **Double Machine Learning** y **Causal Forest**. El giro integrador: **explicamos el CATE con SHAP**, es decir, usamos XAI sobre un objeto *causal*, no sobre una predicción. *Apoyo:* Jiao et al. (2024); Künzel et al. (2019). *Puente:* si ya sabemos el efecto y a quién, ¿cómo lo comunicamos de forma accionable y justa?

**`10_causalidad_explicable_y_auditoria_etica.ipynb`**  ⟵ *cierre del caso crediticio*
*Pregunta:* ¿cómo entregar un modelo explicable, causal, accionable y auditable?
De extremo a extremo sobre German Credit: **descubrimiento causal** (PC / NOTEARS con `causal-learn`), **recurso contrafactual accionable** (DiCE guiado por el grafo), y **auditoría ética**: sesgo y equidad por grupo, **equidad contrafactual**, transparencia y rendición de cuentas. Espejo metodológico de Takahashi et al. (2024) y de los casos de política pública/salud. *Apoyo:* Takahashi et al. (2024); Svetovidov et al. (2021); Mangharamani & Vashishtha (2025); Chou et al. (2022).

---

## Estructura del repositorio

```
inferencia-causal-xai/
├── README.md                 # este archivo (mapa del recorrido)
├── requirements.txt          # dependencias (Python 3.12)
├── data/
│   ├── README.md             # diccionario y fuentes de los datos
│   ├── raw/                  # crudos descargados (german.*, nsw_*)
│   └── processed/            # derivados generados por los cuadernos
├── notebooks/                # los 10 cuadernos del recorrido
├── src/
│   └── data.py               # cargadores reproducibles (German Credit, NSW)
├── img/                      # diagramas de flujo en SVG (uno por cuaderno)
└── reports/figures/          # figuras exportadas para el artículo y el curso
```

## Puesta en marcha

```bash
# 1) Entorno dedicado con Python 3.12 (DoWhy/EconML/causal-learn no soportan 3.14)
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell: .venv\Scripts\Activate.ps1)
python -m pip install -U pip
pip install -r requirements.txt

# 2) Descargar los datos abiertos
python src/data.py              # imprime un diagnóstico si todo está bien

# 3) Abrir el recorrido
jupyter lab
```

> **Nota de entorno.** `pandas`, `scikit-learn`, `shap` y `lime` funcionan en Python 3.14, pero **DoWhy, EconML y causal-learn** (cuadernos 05–10) requieren **3.12**. Por eso fijamos 3.12 para todo el repo y evitamos sorpresas.

## Reproducibilidad
- Semillas fijas en cada cuaderno (`RANDOM_STATE = 42`).
- Los crudos se descargan con `src/data.py`; nada de datos manuales.
- Cada figura del artículo se exporta a `reports/figures/`.

## Bibliografía
Las referencias completas (APA 7) están en la carpeta hermana `../referencias metodológicas/` (con los PDF) y se reutilizan en el `bibliografia.bib` del artículo.
