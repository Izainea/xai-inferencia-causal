# Datos abiertos del proyecto

Dos conjuntos, descargables con `python src/data.py` (los crudos van en `data/raw/`).

## 1. German Credit — *Statlog (German Credit Data)* · UCI
- **Fuente:** UCI Machine Learning Repository, donado por H. Hofmann (Universität Hamburg).
- **URL:** https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- **Archivos:** `german.data` (1000 × 21, códigos `A1x`), `german.data-numeric`, `german.doc` (diccionario oficial).
- **Unidad:** un solicitante de crédito. **Objetivo:** `riesgo` (1 = buen pagador, 2 = mal pagador) → recodificado a `riesgo_bueno` (1/0).
- **Rol en el proyecto:** **caso guía transversal** (scoring crediticio). Aparece en las tres secciones: se *explica* (S1), se *modela causalmente* (S2) y se *audita* (S3). Reproduce el espíritu de Takahashi et al. (2024).
- **Variable sensible derivada:** `sexo` (a partir de `estado_personal_sexo`), usada en la auditoría de equidad del cuaderno 10.
- **Nota ética:** el costo de error es asimétrico (clasificar un mal pagador como bueno cuesta 5× lo contrario, según la matriz de costos oficial). Lo discutimos en los cuadernos 02 y 10.

### Variables (21)
| Columna | Tipo | Descripción |
|---|---|---|
| estado_cuenta | cat | Saldo de la cuenta corriente |
| duracion_meses | num | Duración del crédito (meses) |
| historial_credito | cat | Historial crediticio |
| proposito | cat | Destino del crédito |
| monto_credito | num | Monto solicitado (DM) |
| ahorros | cat | Cuenta de ahorros / bonos |
| empleo_desde | cat | Antigüedad en el empleo |
| tasa_cuota_pct | num | Cuota como % del ingreso disponible |
| estado_personal_sexo | cat | Estado civil y sexo |
| otros_deudores | cat | Codeudores / garantes |
| residencia_desde | num | Años en la residencia actual |
| propiedad | cat | Tipo de propiedad |
| edad | num | Edad (años) |
| otros_planes_pago | cat | Otros planes de financiación |
| vivienda | cat | Régimen de vivienda |
| n_creditos_banco | num | Créditos vigentes en el banco |
| trabajo | cat | Calificación laboral |
| n_dependientes | num | Personas a cargo |
| telefono | cat | Tiene teléfono registrado |
| trabajador_extranjero | cat | Trabajador extranjero |
| riesgo | objetivo | 1 = bueno, 2 = malo |

## 2. NSW / Lalonde — *National Supported Work Demonstration* · NBER
- **Fuente:** experimento aleatorizado (1976–1977) sobre un programa de empleo subsidiado; popularizado por LaLonde (1986) y Dehejia & Wahba (1999, 2002).
- **URL:** https://users.nber.org/~rdehejia/data/ (también `nsw.dta`).
- **Archivos:** `nsw_dw_treated.txt`, `nsw_dw_control.txt` (muestra masculina original: 297 tratados + 425 controles = 722).
- **Unidad:** un participante. **Tratamiento:** `tratamiento` (1 = programa de empleo). **Resultado:** `re78` (ingresos reales 1978).
- **Por qué es el benchmark causal ideal:** al ser **aleatorizado**, la diferencia de medias da el ATE "verdadero" (≈ **\$886** en esta muestra). Luego se reemplaza el grupo de control por una muestra observacional (CPS/PSID) para mostrar cómo la **confusión** destroza esa estimación si se ignora — el corazón de las secciones 2 y 3.
- **Controles observacionales (desafío de LaLonde):** `cps_controls.txt` (15.992 obs) y `psid_controls.txt` (2.490 obs), descargados de NBER; traen `re74` además, y se alinean a las 9 columnas de NSW con `load_lalonde_obs("cps"|"psid")`. El ATE ingenuo cae a −\$8.870 (CPS) / −\$15.578 (PSID). Se usan en el cuaderno 05.

### Variables (9)
| Columna | Tipo | Descripción |
|---|---|---|
| tratamiento | 0/1 | Participó en el programa |
| edad | num | Edad |
| educacion | num | Años de educación |
| afroamericano | 0/1 | Afroamericano |
| hispano | 0/1 | Hispano |
| casado | 0/1 | Casado |
| sin_grado | 0/1 | Sin diploma de secundaria |
| re75 | num | Ingresos reales 1975 (pre-tratamiento) |
| re78 | num | Ingresos reales 1978 (resultado) |

## 3. Palmer Penguins — *palmerpenguins* (Horst, Hill & Gorman, 2020)
- **Fuente:** medidas morfológicas de 342 pingüinos antárticos (Palmer Station LTER, 2007–2009).
- **URL:** https://allisonhorst.github.io/palmerpenguins/ · `data/raw/penguins.csv`.
- **Rol en el proyecto:** demostración visual de la **paradoja de Simpson** en el cuaderno 01 (longitud vs. profundidad del pico: correlación agregada negativa, positiva dentro de cada especie).

> Licencias: estos conjuntos son de uso académico abierto. Citar las fuentes originales en el artículo y en el material del curso.
