# Idea anotada: mejorar los graves del DDSP-Piano con PianoBass + Salamander (y quizá un Steinway propio)

Fecha: 2026-11-16 · Estado: **idea, sin implementar** · Origen: el usuario nota que el grave de las octavas más bajas suena sintético y que el sample `PianoBass` (Samples_samples/Anibal/Piano) suena muy bien.

## Qué se sabe (medido)

| Dato | Valor | Fuente |
|---|---|---|
| `PianoBass` es una sola nota (La0, fundamental 26,9 Hz), 5,6 s, 44,1 kHz/24 bit, casi mono, normalizada a 0 dBFS | — | análisis de `PianoBass [2023-11-11 200718].wav` |
| Inarmonicidad B/fórmula de Rigaud | PianoBass 0,65 · Salamander La0 0,54 · DDSP con Bass Tuning 0,57 | tercera confirmación independiente (MAESTRO 0,42, Salamander 0,38–0,55) |
| En el La0 del DDSP el fundamental sale ~44 dB por encima de lo normal (re. máx. de k ≤ 4) | PianoBass −44 dB, Salamander −43/−45 dB, DDSP 0 dB | los dos pianos reales coinciden; el DDSP genera casi un seno grave |
| T60 de los parciales 1–8 | PianoBass 5–11 s · DDSP 9–13 s | del mismo orden |
| Teclas 24–30 (Salamander, v8): fundamental del DDSP 10–20 dB por encima | −17, −10, −20 dB | **ojo**: MAESTRO daba ~−2 dB en 24–35; la diferencia puede ser de grabación (Salamander: micros a 12 cm; MAESTRO: sala) |
| Datos de graves en MAESTRO 2009 (entrenamiento) | La0–Si0: 6 notas; 24–35: 1.516 (34 limpias) | la red casi no vio las teclas más bajas |

Cautelas: la comparación tecla a tecla de parciales altos es muy ruidosa (saltos de ±50 dB entre vecinas); solo es fiable el patrón del fundamental en La0. No se sabe si PianoBass está procesado ni qué piano es.

## Opciones (de menos a más trabajo)

1. **Referencia de medida (sin tocar el sonido):** usar PianoBass + La0/Do1/Re#1/Fa#1 de Salamander para fijar objetivos de inarmonicidad y de perfil de armónicos de los graves.
2. **Moldeado de graves en el motor (sin muestras):** corregir el nivel de los primeros parciales de las teclas más bajas (fundamental) hacia el perfil de los pianos reales, con un mando propio. Riesgo: depende de qué grabación se tome como objetivo (cerca vs. sala).
3. **Capa híbrida de graves (con muestras):** para las teclas ~21–32, mezclar o sustituir por una nota real (PianoBass + notas de Salamander con sus velocities), con crossfade hacia el DDSP. Necesita pocas notas (cada muestra cubre ±1–2 semitonos) y recortar tamaños (p. ej. 4 notas × 3 velocities × 8 s mono 16 bit ≈ 6 MB). Licencias: PianoBass es propia; Salamander CC BY 3.0 (atribución).
4. **v9 con datos propios:** si existen más grabaciones del Steinway (notas aisladas, graves, varias velocities), entrenarían una versión sin el problema de licencia de MAESTRO (CC BY-NC-SA).

## Qué necesitaría del usuario
- ¿`PianoBass` es una nota real de un Steinway sin procesar? ¿Hay más notas de esa sesión (A0–F#1, varias fuerzas), secas o con sala?
- ¿Dónde están las grabaciones del Steinway que mencionó (en `Anibal/Piano` solo hay 10 FX/texturas procesadas)?

## Cómo se comprobaría que mejora
- Barrido de calidad (`DDSPiano/Tools/qa_sweep.py`) sin alertas nuevas.
- Perfil de armónicos de las teclas 21–35 frente a las referencias (fundamental dentro de ±6 dB).
- Prueba a ciegas con la música del usuario, como las de `informes/test_*`.
