# Jardín digital — 404

Animación de 36 segundos a partir de una ilustración original. Combina un infierno estilo Bosco con interfaces cyberpunk.

▶️ Video: [`video/jardin_digital.mp4`](video/jardin_digital.mp4)

## Qué pasa en el video

| Tiempo | Escena |
|---|---|
| 0–12 s | Los paneles de código se escriben solos y el binario cambia. El reloj de arena se voltea, los globos giran, el fuego titila, suben brasas y cruzan pájaros. Hay glitches y un zoom lento. |
| 12–13 s | Un glitch abre una ventana de terminal (`jardin.exe — /dev/sueños`). |
| 13–28 s | El texto se escribe con el cursor visible. |
| 28–35 s | El cursor parpadea; después, `404.` se corrompe y la ventana se cierra. |

El último fotograma empalma con el primero, así que el video se puede reproducir en bucle.

## Texto

```
Ven a jugar.
Tú eres la pureza.

Mi modo lírico es lluvia de ideas en tu mente.
Soy la organizadora de tus delirios nocturnos.

Represento lo que odias,
un sucio deseo.
El dolor que sentiste es mío:
siéntete infectado.

Soy el anzuelo de todos tus miedos.
Las serpientes están en el jardín.

404.
```

## Estructura

```
assets/src.png          imagen original (1672×941)
animate.py              loop de 12 s (efectos sobre la imagen)
final.py                loop + ventana de texto (36 s)
video/jardin_digital.mp4
```

## Regenerar el video

Requisitos: Python 3.9+, `ffmpeg` en el PATH y la fuente DejaVu Sans Mono (en Linux viene en `/usr/share/fonts/truetype/dejavu/`; en otro sistema cambia la ruta en los scripts).

```bash
pip install -r requirements.txt
python3 final.py            # genera video/jardin_digital.mp4 (1920×1080, 30 fps)
python3 animate.py          # opcional: solo el loop de 12 s
python3 final.py preview 12.5 20 30   # fotogramas sueltos (en segundos) para revisar
```

Para cambiar el texto, edita `TEXT` en `final.py`. La velocidad de escritura es `RATE`, en caracteres por segundo.
