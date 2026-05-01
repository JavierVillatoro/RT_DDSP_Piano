import json

# Ruta exacta a tu archivo JSON
ruta_json = r"C:\Users\franc\Desktop\proyectos\DDSP\training\rtneural_export\inharmonicity_weights.json"

print("Cargando el cerebro neuronal en memoria...\n")

try:
    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    print("=== RADIOGRAFÍA DE LA CAPA GRU ===")
    capa_gru = datos.get("gru")

    if capa_gru:
        for clave, valores in capa_gru.items():
            if isinstance(valores, list):
                longitud = len(valores)
                # Comprobamos si el primer elemento es otra lista (2D)
                if longitud > 0 and isinstance(valores[0], list):
                    print(f"-> '{clave}': MATRIZ 2D (Lista de listas) | {longitud} filas x {len(valores[0])} columnas")
                else:
                    print(f"-> '{clave}': VECTOR 1D (Lista plana)   | {longitud} elementos en total")
    else:
        print("No se encontró la clave 'gru' en el JSON.")

except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta:\n{ruta_json}")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")