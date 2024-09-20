import cv2

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Comprobar si la cámara está abierta
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

while True:
    # Capturar cuadro por cuadro
    ret, frame = cap.read()

    # Si no se recibió el cuadro, se rompe el ciclo
    if not ret:
        print("No se pudo recibir el cuadro (fin del video)")
        break

    # Mostrar el cuadro en pantalla
    cv2.imshow('Presiona "c" para capturar la imagen', frame)

    # Esperar a que el usuario presione una tecla
    key = cv2.waitKey(1)

    # Si el usuario presiona "c", se captura la imagen
    if key == ord('c'):
        # Guardar la imagen capturada
        cv2.imwrite('captura.png', frame)
        print("Imagen capturada y guardada como captura.png")

    # Si el usuario presiona "q", se sale del bucle
    if key == ord('q'):
        break

# Cuando todo esté listo, liberar el recurso
cap.release()
cv2.destroyAllWindows()