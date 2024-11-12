import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Inicializar Mediapipe para detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Configuración de la ventana
width, height = 960, 580
window_name = "Juego de Pelotas Rojas"
cv2.namedWindow(window_name)

# Variables de juego
ball_radius = 30
balls = []
score = 0
last_ball_creation_time = time.time()  # Para controlar el tiempo entre nuevas pelotas

# Función para agregar una pelota
def create_ball():
    x_pos = random.randint(ball_radius, width - ball_radius)
    y_pos = -ball_radius  # Comienza desde fuera de la pantalla (parte superior)

    # Ajustar la velocidad según el puntaje
    if score >= 30:
        velocity = 40
    elif score >= 15:
        velocity = 15
    else:
        velocity = 5

    balls.append({"pos": [x_pos, y_pos], "velocity": velocity})

# Crear la primera pelota
create_ball()

# Bucle de captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Voltear y redimensionar el frame
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Crear nuevas pelotas cada segundo
    if time.time() - last_ball_creation_time > 3:  # Crear una pelota cada segundo
        create_ball()
        last_ball_creation_time = time.time()

    # Dibujar las pelotas y manejarlas
    for ball in balls[:]:
        ball["pos"][1] += ball["velocity"]
        if ball["pos"][1] - ball_radius > height:
            balls.remove(ball)
            create_ball()  # Crear una nueva pelota si una se pierde de la pantalla

        # Dibujar la pelota
        cv2.circle(frame, (ball["pos"][0], ball["pos"][1]), ball_radius, (0, 0, 255), -1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener las coordenadas de la mano en pixeles
            h, w, _ = frame.shape
            hand_pos = np.array([
                int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
            ])

            # Chequear si la mano toca alguna pelota
            for ball in balls[:]:
                ball_pos = np.array(ball["pos"])
                distance = np.linalg.norm(hand_pos - ball_pos)
                if distance < ball_radius * 1.5:
                    balls.remove(ball)
                    score += 1
                    break

    # Dibujar el puntaje
    cv2.putText(frame, f"Puntaje: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostrar el frame
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
