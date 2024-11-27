import cv2
import mediapipe as mp
import numpy as np

# Inicializar el módulo de MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Inicializar la cámara web
cap = cv2.VideoCapture(0)

# Mapeo de gestos a emojis y significados
gestos_dict = {
    "thumb_up": ("👍", "golpe"),
    "thumb_down": ("👎", "Desaprobación / No me gusta"),
    "peace": ("✌️", "hola"),
    "rock_on": ("🤘", "Actitud de rock"),
    "fist": ("👊", "Fuerza / Luchar"),
    "ok": ("👌", "okey"),
    "praying": ("🙏", "Por favor / Gracias"),
    "offering": ("🤲", "Ofrecer algo"),
    "muscle": ("💪", "exelente"),
    "stop": ("🤚", "Alto / Detente"),
    "raise_hand": ("✋", "Alto / Detente"),
    "wave": ("👋", "Hola / Adiós"),
    "five_fingers": ("🖐", "Adiós"),
    "rock_sign": ("🤟", "Actitud de rock"),
    "nailpolish": ("💅", "Cuidado personal / Belleza"),
    "vulcan_greeting": ("🖖", "Larga vida y prosperidad"),
    "open_hands": ("👐", "Receptividad / Recibir"),
    "left_point": ("👈", " izquierda"),
    "right_point": ("👉", "derecha"),
    "hug": ("👐", "Abrazo"),
    "call_me": ("🤙", "Llamada / Conectando"),
    "shaka": ("🤙", "Aloha / Buena onda"),
    "clap": ("👏", "Aplauso"),
    "raised_fist": ("✊", "Resistencia / Fuerza"),
}

def detectar_gesto(hand_landmarks):
    """
    Función para detectar gestos básicos de la mano basados en las coordenadas de los puntos de referencia.
    """
    if hand_landmarks:
        # Extraer las posiciones de los puntos de referencia (landmarks)
        landmarks = hand_landmarks.landmark
        
        # Pulgar
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        thumb_base = landmarks[mp_hands.HandLandmark.THUMB_CMC]
        
        # Índice y medio
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Anular y meñique
        ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

        # Gesto de "Pulgar hacia arriba" (👍)
        if thumb_tip.y < thumb_base.y and all(landmarks[i].y > landmarks[i - 1].y for i in range(4, 20, 4)):
            return gestos_dict["thumb_up"]
        
        # Gesto de "Pulgar hacia abajo" (👎)
        if thumb_tip.y > thumb_base.y and abs(index_tip.x - thumb_tip.x) > 0.1:
            return gestos_dict["thumb_down"]

        # Gesto de "Paz" (✌️)
        if abs(index_tip.x - middle_tip.x) < 0.1 and abs(index_tip.y - middle_tip.y) < 0.1:
            if all(landmarks[i].y < landmarks[i - 1].y for i in range(8, 20, 4)):
                return gestos_dict["peace"]
        
        # Gesto de "OK" (👌)
        if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
            return gestos_dict["ok"]

        # Gesto de "Puño cerrado" (👊)
        if all(landmarks[i].y < landmarks[i - 1].y for i in range(4, 20, 4)):
            return gestos_dict["fist"]

        # Gesto de "Mano abierta" (🖐)
        if all(landmarks[i].y < landmarks[i - 1].y for i in range(8, 20, 4)):
            return gestos_dict["five_fingers"]

        # Gesto de "Musculo" (💪)
        if abs(index_tip.x - pinky_tip.x) < 0.05 and all(landmarks[i].y > landmarks[i - 1].y for i in range(5, 20, 4)):
            return gestos_dict["muscle"]
        
        # Gesto de "Señalando izquierda" (👈)
        if index_tip.x < thumb_tip.x:
            return gestos_dict["left_point"]
        
        # Gesto de "Señalando derecha" (👉)
        if index_tip.x > thumb_tip.x:
            return gestos_dict["right_point"]

        # Gesto de "Abrazo" (👐)
        if abs(index_tip.x - pinky_tip.x) < 0.1 and all(landmarks[i].y < landmarks[i - 1].y for i in range(4, 20, 4)):
            return gestos_dict["hug"]

        # Gesto de "Llamada" (🤙)
        if abs(thumb_tip.x - pinky_tip.x) < 0.05 and abs(index_tip.x - pinky_tip.x) > 0.15:
            return gestos_dict["call_me"]
        
        # Gesto de "Shaka" (🤙)
        if abs(thumb_tip.x - pinky_tip.x) < 0.05 and abs(index_tip.x - pinky_tip.x) < 0.1:
            return gestos_dict["shaka"]
        
        # Gesto de "Aplauso" (👏)
        if all(landmarks[i].y < landmarks[i - 1].y for i in range(5, 20, 4)) and abs(thumb_tip.y - index_tip.y) < 0.1:
            return gestos_dict["clap"]

        # Gesto de "Fist raised" (✊)
        if abs(thumb_tip.x - pinky_tip.x) < 0.05 and abs(index_tip.x - middle_tip.x) > 0.2:
            return gestos_dict["raised_fist"]

    return "Desconocido: Gestos no reconocidos"

while True:
    # Capturar una imagen desde la cámara
    ret, frame = cap.read()

    # Convertir la imagen a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar las manos
    results = hands.process(frame_rgb)

    # Dibujar las anotaciones en la imagen (puntos de la mano)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detectar gestos y agregar la descripción correspondiente
            gesto = detectar_gesto(hand_landmarks)
            
            # Verificar si el gesto es una tupla con emoji y significado
            if isinstance(gesto, tuple) and len(gesto) == 2:
                emoji, significado = gesto
                # Mostrar el gesto y su descripción en la imagen
                cv2.putText(frame, f"{emoji} - {significado}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                # Si no es un gesto válido, mostrar un mensaje de error
                cv2.putText(frame, gesto, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Mostrar la imagen con los gestos detectados
    cv2.imshow('Reconocimiento de Gestos de Mano', frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
