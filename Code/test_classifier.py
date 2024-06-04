import cv2
import mediapipe as mp
import pickle
import numpy as np

cap = cv2.VideoCapture(0)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: '1', 1: '2', 2: '3'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.extend([x, y])
                x_.append(x)
                y_.append(y)

        # Ensure data_aux is padded to the correct length
        max_len = 42  # This should match the length used during training
        data_aux = data_aux + [0] * (max_len - len(data_aux))

        data_aux = np.asarray(data_aux).reshape(1, -1)
        data_aux = np.concatenate((data_aux, data_aux), axis=1)[:,:max_len]

        prediction = model.predict(data_aux)
        predicted = labels_dict[int(prediction[0])]

        print(predicted)

        x1 = min(x_) * W - 10
        y1 = min(y_) * H - 10
        x2 = max(x_) * W - 10
        y2 = max(y_) * H - 10

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 3)
        cv2.putText(frame, f'The Character is {predicted}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
