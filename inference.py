import cv2
import numpy as np

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # comment if not intel-tensorflow

import tensorflow as tf

model = tf.keras.models.load_model('model32.h5')

labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: "del", 27: 'space', 28: 'nothing'}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    img = cv2.resize(frame, (200, 200))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    pred = model.predict(img)
    pred_label = labels[np.argmax(pred)]
    
    cv2.putText(frame, pred_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
