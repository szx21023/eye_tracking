import cv2
import mediapipe as mp
import numpy as np
import time

# 初始化 MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 設定攝像頭
cap = cv2.VideoCapture(0)

# 畫面大小
WIDTH, HEIGHT = 640, 480
GRID_ROWS, GRID_COLS = 3, 3  # 九宮格
CELL_WIDTH, CELL_HEIGHT = WIDTH // GRID_COLS, HEIGHT // GRID_ROWS

# 追蹤視線的區域
last_active_cell = None
active_start_time = None
HOLD_TIME = 1.0  # 停留 1 秒高亮

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 偵測人臉與眼睛
    results = face_mesh.process(rgb_frame)
    eye_center = None  # 眼睛中心點

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 取得左眼與右眼關鍵點
            left_eye_x = (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2 * WIDTH
            left_eye_y = (face_landmarks.landmark[33].y + face_landmarks.landmark[133].y) / 2 * HEIGHT

            right_eye_x = (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2 * WIDTH
            right_eye_y = (face_landmarks.landmark[362].y + face_landmarks.landmark[263].y) / 2 * HEIGHT

            # 取得眼睛中心點
            eye_center = (int((left_eye_x + right_eye_x) / 2), int((left_eye_y + right_eye_y) / 2))
            cv2.circle(frame, eye_center, 5, (0, 255, 0), -1)  # 畫出眼球中心點

    # 繪製九宮格
    active_cell = None
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1, y1 = col * CELL_WIDTH, row * CELL_HEIGHT
            x2, y2 = x1 + CELL_WIDTH, y1 + CELL_HEIGHT

            # 檢查眼睛中心是否落在這個格子
            if eye_center and x1 <= eye_center[0] < x2 and y1 <= eye_center[1] < y2:
                active_cell = (row, col)

            # 高亮當前視線停留的格子
            color = (255, 255, 255) if (row, col) == last_active_cell else (100, 100, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

            # 畫格線
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # 記錄視線停留時間
    if active_cell:
        if active_cell == last_active_cell:
            if time.time() - active_start_time > HOLD_TIME:
                cv2.rectangle(frame, (active_cell[1] * CELL_WIDTH, active_cell[0] * CELL_HEIGHT),
                              ((active_cell[1] + 1) * CELL_WIDTH, (active_cell[0] + 1) * CELL_HEIGHT), 
                              (0, 255, 0), -1)
        else:
            last_active_cell = active_cell
            active_start_time = time.time()

    # 顯示視訊畫面
    cv2.imshow("Eye Tracking Grid", frame)
    
    # 按下 "q" 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
