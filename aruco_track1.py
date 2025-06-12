import cv2
import numpy as np

def get_marker_angle(corners):
    """
    Tính góc của marker so với trục Ox (góc trên mặt phẳng ảnh).
    corners: (4,2) - 4 điểm góc theo thứ tự: TL, TR, BR, BL
    """
    # Lấy điểm trên trái (TL) và trên phải (TR)
    (tl, tr) = corners[0][0], corners[0][1]
    # Tính vector từ TL -> TR
    vector = tr - tl
    angle_rad = np.arctan2(vector[1], vector[0])  # radian
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Chọn dictionary ArUco 5x5
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Khởi tạo webcam
cap = cv2.VideoCapture(0)  # Nếu camera khác, đổi index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện marker
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    if ids is not None:
        for i, corner in enumerate(corners):
            # Vẽ bounding box
            cv2.polylines(frame, [corner.astype(int)], True, (0, 255, 0), 2)

            # Tính tâm marker
            center = np.mean(corner[0], axis=0).astype(int)
            cv2.circle(frame, tuple(center), 4, (0, 0, 255), -1)

            # Tính góc
            angle = get_marker_angle(corner)
            text = f"ID:{ids[i][0]}, Angle:{angle:.1f} deg"

            # Hiển thị text
            cv2.putText(frame, text, (center[0]+10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow('ArUco 5x5 Scanner', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
