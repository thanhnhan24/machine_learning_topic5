import os
import sys
import time
import threading
import math
import json
import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

# --- Biến toàn cục ---
centroid_lock = threading.Lock()
global_centroid = None
center_x = None
center_y = None
marker_angle = None

calib_z = 0
calib_x = 50
calib_y = -26
offset_theta = 180

    # ==================== Kết nối Robot ====================
def get_robot_ip():
    if len(sys.argv) >= 2:
        return sys.argv[1]
    else:
        try:
            from configparser import ConfigParser
            parser = ConfigParser()
            parser.read('../robot.conf')
            return parser.get('xArm', 'ip')
        except:
            return '192.168.1.165'

def init_robot(ip):
    arm = XArmAPI(ip)
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    arm.clean_error()
    time.sleep(1)
    arm.move_gohome(wait=True)
    return arm

def move_robot_to_position(arm, pos_index='home'):
    positions = {
        'home':       {'x': 190, 'y': 0, 'z': 400, 'roll': 180, 'pitch': 0, 'yaw': 0, 'speed': 800},
        'default':    {'x': 87, 'y': 0, 'z': 155, 'roll': 180, 'pitch': 0, 'yaw': 0, 'speed': 800},
        'workspace1': {'x': 38, 'y': -200, 'z': 400, 'roll': 180, 'pitch': 0, 'yaw': 0, 'speed': 800},
        'workspace2': {'x': 222, 'y': -160, 'z': 430, 'roll': 180, 'pitch': 0, 'yaw': 0, 'speed': 800},
        'workspace3': {'x': 190, 'y': 168, 'z': 400, 'roll': 180, 'pitch': 0, 'yaw': 0, 'speed': 800},
    }
    pos = positions.get(pos_index)
    if pos:
        arm.set_position(**pos, wait=True)
        time.sleep(0.3)

# ==================== Camera Thread ====================
def camera_thread():
    global global_centroid, center_x, center_y, marker_angle
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    prev_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            corners, ids, _ = detector.detectMarkers(frame)

            if ids is not None and len(ids) >= 4:
                marker_centroids = []
                for i in range(min(4, len(corners))):
                    pts = corners[i][0]                                             
                    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
                    marker_centroids.append([cx, cy])

                centroid_np = np.array(marker_centroids, dtype=np.float32)
                (cx_total, cy_total) = np.mean(centroid_np, axis=0).astype(int)
                angle_deg = cv2.minAreaRect(centroid_np)[2]
                if cv2.minAreaRect(centroid_np)[1][0] < cv2.minAreaRect(centroid_np)[1][1]:
                    angle_deg += 90

                with centroid_lock:
                    global_centroid = (cx_total, cy_total)
                    center_x = cx_total
                    center_y = cy_total
                    marker_angle = angle_deg

                cv2.circle(frame, (cx_total, cy_total), 7, (0, 0, 255), -1)
                cv2.putText(frame, f'Centroid: ({cx_total}, {cy_total})', (cx_total + 10, cy_total),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f'Angle: {angle_deg:.2f} deg', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            fps = 1.0 / (time.time() - prev_time)
            prev_time = time.time()
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.imshow("Camera View - ArUco", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# ==================== Robot Calibration Thread ====================
def robot_thread(arm):
    frame_width, frame_height = 640, 480
    stable_duration, threshold = 1.0, 5
    last_centroid, stable_start_time = None, None

    move_robot_to_position(arm, 'home')
    choice = input("Nhập workspace cần calib (VD: 1): ").strip()
    if choice == '1':
        move_robot_to_position(arm, 'workspace1')
    elif choice == '2':
        move_robot_to_position(arm, 'workspace2')
    elif choice == '3':
        move_robot_to_position(arm, 'workspace3')
    while True:
        with centroid_lock:
            cx, cy = center_x, center_y
            angle = marker_angle
            centroid = global_centroid

        if centroid and cx is not None and cy is not None:
            if last_centroid is None:
                last_centroid = centroid
                stable_start_time = time.time()
            else:
                dx, dy = abs(centroid[0] - last_centroid[0]), abs(centroid[1] - last_centroid[1])
                if dx < threshold and dy < threshold:
                    if time.time() - stable_start_time >= stable_duration:
                        error_x = cx - frame_width / 2
                        error_y = cy - frame_height / 2
                        scale_mm_per_pixel = 0.5

                        delta_x = error_y * scale_mm_per_pixel
                        delta_y = error_x * scale_mm_per_pixel

                        code, pos = arm.get_position(is_radian=False)
                        if code != 0 or pos is None:
                            print("Lỗi khi lấy vị trí robot.")
                            break

                        target_x, target_y = pos[0] + delta_x, pos[1] + delta_y
                        print(f"Centroid ổn định. Di chuyển tới ({target_x:.1f}, {target_y:.1f})")
                        arm.set_position(x=target_x, y=target_y, z=pos[2],
                                         yaw= offset_theta - angle, pitch=0, speed=1000, wait=True)

                        code, final_pos = arm.get_position(is_radian=False)
                        if code != 0 or final_pos is None:
                            print("Lỗi lấy vị trí sau khi di chuyển.")
                            break

                        fx, fy, fz = final_pos[0], final_pos[1], final_pos[2]
                        arm.set_position(x=fx + calib_x, y=fy + calib_y, z=fz, wait=True)
                        arm.set_position(x=fx + calib_x, y=fy + calib_y, z=100, wait=True)

                        confirm = input(f"Chấp nhận vị trí [{fx}, {fy}, {fz}, góc {angle:.2f}]? (y/n): ").lower()
                        data = {
                            "workspace": choice,
                            "position": {
                                "x": fx,
                                "y": fy,
                                "z": fz,
                                "yaw": math.ceil(angle)
                            },
                            "calibration": {
                                "calib_x": calib_x,
                                "calib_y": calib_y,
                                "calib_z": calib_z,
                                "offset_theta": offset_theta
                            }
                        }
                        if confirm == 'y':
                            file_path = "theta_cal.json"
                            if os.path.exists(file_path):
                                with open(file_path, "r", encoding="utf-8") as f:
                                    try:
                                        existing_data = json.load(f)
                                    except json.JSONDecodeError:
                                        existing_data = {}
                            else:
                                existing_data = {}

                            # Bước 2: Cập nhật workspace tương ứng
                            existing_data[choice] = {
                                "position": {
                                    "x": fx,
                                    "y": fy,
                                    "z": fz,
                                    "yaw": math.ceil(angle)
                                },
                                "calibration": {
                                    "calib_x": calib_x,
                                    "calib_y": calib_y,
                                    "calib_z": calib_z,
                                    "offset_theta": offset_theta
                                }
                            }

                            # Bước 3: Ghi lại toàn bộ
                            with open(file_path, "w", encoding="utf-8") as f:
                                json.dump(existing_data, f, indent=4)
                            print("Đã cập nhật thông số vào theta_cal.json")
                        else:
                            print("Calibration bị hủy.")
                        again = input("Tiếp tục calibration khác? (y/n): ").lower()
                        if again == 'y':
                            move_robot_to_position(arm, 'home')
                            last_centroid = None
                            continue
                        else:
                            move_robot_to_position(arm, 'default')
                            print("Kết thúc calibration.")
                            break
                else:
                    stable_start_time = time.time()
                    last_centroid = centroid

        time.sleep(0.1)

# ==================== Main ====================
def main():
    ip = get_robot_ip()
    arm = init_robot(ip)

    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    rob_thread = threading.Thread(target=robot_thread, args=(arm,), daemon=True)

    cam_thread.start()
    rob_thread.start()

    cam_thread.join()

if __name__ == "__main__":
    main()
