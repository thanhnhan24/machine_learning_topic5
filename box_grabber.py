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
import joblib

# ====== Biến toàn cục ======
shared_frame = None
shared_centroid = None
shared_cropped_box = None
robot_arm = None
z_fixed = 400  # mm
lock = threading.Lock()
stop_event = threading.Event()
workspace_calib = None
shared_predicted_id = None
workspace = None
shared_box_angle = None 
shared_arrow = None  # (start_point, end_point)
arm = None
# ====== Hàm kết nối robot ======
def get_robot_ip():
    if len(sys.argv) >= 2:
        return sys.argv[1]
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

# ====== Đọc JSON vị trí ======
class WorkspaceDict(dict):
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            workspace_id, field = key
            return super().__getitem__(workspace_id)[field]
        return super().__getitem__(key)

def get_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return WorkspaceDict(data)

# ====== Thread 1: ArUco Detection ======
def detect_aruco_4x4_50():
    global shared_frame, shared_centroid, shared_cropped_box, shared_box_angle, shared_arrow
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())

            corners, ids, _ = detector.detectMarkers(color_image)

            with lock:
                shared_frame = color_image

                if ids is not None and len(ids) >= 4:
                    all_pts = np.concatenate([c[0] for c in corners])
                    shared_centroid = np.mean(all_pts, axis=0).astype(int)
                    x, y, w, h = cv2.boundingRect(all_pts.astype(np.int32))
                    shared_cropped_box = color_image[y:y+h, x:x+w]

                    try:
                        # Gom tất cả điểm 4 marker
                        all_pts = np.concatenate([c[0] for c in corners[:4]]).astype(np.float32)
                        rect = cv2.minAreaRect(all_pts)  # center, (w,h), angle
                        angle_deg = rect[2]

                        # Chuẩn hoá
                        if rect[1][0] < rect[1][1]:  # w < h
                            angle_deg = angle_deg
                        else:
                            angle_deg += 90
                        shared_box_angle = round(angle_deg, 2)

                        # ==== TÍNH VECTOR MŨI TÊN HƯỚNG RA PHÍA TRƯỚC BOX ====
                        box_center = np.array(rect[0], dtype=np.int32)
                        length = 80  # độ dài vector
                        angle_rad = math.radians(angle_deg)

                        dx = int(length * math.cos(angle_rad))
                        dy = int(length * math.sin(angle_rad))
                        start_pt = tuple(box_center)
                        end_pt = (box_center[0] + dx, box_center[1] + dy)
                        shared_arrow = (start_pt, end_pt)
                    except:
                        shared_box_angle = None
                        shared_arrow = None

                else:
                    shared_centroid = None
                    shared_cropped_box = None
                    shared_box_angle = None
    finally:
        pipeline.stop()


# ====== Thread 2: Hiển thị frame chính ======
def display_main_frame():
    global shared_box_angle, shared_arrow
    prev_time = time.time()
    while not stop_event.is_set():
        with lock:
            frame = shared_frame.copy() if shared_frame is not None else None
            centroid = shared_centroid.copy() if shared_centroid is not None else None
            angle = shared_box_angle

        if frame is not None:
            if centroid is not None:
                cv2.circle(frame, tuple(centroid), 7, (0, 255, 0), -1)
                cv2.putText(frame, f"Centroid: {tuple(centroid)}", tuple(centroid + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            if angle is not None:
                cv2.putText(frame, f"Angle: {angle:.2f} deg", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            fps = 1.0 / (time.time() - prev_time)
            prev_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            with lock:
                arrow = shared_arrow

            if arrow is not None:
                cv2.arrowedLine(frame, arrow[0], arrow[1], (0, 0, 255), 3, tipLength=0.2)
            cv2.imshow("Main View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyWindow("Main View")

# ====== Thread 3: Hiển thị crop và phân loại BoVW ======
def display_cropped_box():
    global shared_predicted_id
    model_dir = r"C:\\Users\\Duc An Ho\\Desktop\\Robot-theta"
    bow_kmeans = joblib.load(os.path.join(model_dir, "bow_kmeans.pkl"))
    pca = joblib.load(os.path.join(model_dir, "pca.pkl"))
    kmeans = joblib.load(os.path.join(model_dir, "kmeans.pkl"))
    sift = cv2.SIFT_create()

    IMAGE_SIZE = (200, 200)
    N_VISUAL_WORDS = bow_kmeans.n_clusters
    cluster_to_id = {0: "id_2", 1: "id_3", 2: "id_1"}

    save_dir = os.path.join("train", "id_1")
    os.makedirs(save_dir, exist_ok=True)
    image_index = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])

    while not stop_event.is_set():
        with lock:
            cropped = shared_cropped_box.copy() if shared_cropped_box is not None else None

        if cropped is not None:
            gray = cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), IMAGE_SIZE)
            kp, des = sift.detectAndCompute(gray, None)
            if des is not None:
                words = bow_kmeans.predict(des)
                hist = np.bincount(words, minlength=N_VISUAL_WORDS)
                hist_pca = pca.transform([hist])
                cluster_id = kmeans.predict(hist_pca)[0]
                predicted_id = cluster_to_id.get(cluster_id, "unknown")
                shared_predicted_id = predicted_id
                cv2.putText(cropped, predicted_id, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(cropped, "No Features", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Warped Box", cropped)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break
        elif key == ord('1') and cropped is not None:
            filename = os.path.join(save_dir, f"{image_index}.jpg")
            cv2.imwrite(filename, cropped)
            print(f"[INFO] Saved: {filename}")
            image_index += 1

    cv2.destroyWindow("Warped Box")

# ====== Thread 4: Điều khiển robot đến centroid theo xác định ổn định ======
def move_robot_to_centroid():
    global robot_arm, workspace_calib, shared_predicted_id, shared_box_angle
    frame_width, frame_height = 640, 480
    stable_duration, threshold = 1.0, 5
    last_centroid, stable_start_time = None, None   

    while not stop_event.is_set():
        with lock:
            centroid = shared_centroid.copy() if shared_centroid is not None else None

        if centroid is not None:
            if last_centroid is None:
                last_centroid = centroid
                stable_start_time = time.time()
            else:
                dx = abs(centroid[0] - last_centroid[0])
                dy = abs(centroid[1] - last_centroid[1])
                if dx < threshold and dy < threshold:
                    if time.time() - stable_start_time >= stable_duration:
                        error_x = centroid[0] - frame_width / 2
                        error_y = centroid[1] - frame_height / 2
                        scale_mm_per_pixel = 0.5
                        delta_x = error_y * scale_mm_per_pixel
                        delta_y = error_x * scale_mm_per_pixel

                        code, pos = robot_arm.get_position(is_radian=False)
                        if code != 0 or pos is None:
                            print("[ERROR] Lỗi lấy vị trí hiện tại của robot")
                            break

                        target_x = pos[0] + delta_x
                        target_y = pos[1] + delta_y
                        if shared_box_angle is not None:
                            target_angle = shared_box_angle
                        print(f"[INFO] Di chuyển robot đến ({target_x:.1f}, {target_y:.1f})")
                        robot_arm.set_position(x=target_x + workspace_calib['calib_x'], y=target_y + workspace_calib['calib_y'], z=175,yaw =  target_angle + workspace_calib['offset_theta'],  wait=True)
                        robot_arm.open_lite6_gripper(1)
                        time.sleep(0.5)                                  
                        robot_arm.set_position(x=target_x + workspace_calib['calib_x'], y=target_y + workspace_calib['calib_y'], z=155,yaw =  target_angle + workspace_calib['offset_theta'],  wait=True)
                        robot_arm.close_lite6_gripper(1)
                        time.sleep(0.5)
                        robot_arm.set_position(x=target_x + workspace_calib['calib_x'], y=target_y + workspace_calib['calib_y'], z=400,yaw =  0,  wait=True)
                        destination_x = None
                        destination_y = None
                        destination_z = None
                        destination_yaw = None
                        if shared_predicted_id == "id_2":
                            destination_x, destination_y, destination_z, destination_yaw = workspace["2", "position"]
                        last_centroid = None
                        time.sleep(2)
                else:
                    stable_start_time = time.time()
                    last_centroid = centroid          
        time.sleep(0.1)

# ====== MAIN ======
def main():
    global robot_arm, workspace_calib, workspace
    ip = get_robot_ip()
    robot_arm = init_robot(ip)

    workspace = get_json_data('theta_cal.json')
    pos = workspace["1", "position"]
    calib = workspace["1", "calibration"]
    workspace_calib = calib

    robot_arm.set_position(x=pos['x']+calib['calib_x'], y=pos['y']+calib['calib_y'], z=pos['z'] + calib['calib_z'], wait=True)

    t1 = threading.Thread(target=detect_aruco_4x4_50, daemon=True)
    t2 = threading.Thread(target=display_main_frame, daemon=True)
    t3 = threading.Thread(target=display_cropped_box, daemon=True)
    t4 = threading.Thread(target=move_robot_to_centroid, daemon=True)

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()

if __name__ == '__main__':
    main()