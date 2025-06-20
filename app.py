from PySide2.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog
from PySide2.QtCore import QStringListModel, QThread, Signal, QTimer    
from PySide2.QtGui import QImage, QPixmap
from ui_source import Ui_MainWindow, Ui_Dialog
from xarm.wrapper import XArmAPI

import sys, json, time, datetime, psutil, cv2, joblib, os, queue, math
import threading
import pyrealsense2 as rs
import numpy as np

class BoxClassifierThread(QThread):
    result_signal = Signal(str)
    label_signal = Signal(str)
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.running = True

    def run(self):
        while self.running:
            time.sleep(0.1)
            with self.app.shared_lock:
                if self.app.shared_cropped_box is not None:
                    cropped = self.app.shared_cropped_box.copy()
                else:
                    continue

            try:
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (200, 200))
                kp, des = cv2.SIFT_create().detectAndCompute(resized, None)

                if des is None or self.app.bow_kmeans is None:
                    continue

                words = self.app.bow_kmeans.predict(des)
                hist = np.bincount(words, minlength=self.app.bow_kmeans.n_clusters)
                hist_pca = self.app.pca.transform([hist])
                cluster_id = self.app.kmeans.predict(hist_pca)[0]
                cluster_to_id = {0: "id_1", 1: "id_2", 2: "id_3"}
                result = cluster_to_id.get(cluster_id, "unknown")

                self.label_signal.emit(result)
                self.result_signal.emit(result)
                with self.app.shared_lock:
                    self.app.shared_cropped_box = None  # reset sau mỗi lần phân loại
            except Exception as e:
                self.result_signal.emit(f"Lỗi phân loại: {e}")

    def stop(self):
        self.running = False
        self.wait()

class ArucoDetectThread(QThread):
    image_signal = Signal(QImage)
    position_signal = Signal(tuple)  # (x, y, angle)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())

    def run(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

        try:
            while not self.app.stop_event.is_set():
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                corners, ids, _ = self.detector.detectMarkers(color_image)

                with self.app.shared_lock:
                    self.app.shared_frame = color_image.copy()

                    if ids is not None and len(ids) >= 4:
                        all_pts = np.concatenate([c[0] for c in corners])
                        centroid = np.mean(all_pts, axis=0).astype(int)
                        self.app.shared_centroid = centroid

                        try:
                            # Tính rect và corrected_angle
                            rect = cv2.minAreaRect(all_pts.astype(np.float32))
                            raw_angle = rect[2]
                            if rect[1][0] < rect[1][1]:
                                corrected_angle = raw_angle
                            else:
                                corrected_angle = raw_angle + 90

                            # Xác định hướng mũi tên từ mã ArUco đầu tiên
                            marker_pts = corners[0][0]
                            pt1, pt2 = marker_pts[0], marker_pts[1]
                            vec = pt2 - pt1
                            marker_angle = np.degrees(np.arctan2(vec[1], vec[0]))

                            # Flip nếu lệch 180 độ
                            if abs(corrected_angle - marker_angle) > 90:
                                corrected_angle = (corrected_angle + 180) % 360

                            self.app.shared_box_angle = round(corrected_angle, 2)
                            self.position_signal.emit((centroid[0], centroid[1], corrected_angle))

                            # Tính mũi tên hướng từ rect
                            center = np.array(rect[0], dtype=np.int32)
                            angle_rad = math.radians(corrected_angle)
                            dx = int(80 * math.cos(angle_rad))
                            dy = int(80 * math.sin(angle_rad))
                            self.app.shared_arrow = (tuple(center), (center[0] + dx, center[1] + dy))

                        except Exception as e:
                            self.app.shared_box_angle = None
                            self.app.shared_arrow = None

                    else:
                        self.app.shared_centroid = None
                        self.app.shared_box_angle = None
                        self.app.shared_arrow = None

                # Vẽ kết quả và emit ảnh
                display = color_image.copy()
                if self.app.shared_centroid is not None:
                    cv2.circle(display, tuple(self.app.shared_centroid), 5, (0, 255, 0), -1)
                    cv2.putText(display, f"Angle: {self.app.shared_box_angle:.1f}",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if self.app.shared_arrow is not None:
                    cv2.arrowedLine(display, self.app.shared_arrow[0], self.app.shared_arrow[1],
                                    (0, 0, 255), 2, tipLength=0.2)

                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.image_signal.emit(qimg)

        finally:
            pipeline.stop()

class RobotControlThread(QThread):
    def __init__(self, arm, lock, app = None):  # thêm app ở đây
        super().__init__()
        self.arm = arm
        self.lock = lock
        self.app = app
        self.command_queue = queue.Queue()
        self.running = True


    def run(self):
        while not self.app.stop_event.is_set():
            try:
                command = self.command_queue.get(timeout=0.1)

                if command['type'] == 'move':
                    self.set_position(**command['params'])

                elif command['type'] == 'gripper':
                    if command['action'] == 'open':
                        self.open_gripper()
                    elif command['action'] == 'close':
                        self.close_gripper()

                elif command['type'] == 'sequence':
                    for step in command['steps']:
                        if 'move' in step:
                            self.set_position(**step['move'])
                        if step.get('gripper') == 'open':
                            self.open_gripper()
                        elif step.get('gripper') == 'close':
                            self.close_gripper()
                        time.sleep(step.get('delay', 0.5))

            except queue.Empty:
                continue


    def stop(self):
        self.running = False
        self.wait()

    def set_position(self, x, y, z, yaw):
        self.command_queue.put({'type': 'move', 'x': x, 'y': y, 'z': z, 'yaw': yaw})

    def open_gripper(self):
        self.command_queue.put({'type': 'open_gripper'})
    
    def close_gripper(self):
        self.command_queue.put({'type': 'close_gripper'})
class RobotMonitorThread(QThread):
    update_signal = Signal(float, float, float, float)

    def __init__(self, arm, lock):
        super().__init__()
        self.arm = arm
        self.lock = lock
        self.running = True


    def run(self):
        while self.running:
            try:
                with self.lock:
                    pos = self.arm.get_position(is_radian=False)
                if pos and pos[0] == 0:
                    x, y, z, roll, pitch, yaw = pos[1]
                    self.update_signal.emit(x, y, z, yaw)
            except Exception as e:
                print("Lỗi khi đọc vị trí robot:", e)
            time.sleep(0.5) 

    def stop(self):
        self.running = False
        self.wait()

class CameraThread(QThread):
    frame_signal = Signal(QImage)

    def run(self):
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)

            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to QImage
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_signal.emit(qt_image)

        except Exception as e:
            print("Lỗi camera:", e)

class ParamDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.toolButton.clicked.connect(self.open_file_dialog)
        self.ui.pushButton.clicked.connect(self.accept)
        self.ui.pushButton_2.clicked.connect(self.reject)   
    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Open File", "", "JSON Files (*.json)")
        if file_path:
            self.ui.textBrowser.setMarkdown(file_path)
        else:
            self.ui.textBrowser.setMarkdown("")

class SystemMonitorThread(QThread):
    update_signal = Signal(str, str, int, float, float)

    def run(self):
        while True:
            now = datetime.datetime.now()
            time_str = now.strftime("%H:%M:%S")
            date_str = now.strftime("%d/%m/%Y")
            thread_count = psutil.cpu_count(logical=True)
            cpu_percent = psutil.cpu_percent(interval=1)
            ram_percent = psutil.virtual_memory().percent

            # Phát tín hiệu về GUI
            self.update_signal.emit(time_str, date_str, thread_count, cpu_percent, ram_percent)

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.arm = None
        self.arm_lock = threading.Lock()
        self.pallet1Calib_x = None
        self.pallet1Calib_y = None
        self.pallet1Offset_theta = None
        self.pallet2Calib_x = None
        self.pallet2Calib_y = None
        self.pallet2Offset_theta = None
        self.pallet3Calib_x = None
        self.pallet3Calib_y = None
        self.pallet3Offset_theta = None
        self.zeroPoint_X = None
        self.zeroPoint_Y = None
        self.zeroPoint_Z = None
        self.zeroTheta = None
        self.homePoint_X = None
        self.homePoint_Y = None
        self.homePoint_Z = None
        self.homeTheta = None
        self.shared_frame = None
        self.shared_centroid = None
        self.shared_box_angle = None
        self.shared_arrow = None
        self.shared_predicted_id = None
        self.shared_cropped_box = None
        self.shared_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.bow_kmeans = None
        self.pca = None
        self.kmeans = None
        self.current_predicted_id = None
        self.app = app  # gán app để dùng stop_event
        self.command_queue = queue.Queue()
        self.last_centroid = None
        self.stable_start_time = None
        self.stable_duration = 1.0  # thời gian ổn định yêu cầu (giây)
        self.stable_threshold = 5   # sai số pixel cho ổn định
        self.robot_control = RobotControlThread(self.arm, self.arm_lock, app=self)
        self.robot_control.start()
        # Khởi động luồng theo dõi hệ thống
        self.sys_monitor = SystemMonitorThread()
        self.sys_monitor.update_signal.connect(self.update_system_status)
        self.sys_monitor.start()


        # Kết nối nút mở dialog (ví dụ pushButton_2)
        self.ui.stage2button.clicked.connect(self.param_select_popup)
        self.ui.stage1button.clicked.connect(self.harwareManager)   
        self.ui.main_dirChoose.clicked.connect(self.mainDirPopup)
        self.ui.xarmZeroPoint.clicked.connect(self.moveZeroPoint)
        self.ui.xarmHomePoint.clicked.connect(self.moveHomePoint)
        self.ui.xarmWorkspaceSwitching.clicked.connect(self.sortingPallet)
        self.ui.pushButton.clicked.connect(self.MainProcess)

    def param_select_popup(self):
        # Tạm dừng robot_monitor nếu đang chạy
        if hasattr(self, 'robot_monitor') and self.robot_monitor.isRunning():
            self.robot_monitor.running = False
            self.robot_monitor.wait()

        dialog = ParamDialog()
        result = dialog.exec_()

        if result == QDialog.Accepted:
            file_path = dialog.ui.textBrowser.toMarkdown()
            if file_path != "":
                try:
                    raw = self.jsonDataLoader(file_path.strip())
                    QTimer.singleShot(0, lambda: self.display_data(raw))
                except Exception as e:
                    self.print_to_listview("Lỗi khi đọc file JSON: " + str(e))
            else:
                self.print_to_listview("Invalid file path")
        else:
            print("Cancel")

        # Khởi động lại robot_monitor sau khi dialog đóng
        if hasattr(self, 'arm') and self.arm is not None:
            self.robot_monitor = RobotMonitorThread(self.arm, self.arm_lock)
            self.robot_monitor.update_signal.connect(self.update_robot_position)
            self.robot_monitor.start()

    def mainDirPopup(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.print_to_listview("Selected directory:" + str(dir_path))
            self.ui.maindirText.setPlainText(dir_path)

            #  Tạm dừng robot_monitor
            if hasattr(self, 'robot_monitor') and self.robot_monitor.isRunning():
                self.robot_monitor.running = False
                self.robot_monitor.wait()

            # Delay xử lý model sau 1 vòng event loop
            QTimer.singleShot(0, lambda: self.load_kmean_safe(dir_path))
        else:
            self.print_to_listview("No directory selected")

    def load_kmean_safe(self, model_dir):
        try:
            model_dir = model_dir.strip()
            self.bow_kmeans = joblib.load(os.path.join(model_dir, "bow_kmeans.pkl"))
            self.pca = joblib.load(os.path.join(model_dir, "pca.pkl"))
            self.kmeans = joblib.load(os.path.join(model_dir, "kmeans.pkl"))

            self.print_to_listview("Load model success")
            self.ui.checkBox.setChecked(True)
            self.ui.checkBox_2.setChecked(True)
            self.ui.checkBox_3.setChecked(True)
        except Exception as e:
            self.print_to_listview("Lỗi khi load mô hình: " + str(e))

        #  Khởi động lại robot_monitor
        if hasattr(self, 'arm') and self.arm is not None:
            self.robot_monitor = RobotMonitorThread(self.arm, self.arm_lock)
            self.robot_monitor.update_signal.connect(self.update_robot_position)
            self.robot_monitor.start()

    def jsonDataLoader(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data
    
    def print_to_listview(self, msg):
            if not hasattr(self, 'log_model'):
                self.log_model = QStringListModel()
                self.log_data = []
                self.ui.logView.setModel(self.log_model)

            # Thêm msg vào danh sách
            self.log_data.append(msg)

            # Cập nhật model
            self.log_model.setStringList(self.log_data)

            # Tự động scroll xuống dòng cuối cùng
            index = self.log_model.index(len(self.log_data) - 1)
            self.ui.logView.scrollTo(index)

    def harwareManager(self):
    # ==== Kiểm tra xArm ====
        try:
            self.ip = '192.168.1.165'
            self.ui.ip_addr.setPlainText(self.ip)
            self.arm = XArmAPI(self.ip)
            self.arm.motion_enable(True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            self.arm.clean_error()
            time.sleep(1)
            self.arm.move_gohome(wait=True)
            # Sau khi self.arm.move_gohome(wait=True)
            if not hasattr(self, 'robot_monitor') or not self.robot_monitor.isRunning():
                self.robot_monitor = RobotMonitorThread(self.arm, self.arm_lock)
                self.robot_monitor.update_signal.connect(self.update_robot_position)
                self.robot_monitor.start()
                self.robot_control = RobotControlThread(self.arm, self.arm_lock)
                self.robot_control.start()


            self.print_to_listview("Kết nối xArm thành công.")
            self.ui.xarmManager.setChecked(True)
        except Exception as e:
            self.print_to_listview("Không thể kết nối với xArm: " + str(e))

        # ==== Kiểm tra camera Intel RealSense D435i ====
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            time.sleep(1)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                self.print_to_listview("Camera Intel RealSense D435i đã được kết nối.")
                self.ui.webcamManager.setChecked(True)
                pipeline.stop()

                # Khởi động camera thread nếu chưa chạy
                if not hasattr(self, 'camera_thread') or not self.camera_thread.isRunning():
                    self.camera_thread = CameraThread()
                    self.camera_thread.frame_signal.connect(self.update_camera_view)
                    self.camera_thread.start()

            else:
                self.print_to_listview("Không lấy được frame màu từ camera.")
                pipeline.stop()
        except Exception as e:
            self.print_to_listview("Không phát hiện camera Intel RealSense D435i: " + str(e))

    def display_data(self, data):
        self.ui.pallet1_X.setPlainText(f"X = {data['1']['position']['x']}")
        self.ui.pallet1_Y.setPlainText(f"Y = {data['1']['position']['y']}")
        self.ui.pallet1_Z.setPlainText(f"Z = {data['1']['position']['z']}")
        self.ui.pallet1_theta.setPlainText(u"\u03b8 = " + f"{data['1']['position']['yaw']}")

        self.ui.pallet2_X.setPlainText(f"X = {data['2']['position']['x']}")
        self.ui.pallet2_Y.setPlainText(f"Y = {data['2']['position']['y']}")
        self.ui.pallet2_Z.setPlainText(f"Z = {data['2']['position']['z']}")
        self.ui.pallet2_theta.setPlainText(u"\u03b8 = " + f"{data['2']['position']['yaw']}")

        self.ui.pallet3_X.setPlainText(f"X = {data['3']['position']['x']}")
        self.ui.pallet3_Y.setPlainText(f"Y = {data['3']['position']['y']}")
        self.ui.pallet3_Z.setPlainText(f"Z = {data['3']['position']['z']}")
        self.ui.pallet3_theta.setPlainText(u"\u03b8 = " + f"{data['3']['position']['yaw']}")

        self.ui.pallet4_check.setText("In development")
        self.ui.pallet1_check.setChecked(True)
        self.ui.pallet2_check.setChecked(True)
        self.ui.pallet3_check.setChecked(True)

        self.pallet1Calib_x = data['1']['calibration']['calib_x']
        self.pallet1Calib_y = data['1']['calibration']['calib_y']
        self.pallet1Offset_theta = data['1']['calibration']['offset_theta']
        self.pallet2Calib_x = data['2']['calibration']['calib_x']
        self.pallet2Calib_y = data['2']['calibration']['calib_y']
        self.pallet2Offset_theta = data['2']['calibration']['offset_theta']
        self.pallet3Calib_x = data['3']['calibration']['calib_x']
        self.pallet3Calib_y = data['3']['calibration']['calib_y']
        self.pallet3Offset_theta = data['3']['calibration']['offset_theta']
        self.zeroPoint_X = data['default']['x']
        self.zeroPoint_Y = data['default']['y']
        self.zeroPoint_Z = data['default']['z']
        self.zeroPoint_theta = data['default']['yaw']
        self.homePoint_X = data['home']['x']
        self.homePoint_Y = data['home']['y']
        self.homePoint_Z = data['home']['z']
        self.homeTheta = data['home']['yaw']

    def update_system_status(self, time_str, date_str, thread_count, cpu_percent, ram_percent):
        self.ui.RealTime.setPlainText(time_str)
        self.ui.Date.setPlainText(date_str)
        self.ui.threadcount.setPlainText(str(thread_count))

        # Giới hạn max cho progressbar (0-100)
        self.ui.cpu_bar.setValue(int(cpu_percent))
        self.ui.ram_bar.setValue(int(ram_percent))

    def update_camera_view(self, image):
        pixmap = QPixmap.fromImage(image)
        self.ui.cameraView.setPixmap(pixmap.scaled(
            self.ui.cameraView.width(),
            self.ui.cameraView.height()
        ))

    def update_robot_position(self, x, y, z, yaw):
        self.ui.xarm_X.setPlainText(f"X = {x:.2f}")
        self.ui.xarm_Y.setPlainText(f"Y = {y:.2f}")
        self.ui.xarm_Z.setPlainText(f"Z = {z:.2f}")
        self.ui.xarm_theta.setPlainText(u"\u03b8 = " + f"{yaw:.2f}")
    def moveZeroPoint(self):
        try:
            self.moveHomePoint()

            command = {
                'type': 'move',
                'params': {
                    'x': self.zeroPoint_X,
                    'y': self.zeroPoint_Y,
                    'z': self.zeroPoint_Z,
                    'yaw': self.zeroPoint_theta
                }
            }
            self.robot_control.command_queue.put(command)
            self.print_to_listview("Đã gửi lệnh: Di chuyển đến Zero Point")
        except Exception as e:
            self.print_to_listview(f"Lỗi khi di chuyển đến Zero Point: {str(e)}")


    def moveHomePoint(self):
        try:
            command = {
                'type': 'move',
                'params': {
                    'x': self.homePoint_X,
                    'y': self.homePoint_Y,
                    'z': self.homePoint_Z,
                    'yaw': self.homeTheta
                }
            }
            self.robot_control.command_queue.put(command)
            self.print_to_listview("Đã gửi lệnh: Di chuyển đến Home Point")
        except Exception as e:
            self.print_to_listview(f"Lỗi khi di chuyển đến Home Point: {str(e)}")


    def sortingPallet(self):
        self.moveHomePoint()  # về vị trí gốc trước

        try:
            # Đọc tọa độ từ giao diện
            x = float(self.ui.pallet1_X.toPlainText().split(" ")[2])
            y = float(self.ui.pallet1_Y.toPlainText().split(" ")[2])
            z = float(self.ui.pallet1_Z.toPlainText().split(" ")[2])
            yaw = float(self.ui.pallet1_theta.toPlainText().split(" ")[2])

            command = {
                'type': 'move',
                'params': {
                    'x': x,
                    'y': y,
                    'z': z,
                    'yaw': yaw  # Nếu bạn muốn cố định yaw = 0 thì sửa thành 'yaw': 0
                }
            }

            self.robot_control.command_queue.put(command)
            self.print_to_listview(f"Đã gửi lệnh: Di chuyển đến pallet tại ({x:.1f}, {y:.1f}, {z:.1f}, yaw={yaw:.1f})")

        except Exception as e:
            self.print_to_listview("Lỗi khi đọc tọa độ pallet: " + str(e))

    def handle_centroid(self, x, y, yaw):
        if not hasattr(self, 'last_centroid'):
            self.last_centroid = (x, y)
            self.centroid_start_time = time.time()
            return

        dx = abs(x - self.last_centroid[0])
        dy = abs(y - self.last_centroid[1])

        if dx < 5 and dy < 5:
            if time.time() - self.centroid_start_time >= 1.0:
                target_x = x + self.workspace_calib['calib_x']
                target_y = y + self.workspace_calib['calib_y']
                target_z = 175
                target_yaw = yaw + self.workspace_calib['offset_theta']

                self.robot_control.set_position(x=target_x, y=target_y, z=target_z, yaw=target_yaw)
                self.print_to_listview(f"[ĐÃ ỔN ĐỊNH] Di chuyển đến: ({target_x:.1f}, {target_y:.1f})")
                self.robot_control.open_gripper()
                time.sleep(1)
                self.robot_control.set_position(x=target_x, y=target_y, z=160, yaw=target_yaw)
                self.robot_control.close_gripper()
                time.sleep(0.75)
                self.robot_control.set_position(x=target_x, y=target_y, z=400, yaw=target_yaw)

                self.centroid_start_time = time.time() + 5  # thêm delay
        else:
            self.last_centroid = (x, y)
            self.centroid_start_time = time.time()

 


    def MainProcess(self):
        self.sortingPallet()

        # ===== Khởi động ArucoDetectThread =====
        self.aruco_thread = ArucoDetectThread(self)
        self.aruco_thread.image_signal.connect(self.update_camera_view)  # để hiển thị hình ảnh
        self.aruco_thread.position_signal.connect(self.handle_centroid)  # để xử lý centroid
        self.aruco_thread.start()

        # ===== Khởi động BoxClassifierThread =====
        self.classifier_thread = BoxClassifierThread(self)
        self.classifier_thread.result_signal.connect(self.print_to_listview)
        self.classifier_thread.start()
        self.classifier_thread.label_signal.connect(self.update_predicted_label)

    def update_predicted_label(self, label):
        self.current_predicted_id = label




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
