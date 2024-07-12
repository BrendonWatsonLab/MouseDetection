#!/usr/bin/env python3
import cv2
import csv
import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox, QVBoxLayout, QProgressBar, QMessageBox, QScrollArea, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRect, QPoint, QSize
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage
import numpy as np
import argparse
import os
import re
import datetime
import time
from datetime import datetime

class Worker(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self, callable, *args, **kwargs):
        super().__init__()
        self.callable = callable
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.callable(*self.args, **self.kwargs)

class ClickableLabel(QLabel): # this class is for the ROI selection
    def __init__(self, *args, **kwargs):
        super(ClickableLabel, self).__init__(*args, **kwargs)
        self.setMinimumSize(640, 480)  # Set minimum size for the label
        self.setMaximumSize(1280, 960)
        self.origin = QPoint()
        self.current_rect = QRect()
        self.is_selecting = False

    def mousePressEvent(self, event):
        self.origin = event.pos()
        self.current_rect = QRect(self.origin, QSize())
        self.is_selecting = True
        self.update()  # Trigger paint event

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.current_rect.setBottomRight(event.pos())
            self.update()  # Trigger paint event

    def mouseReleaseEvent(self, event):
        self.is_selecting = False

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.current_rect.isNull():
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
            painter.drawRect(self.current_rect.normalized())

class ActigraphyProcessorApp(QWidget):
    def __init__(self, actigraphy_processor):
        super().__init__()
        self.roi = None
        self.actigraphy_processor = actigraphy_processor
        self.output_directory = None
        self.thread = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.scroll_area = QScrollArea()  # Create a new QScrollArea
        self.scroll_area.setWidgetResizable(True)
        layout = QVBoxLayout()

        # creating buttons for the GUI layout
        self.video_file_label = QLabel("Video File:")
        self.video_file_edit = QLineEdit()
        self.video_file_button = QPushButton("Browse Files")
        self.video_file_button.clicked.connect(self.browse_video_file)

        self.video_folder_label = QLabel("Video Folder:")
        self.video_folder_edit = QLineEdit()
        self.video_folder_button = QPushButton("Browse Folders")
        self.video_folder_button.clicked.connect(self.browse_video_folder)

        self.min_size_threshold_label = QLabel("Minimum Size Threshold:")
        self.min_size_threshold_edit = QLineEdit("30")

        self.global_threshold_label = QLabel("Global Threshold:")
        self.global_threshold_edit = QLineEdit("35")

        self.percentage_threshold_label = QLabel("Percentage Threshold:")
        self.percentage_threshold_edit = QLineEdit("160")

        self.dilation_kernel_label = QLabel("Dilation Kernel:")
        self.dilation_kernel_edit = QLineEdit("3")

        self.oaf_check = QCheckBox("Override Files")
        self.name_stamp_check = QCheckBox("Use Name Stamp")
        self.name_stamp_check.setChecked(True)

        self.start_button = QPushButton("Start Detection", self)
        self.start_button.clicked.connect(self.run)

        self.progress_bar = QProgressBar(self)

        self.output_directory_label = QLabel("Output CSV File:")
        self.output_directory_edit = QLineEdit()
        self.output_directory_button = QPushButton("Select Output File Destination")
        self.output_directory_button.clicked.connect(self.select_output_file_destination)

        # ROI Manual input fields and button
        self.manual_roi_label = QLabel("Manual ROI Coordinates (x, y, w, h):")
        self.manual_roi_x_edit = QLineEdit()
        self.manual_roi_y_edit = QLineEdit()
        self.manual_roi_w_edit = QLineEdit()
        self.manual_roi_h_edit = QLineEdit()
        self.manual_roi_confirm_btn = QPushButton("Confirm Manual ROI", self)
        self.manual_roi_confirm_btn.clicked.connect(self.confirm_manual_roi)

        # ROI stuff
        self.btn_confirm_roi = QPushButton("Confirm ROI", self)
        self.btn_confirm_roi.clicked.connect(self.confirm_roi)
        self.video_display_label = ClickableLabel()
        self.roi_status_label = QLabel("ROI not set", self)

        #formally adds all widgets
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.video_file_label)
        layout.addWidget(self.video_file_edit)
        layout.addWidget(self.video_file_button)
        layout.addWidget(self.video_folder_label)
        layout.addWidget(self.video_folder_edit)
        layout.addWidget(self.video_folder_button)
        layout.addWidget(self.min_size_threshold_label)
        layout.addWidget(self.min_size_threshold_edit)
        layout.addWidget(self.global_threshold_label)
        layout.addWidget(self.global_threshold_edit)
        layout.addWidget(self.percentage_threshold_label)
        layout.addWidget(self.percentage_threshold_edit)
        layout.addWidget(self.dilation_kernel_label)
        layout.addWidget(self.dilation_kernel_edit)
        layout.addWidget(self.oaf_check)
        layout.addWidget(self.name_stamp_check)
        layout.addWidget(self.start_button)
        layout.addWidget(self.output_directory_label)
        layout.addWidget(self.output_directory_edit)
        layout.addWidget(self.output_directory_button)

        # ROI widgets
        layout.addWidget(self.manual_roi_label)
        layout.addWidget(QLabel("x:"))
        layout.addWidget(self.manual_roi_x_edit)
        layout.addWidget(QLabel("y:"))
        layout.addWidget(self.manual_roi_y_edit)
        layout.addWidget(QLabel("w:"))
        layout.addWidget(self.manual_roi_w_edit)
        layout.addWidget(QLabel("h:"))
        layout.addWidget(self.manual_roi_h_edit)
        layout.addWidget(self.manual_roi_confirm_btn)
        layout.addWidget(self.video_display_label)
        layout.addWidget(self.btn_confirm_roi)
        layout.addWidget(self.roi_status_label)

        # Create a container widget for the layout
        container = QWidget()
        container.setLayout(layout)

        # Set the layout container as the scroll area's widget
        self.scroll_area.setWidget(container)

        # Create a new layout to hold the scroll area
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scroll_area)

        # Set the main layout for the window
        self.setLayout(main_layout)
        self.setWindowTitle('Mouse Detection-inator')
    
        # Set the minimum width and maximum height of the window
        self.setMinimumWidth(800)
        self.setMaximumHeight(600)
    
    def select_output_file_destination(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",  # You can specify a default path here
            options=options
        )
        if directory:
            # Assuming you want to set the output directory to a class member
            self.output_directory = directory
            self.output_directory_edit.setText(directory)

    def browse_video_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Video File', '', 'MP4 files (*.mp4)')
        self.video_file_edit.setText(file_name)
        
        if file_name:
            cap = cv2.VideoCapture(file_name)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.original_frame = frame  # Store the original frame
                self.display_frame(frame)  # Display this frame on video_display_label
                self.btn_confirm_roi.setEnabled(True)  # Enable the Confirm ROI button

    def browse_video_folder(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Open Video Folder')
        self.video_folder_edit.setText(dir_name)
        
        if dir_name:
            mp4_files = [f for f in os.listdir(dir_name) if f.endswith('.mp4')]
            
            if mp4_files:
                first_video_file = os.path.join(dir_name, mp4_files[0])
                cap = cv2.VideoCapture(first_video_file)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    self.original_frame = frame  # Store the original frame
                    self.display_frame(frame)  # Display this frame on video_display_label
                    self.btn_confirm_roi.setEnabled(True)  # Enable the Confirm ROI button
                else:
                    QMessageBox.warning(self, "Error", "Could not read the first frame of the first video file.")
            else:
                QMessageBox.warning(self, "Error", "No MP4 files found in the selected folder.")

    def display_frame(self, frame):
        """Display the frame in the ClickableLabel."""
        qt_img = self.convert_cv_qt(frame)
        self.video_display_label.setPixmap(qt_img)

        if self.roi is not None:
            painter = QPainter(self.video_display_label.pixmap())
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            scaling_factor_width = self.video_display_label.width() / frame.shape[1]
            scaling_factor_height = self.video_display_label.height() / frame.shape[0]
            scaled_roi = (
                int(self.roi[0] * scaling_factor_width),
                int(self.roi[1] * scaling_factor_height),
                int(self.roi[2] * scaling_factor_width),
                int(self.roi[3] * scaling_factor_height)
            )
            painter.drawRect(*scaled_roi)
            painter.end()

    def run(self):
        video_file = self.video_file_edit.text()
        video_folder = self.video_folder_edit.text()

        try:
            min_size_threshold = int(self.min_size_threshold_edit.text())
            global_threshold = int(self.global_threshold_edit.text())
            percentage_threshold = int(self.percentage_threshold_edit.text())
            dilation_kernel = int(self.dilation_kernel_edit.text())
        except ValueError as ve:
            print("Please enter valid integer values for thresholds and dilation kernel.", ve)
            self.start_button.setEnabled(True)
            return

        oaf = self.oaf_check.isChecked()
        name_stamp = self.name_stamp_check.isChecked()

        self.actigraphy_processor.set_processing_parameters(global_threshold, min_size_threshold, percentage_threshold, dilation_kernel)

        output_file_path = self.output_directory_edit.text().strip()
        self.actigraphy_processor.output_file_path = output_file_path if output_file_path else None

        if video_file and self.roi is not None:
            self.worker = Worker(
                self.actigraphy_processor.process_single_video_file,
                video_file, name_stamp, self.roi, self.output_directory
            )
            self.worker.kwargs['progress_callback'] = self.worker.progress_signal
            self.worker.progress_signal.connect(self.update_progress_bar)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.start()
        elif video_folder and self.roi is not None:
            self.worker = Worker(
                self.actigraphy_processor.process_video_files, 
                video_folder, oaf, name_stamp, self.roi, self.output_directory
            )
            self.worker.kwargs['progress_callback'] = self.worker.progress_signal
            self.worker.progress_signal.connect(self.update_folder_progress_bar)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.start()
        else:
            print("No video file or folder has been selected, or ROI not set.")
            self.start_button.setEnabled(True)
        
    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def update_folder_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def on_processing_finished(self):
        self.progress_bar.setValue(100)
        self.roi_status_label.setText("ROI not set")
        self.roi_status_label.setStyleSheet("")
        self.start_button.setEnabled(True)
        self.btn_confirm_roi.setEnabled(False)
        print("Actigraphy processing has been completed.")
        QMessageBox.information(self, "Actigraphy Processing", "Actigraphy processing has been completed.")

    def confirm_roi(self):
        if not self.video_display_label.current_rect.isNull():
            rect = self.video_display_label.current_rect.normalized()
            scaling_factor_width = self.original_frame.shape[1] / self.video_display_label.width()
            scaling_factor_height = self.original_frame.shape[0] / self.video_display_label.height()
            self.roi = (
                int(rect.x() * scaling_factor_width),
                int(rect.y() * scaling_factor_height),
                int(rect.width() * scaling_factor_width),
                int(rect.height() * scaling_factor_height)
            )
            self.roi_status_label.setText("ROI set. Ready to start!")
            self.roi_status_label.setStyleSheet("color: green;")
            self.display_frame(self.original_frame)  # Call without highlight_roi
            self.start_button.setEnabled(True)

    def confirm_manual_roi(self):
        try:
            x = int(self.manual_roi_x_edit.text())
            y = int(self.manual_roi_y_edit.text())
            w = int(self.manual_roi_w_edit.text())
            h = int(self.manual_roi_h_edit.text())
            
            if x >= 0 and y >= 0 and w > 0 and h > 0:
                self.roi = (x, y, w, h)
                self.roi_status_label.setText("Manual ROI set. Ready to start!")
                self.roi_status_label.setStyleSheet("color: green;")
                self.display_frame(self.original_frame)  # Call without highlight_roi
                self.start_button.setEnabled(True)
            else:
                raise ValueError("Coordinates must be non-negative and width/height must be positive.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid ROI coordinates: {e}")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_display_label.width(), self.video_display_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class ActigraphyProcessor:
    def __init__(self):
        self.roi_pts=None
        self.output_file_path=None
        self.min_size_threshold = None
        self.global_threshold = None
        self.percentage_threshold = None
        self.dilation_kernel = None

    def set_processing_parameters(self, global_threshold, min_size_threshold, percentage_threshold, dilation_kernel):
            self.global_threshold = global_threshold
            self.min_size_threshold = min_size_threshold
            self.percentage_threshold = percentage_threshold
            self.dilation_kernel = dilation_kernel

    def get_nested_paths(self, root_dir):
        queue = [root_dir]
        paths = []
        print('Here are all the nested folders within the selected directory:')
        while queue:
            current_dir = queue.pop(0)
            paths.append(current_dir)
            print(current_dir)

            for child_dir in sorted(os.listdir(current_dir)):
                child_path = os.path.join(current_dir, child_dir)
                if os.path.isdir(child_path):
                    queue.append(child_path)

        return paths

    def list_mp4_files(self, directory_path, oaf):
        mp4_files = [f for f in os.listdir(directory_path) if f.endswith('.mp4')]
        csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
        
        if mp4_files:
            updated_mp4_files = []
            print("List of all the MP4 files in {}: ".format(directory_path))
            for mp4_file in mp4_files:
                print(mp4_file)
                if mp4_file[:-4] + "_actigraphy.csv" in csv_files:
                    print("Actigraphy file already found for {}.".format(mp4_file))
                    if oaf:
                        print("Overide Actigraphy Files set True, Redoing this file.")
                    else:
                        continue
                
                updated_mp4_files.append(mp4_file)
            mp4_files = updated_mp4_files
        else:
            print("No MP4 files found in {}.".format(directory_path))
        
        return mp4_files

    def process_single_video_file(self, video_file, name_stamp, roi, output_directory, progress_callback):

        # Determine whether to use creation time from the file name or os.path.getctime
        if name_stamp or name_stamp is None:
            print("Extracting creation time from the name.")
            creation_time = self._get_creation_time_from_name(video_file)
        else:
            print("Using the file's actual creation time.")
            creation_time = int(os.path.getctime(video_file)*1000)
            
        cap = cv2.VideoCapture(video_file)
        frame_number = 0

        # Automatically generate the output CSV file path based on the video file name
        outputfile_name = os.path.splitext(os.path.basename(video_file))[0] + "_actigraphy.csv"
        # If an output directory is provided, use it; otherwise, save next to the video file
        output_file_path = os.path.join(output_directory, outputfile_name) if output_directory else os.path.join(os.path.dirname(video_file), outputfile_name)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        is_rat_present = False
        prev_frame = None
        result_rows = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            frame_number += 1
            elapsed_millis = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Apply the defined ROI
            roi_frame = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

            if prev_frame is not None:
                motion_detected = self.detect_motion(
                roi_frame, prev_frame, 
                self.global_threshold, self.min_size_threshold, 
                self.percentage_threshold, self.dilation_kernel)            
                posix_time = int(creation_time + (elapsed_millis))

                if motion_detected and not is_rat_present:
                    is_rat_present = True
                    start_time = datetime.now()
                    start_time_posix = int(start_time.timestamp() * 1000)
                elif not motion_detected and is_rat_present:
                    is_rat_present = False
                    end_time = datetime.now()
                    end_time_posix = int(end_time.timestamp() * 1000)
                    result_rows.append((start_time_posix, end_time_posix))

            prev_frame = roi_frame

            if progress_callback and frame_number % 100 == 0:  # Updates every 100 frames for progress bar
                progress = (frame_number / total_frames) * 100
                progress_callback.emit(int(progress))

        # Write the POSIX timestamps to the CSV
        with open(output_file_path, 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(['Start Time (ms)', 'End Time (ms)'])
            for start, end in result_rows:
                writer.writerow([start, end])
        cap.release()
        print(f"Actigraphy processing completed for {video_file}")
        print("*" * 75)

    def process_video_files(self, video_folder, oaf, name_stamp, roi, output_directory, progress_callback=None):
        start_time = time.time()
        total_frames_processed = 0
        total_time_taken = 0

        nested_folders = self.get_nested_paths(video_folder)
        all_mp4_files = [
            os.path.join(folder, mp4_file)
            for folder in nested_folders
            for mp4_file in self.list_mp4_files(folder, oaf)
        ]
        total_files = len(all_mp4_files)
        files_processed = 0

        if total_files == 0:
            print("No video files to process.")
            return

        for mp4_file in all_mp4_files:
            file_start_time = time.time()
            self.process_single_video_file(mp4_file, name_stamp, roi, output_directory, None)
            file_end_time = time.time()
            total_time_taken += (file_end_time - file_start_time)
            files_processed += 1

            if progress_callback:
                progress = int((files_processed / total_files) * 100)
                progress_callback.emit(progress)

            cap = cv2.VideoCapture(mp4_file)
            total_frames_processed += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        
        end_time = time.time()
        total_time_taken = end_time - start_time
        time_per_frame = total_time_taken / total_frames_processed if total_frames_processed else float('inf')

        print("Total Time Taken for All Videos: {:.2f} seconds".format(total_time_taken))
        print("Total Frames Processed for All Videos: {}".format(total_frames_processed))
        print("Average Time Per Frame for All Videos: {:.4f} seconds".format(time_per_frame))
        
        if progress_callback:
            progress_callback.emit(100)

    def detect_motion(self, frame, prev_frame, global_threshold, min_size_threshold, percentage_threshold, dilation_kernel):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        abs_diff = np.abs(frame_gray.astype(np.float32) - prev_frame_gray.astype(np.float32))
        raw_diff = np.sum(abs_diff)
        rmse = np.sqrt(np.mean(abs_diff ** 2))
        return rmse > 1  # Adjust the pixel count threshold as needed

    @staticmethod
    def _get_creation_time_from_name(filename):
        regex_pattern_1 = r'(\d{8}_\d{9})'
        regex_pattern_2 = r'(\d{8}_\d{6})'
        
        # Try the first regex pattern
        match = re.search(regex_pattern_1, os.path.basename(filename))
        
        if match:
            # Extract the matched date and time
            date_time_str = match.group(1)
            #print(date_time_str)
            # Include milliseconds in the format
            date_time_format = '%Y%m%d_%H%M%S%f'
            
            # Convert the date and time string to a datetime object
            date_time_obj = datetime.strptime(date_time_str, date_time_format)
            
            # Get the POSIX timestamp in milliseconds
            posix_timestamp_ms = int(date_time_obj.timestamp() * 1000)
            
            return posix_timestamp_ms
        else:
            # If the first pattern didn't match, try the second pattern
            match = re.search(regex_pattern_2, os.path.basename(filename))
            
            if match:
                # Extract the matched date and time from the second pattern
                date_time_str = match.group(1)
                
                # Include milliseconds in the format
                date_time_format = '%Y%m%d_%H%M%S'
                
                # Convert the date and time string to a datetime object
                date_time_obj = datetime.strptime(date_time_str, date_time_format)
                
                # Get the POSIX timestamp in milliseconds
                posix_timestamp_ms = int(date_time_obj.timestamp() * 1000)
                
                return posix_timestamp_ms
            else:
                print("Failed to extract creation time from the file name. Using file generated time instead.")
                return int(os.path.getctime(filename)*1000)


if __name__ == "__main__":

    # Launching the PyQt5 application
    app = QApplication(sys.argv)
    
    actigraphy_processor = ActigraphyProcessor()  # Instantiate the main logic class

    # The ActigraphyProcessorApp now takes the main logic class as an argument
    window = ActigraphyProcessorApp(actigraphy_processor)
    window.show()

    sys.exit(app.exec_())