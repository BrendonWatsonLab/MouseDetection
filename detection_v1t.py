import cv2
import csv
import sys
import time
import numpy as np
import os
import re
import datetime
from datetime import datetime
import threading
import argparse
from tkinter import *
from tkinter import filedialog, messagebox, ttk

class ActigraphyProcessorApp:
    def __init__(self, root, actigraphy_processor):
        self.root = root
        self.roi = None
        self.actigraphy_processor = actigraphy_processor
        self.output_directory = None
        self.init_ui()
    
    def init_ui(self):
        self.root.title("Actigraphy")
        self.root.geometry("800x600")

        # Labels, Entries, Buttons and Checkbuttons
        self.video_file_label = Label(self.root, text="Video File:")
        self.video_file_label.pack()
        self.video_file_edit = Entry(self.root, width=100)
        self.video_file_edit.pack()
        self.video_file_button = Button(self.root, text="Browse Files", command=self.browse_video_file)
        self.video_file_button.pack()

        self.video_folder_label = Label(self.root, text="Video Folder:")
        self.video_folder_label.pack()
        self.video_folder_edit = Entry(self.root, width=100)
        self.video_folder_edit.pack()
        self.video_folder_button = Button(self.root, text="Browse Folders", command=self.browse_video_folder)
        self.video_folder_button.pack()
        
        self.min_size_threshold_label = Label(self.root, text="Minimum Size Threshold:")
        self.min_size_threshold_label.pack()
        self.min_size_threshold_edit = Entry(self.root, width=50)
        self.min_size_threshold_edit.pack()
        
        self.global_threshold_label = Label(self.root, text="Global Threshold:")
        self.global_threshold_label.pack()
        self.global_threshold_edit = Entry(self.root, width=50)
        self.global_threshold_edit.pack()
        
        self.percentage_threshold_label = Label(self.root, text="Percentage Threshold:")
        self.percentage_threshold_label.pack()
        self.percentage_threshold_edit = Entry(self.root, width=50)
        self.percentage_threshold_edit.pack()
        
        self.dilation_kernel_label = Label(self.root, text="Dilation Kernel:")
        self.dilation_kernel_label.pack()
        self.dilation_kernel_edit = Entry(self.root, width=50)
        self.dilation_kernel_edit.pack()
        
        self.oaf_check = IntVar()
        self.name_stamp_check = IntVar()
        self.oaf_check_button = Checkbutton(self.root, text="Override Actigraphy Files", variable=self.oaf_check)
        self.oaf_check_button.pack()
        self.name_stamp_check_button = Checkbutton(self.root, text="Use Name Stamp", variable=self.name_stamp_check)
        self.name_stamp_check_button.pack()
        self.name_stamp_check_button.select()

        self.start_button = Button(self.root, text="Start Detection", command=self.run)
        self.start_button.pack()
        
        self.progress_bar = ttk.Progressbar(self.root, orient=HORIZONTAL, length=500, mode='determinate')
        self.progress_bar.pack(pady=20)

        self.output_directory_label = Label(self.root, text="Output CSV File:")
        self.output_directory_label.pack()
        self.output_directory_edit = Entry(self.root, width=100)
        self.output_directory_edit.pack()
        self.output_directory_button = Button(self.root, text="Select Output File Destination", command=self.select_output_file_destination)
        self.output_directory_button.pack()

        self.video_display_label = Label(self.root)
        self.video_display_label.pack()
        
        self.btn_confirm_roi = Button(self.root, text="Confirm ROI", command=self.confirm_roi)
        self.btn_confirm_roi.pack()
        self.roi_status_label = Label(self.root, text="ROI not set")
        self.roi_status_label.pack()

        # Initialize video capture variables
        self.cap = None
        self.original_frame = None

    def select_output_file_destination(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_directory = directory
            self.output_directory_edit.delete(0, END)
            self.output_directory_edit.insert(0, directory)

    def browse_video_file(self):
        file_name = filedialog.askopenfilename(filetypes=[('MP4 files', '*.mp4')])
        self.video_file_edit.delete(0, END)
        self.video_file_edit.insert(0, file_name)
        
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            ret, frame = self.cap.read()
            if ret:
                self.original_frame = frame
                self.display_frame(frame)

    def browse_video_folder(self):
        dir_name = filedialog.askdirectory()
        self.video_folder_edit.delete(0, END)
        self.video_folder_edit.insert(0, dir_name)

    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_display_label.imgtk = imgtk
        self.video_display_label.configure(image=imgtk)

    def run(self):
        video_file = self.video_file_edit.get()
        video_folder = self.video_folder_edit.get()
        
        try:
            min_size_threshold = int(self.min_size_threshold_edit.get())
            global_threshold = int(self.global_threshold_edit.get())
            percentage_threshold = int(self.percentage_threshold_edit.get())
            dilation_kernel = int(self.dilation_kernel_edit.get())
        except ValueError as ve:
            messagebox.showwarning("Invalid Input", "Please enter valid integer values for thresholds and dilation kernel.")
            self.start_button["state"] = "normal"
            return
        
        oaf = self.oaf_check.get()
        name_stamp = self.name_stamp_check.get()
        
        self.actigraphy_processor.set_processing_parameters(global_threshold, min_size_threshold, percentage_threshold, dilation_kernel)

        output_file_path = self.output_directory_edit.get().strip()
        if output_file_path:
            self.actigraphy_processor.output_file_path = output_file_path
        else:
            self.actigraphy_processor.output_file_path = None
        
        if video_file and self.roi is not None:
            self.thread = threading.Thread(target=self.actigraphy_processor.process_single_video_file, args=(
                video_file, name_stamp, self.roi, self.output_directory, self.update_progress_bar))
            self.thread.start()
        elif video_folder:
            self.thread = threading.Thread(target=self.actigraphy_processor.process_video_files, args=(
                video_folder, oaf, self.roi, name_stamp, self.output_directory, self.update_progress_bar))
            self.thread.start()
        else:
            messagebox.showwarning("Input Error", "No video file or folder has been selected.")
            self.start_button["state"] = "normal"

    def update_progress_bar(self, value):
        self.progress_bar["value"] = value
        self.root.update_idletasks()

    def confirm_roi(self):
        if not self.video_display_label.image:
            messagebox.showwarning("ROI Error", "Please select a video file first.")
            return
        
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi = [(x, y)]
            elif event == cv2.EVENT_LBUTTONUP:
                self.roi.append((x, y))
                cv2.rectangle(param, self.roi[0], self.roi[1], (0, 255, 0), 2)
                cv2.imshow("ROI selection", param)
        
        clone = self.original_frame.copy()
        cv2.imshow("ROI selection", clone)
        cv2.setMouseCallback("ROI selection", on_mouse, clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if len(self.roi) == 2:
            self.roi = (self.roi[0][0], self.roi[0][1], self.roi[1][0] - self.roi[0][0], self.roi[1][1] - self.roi[0][1])
            self.roi_status_label.config(text="ROI set. Ready to start!", fg="green")

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
            print(f"List of all the MP4 files in {directory_path}:")
            for mp4_file in mp4_files:
                print(mp4_file)
                if mp4_file[:-4] + "_actigraphy.csv" in csv_files:
                    print(f"Actigraphy file already found for {mp4_file}.")
                    if oaf:
                        print("Overide Actigraphy Files set True, Redoing this file.")
                    else:
                        continue
                updated_mp4_files.append(mp4_file)
            mp4_files = updated_mp4_files
        else:
            print(f"No MP4 files found in {directory_path}.")
        return mp4_files

    def process_single_video_file(self, video_file, name_stamp, roi, output_directory, progress_callback):
        if name_stamp:
            print("Extracting creation time from the name.")
            creation_time = self._get_creation_time_from_name(video_file)
        else:
            print("Using the file's actual creation time.")
            creation_time = int(os.path.getctime(video_file) * 1000)
            
        cap = cv2.VideoCapture(video_file)
        frame_number = 0

        outputfile_name = os.path.splitext(os.path.basename(video_file))[0] + "_actigraphy.csv"
        output_file_path = os.path.join(output_directory, outputfile_name) if output_directory else os.path.join(os.path.dirname(video_file), outputfile_name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        is_rat_present = False
        prev_frame = None
        result_rows = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            elapsed_millis = cap.get(cv2.CAP_PROP_POS_MSEC)
            roi_frame = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            if prev_frame is not None:
                motion_detected = self.detect_motion(
                    roi_frame, prev_frame, 
                    self.global_threshold, self.min_size_threshold, 
                    self.percentage_threshold, self.dilation_kernel
                )
                posix_time = int(creation_time + elapsed_millis)
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

            if progress_callback and frame_number % 100 == 0:
                progress = (frame_number / total_frames) * 100
                progress_callback(progress)

        print("Result rows:", result_rows)
        with open(output_file_path, 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(['Start Time (ms)', 'End Time (ms)'])
            for start, end in result_rows:
                writer.writerow([start, end])
        cap.release()
        print(f"Actigraphy processing completed for {video_file}")
        print("*" * 75)

    def process_video_files(self, video_folder, oaf, set_roi, name_stamp, output_directory, progress_callback=None):
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

        # Initialize roi_pts here if set_roi is True and the ROI hasn't been set yet
        if set_roi and not self.roi_pts and all_mp4_files:
            first_video_file = all_mp4_files[0]
            cap = cv2.VideoCapture(first_video_file)
            if cap.isOpened():
                self.roi_pts = self._select_roi_from_first_frame(cap)
                cap.release()
            else:
                print(f"Failed to open the first video file: {first_video_file}")
                return

        for mp4_file in all_mp4_files:
            file_start_time = time.time()
            self.process_single_video_file(mp4_file, name_stamp, set_roi, output_directory, None, self.roi_pts)
            file_end_time = time.time()
            file_time_taken = file_end_time - file_start_time
            total_time_taken += file_time_taken

            files_processed += 1
            if progress_callback:
                folder_progress = int((files_processed / total_files) * 100)
                progress_callback(folder_progress)

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
            progress_callback(100)

    def detect_motion(self, frame, prev_frame, global_threshold, min_size_threshold, percentage_threshold, dilation_kernel):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        abs_diff = np.abs(frame_gray.astype(np.float32) - prev_frame_gray.astype(np.float32))
        raw_diff = np.sum(abs_diff)
        rmse = np.sqrt(np.mean(abs_diff ** 2))
        print(rmse)
        return rmse > 1  # Adjust the pixel count threshold as needed

    @staticmethod
    def _get_creation_time_from_name(filename):
        regex_pattern_1 = r'(\d{8}_\d{9})'
        regex_pattern_2 = r'(\d{8}_\d{6})'
        
        match = re.search(regex_pattern_1, os.path.basename(filename))
        if match:
            date_time_str = match.group(1)
            date_time_format = '%Y%m%d_%H%M%S%f'
            date_time_obj = datetime.strptime(date_time_str, date_time_format)
            posix_timestamp_ms = int(date_time_obj.timestamp() * 1000)
            return posix_timestamp_ms
        else:
            match = re.search(regex_pattern_2, os.path.basename(filename))
            if match:
                date_time_str = match.group(1)
                date_time_format = '%Y%m%d_%H%M%S'
                date_time_obj = datetime.strptime(date_time_str, date_time_format)
                posix_timestamp_ms = int(date_time_obj.timestamp() * 1000)
                return posix_timestamp_ms
            else:
                print("Failed to extract creation time from the file name. Using file generated time instead.")
                return int(os.path.getctime(filename) * 1000)

if __name__ == "__main__":
    root = Tk()
    actigraphy_processor = ActigraphyProcessor()
    app = ActigraphyProcessorApp(root, actigraphy_processor)
    root.mainloop()