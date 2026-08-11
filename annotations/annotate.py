"""
Author: Giacomo D'Amicantonio
"""

import cv2
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import threading
import time
from PIL import Image, ImageTk
import numpy as np
import queue
import csv
import ast
# --- Constants ---
# Use a queue for thread-safe communication between video decoder and GUI
FRAME_QUEUE = queue.Queue(maxsize=10)

class VideoAnnotator:
    def __init__(self, file_names, base_path):
        self.file_names = file_names
        self.base_path = base_path
        self.current_video_idx = 0
        self.cap = None
        self.total_frames = 0
        self.fps = 30

        # --- Playback and Threading Control ---
        self.is_playing = False
        self.playback_speed = 3.0
        self.video_thread = None
        self.stop_thread = threading.Event()
        # Use a dedicated queue for commands to avoid thread-local issues.
        self.command_queue = queue.Queue()

        # --- Annotation Data Structures ---
        self.annotations = []
        self.bounding_boxes = {}
        
        # --- State Variables ---
        self.start_frame = 0
        self.end_frame = 0
        self.current_clip_id = None
        # NEW: Keep track of the active point annotation to replace it
        self.active_point_id = None
        
        # --- Bounding Box Drawing State ---
        self.is_drawing_bb = False
        self.current_bb_points = []
        
        # --- Frame and Scaling Variables ---
        self.original_frame_dims = (0, 0)

        # --- GUI Setup ---
        self.root = tk.Tk()
        self.root.title("Advanced Video Annotator")
        self.root.geometry("900x800")
        self.setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        # Main layout frames
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Use a container frame for the video to stabilize resizing
        self.video_container_frame = ttk.Frame(self.root, style='Black.TFrame')
        self.video_container_frame.pack(fill=tk.BOTH, expand=True)

        s = ttk.Style()
        s.configure('Black.TFrame', background='black')
        
        self.video_label = ttk.Label(self.video_container_frame, background='black')
        # Use .place() to prevent the label's size from influencing the container's size
        self.video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.video_label.bind("<Button-1>", self.on_video_click)

        controls_frame = ttk.Frame(self.root)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # --- Video and File Controls ---
        file_controls = ttk.LabelFrame(top_frame, text="Video Controls")
        file_controls.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        self.video_info_label = ttk.Label(file_controls, text="No video loaded.", width=40)
        self.video_info_label.pack(pady=5, padx=5)
        ttk.Button(file_controls, text="◄ Prev Video", command=self.prev_video).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_controls, text="Next Video ►", command=self.next_video).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_controls, text="Load Annotations", command=self.load_annotations_from_file).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_controls, text="Save Annotations", command=self.export_annotations_to_json).pack(fill=tk.X, padx=5, pady=2)
        
        # --- Playback Controls ---
        playback_controls = ttk.LabelFrame(top_frame, text="Playback")
        playback_controls.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        self.play_pause_btn = ttk.Button(playback_controls, text="▶ Play", command=self.toggle_play)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        speed_frame = ttk.Frame(playback_controls)
        speed_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.speed_label = ttk.Label(speed_frame, text=f"Speed: {self.playback_speed:.1f}x")
        self.speed_label.pack()
        self.speed_var = tk.DoubleVar(value=self.playback_speed)
        speed_scale = ttk.Scale(speed_frame, from_=0.5, to=10.0, variable=self.speed_var, orient=tk.HORIZONTAL, command=self.update_speed)
        speed_scale.pack()

        # --- Scrubber / Frame Slider ---
        scrubber_frame = ttk.Frame(top_frame)
        scrubber_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.current_frame_label = ttk.Label(scrubber_frame, text="Frame: 0 / 0")
        self.current_frame_label.pack()
        self.frame_var = tk.IntVar()
        self.frame_scale = ttk.Scale(scrubber_frame, from_=0, to=100, variable=self.frame_var, orient=tk.HORIZONTAL, command=self.seek_frame)
        self.frame_scale.pack(fill=tk.X, expand=True)

        # --- Annotation Controls (Bottom Frame) ---
        temporal_frame = ttk.LabelFrame(controls_frame, text="1. Temporal Anomaly")
        temporal_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(temporal_frame, text="Set Start Frame", command=self.set_start_frame).pack(pady=2, padx=5)
        self.start_frame_label = ttk.Label(temporal_frame, text="Start: 0")
        self.start_frame_label.pack(pady=2, padx=5)
        ttk.Button(temporal_frame, text="Set End Frame", command=self.set_end_frame).pack(pady=2, padx=5)
        self.end_frame_label = ttk.Label(temporal_frame, text="End: 0")
        self.end_frame_label.pack(pady=2, padx=5)
        ttk.Button(temporal_frame, text="Save Temporal Clip", command=self.save_temporal_annotation).pack(pady=5, padx=5)

        bbox_frame = ttk.LabelFrame(controls_frame, text="2. Bounding Box")
        bbox_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.draw_bb_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bbox_frame, text="Draw BBox", variable=self.draw_bb_var, command=self.toggle_bb_drawing).pack(anchor=tk.W, padx=5)
        
        bbox_validity_frame = ttk.Frame(bbox_frame)
        bbox_validity_frame.pack(pady=5, padx=5)
        
        ttk.Label(bbox_validity_frame, text="Valid From:").grid(row=0, column=0, padx=2, pady=2, sticky='w')
        self.bb_start_var = tk.IntVar()
        ttk.Entry(bbox_validity_frame, textvariable=self.bb_start_var, width=7).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(bbox_validity_frame, text="Set", command=self.set_bbox_start_frame).grid(row=0, column=2, padx=2, pady=2)

        ttk.Label(bbox_validity_frame, text="Valid To:").grid(row=1, column=0, padx=2, pady=2, sticky='w')
        self.bb_end_var = tk.IntVar()
        ttk.Entry(bbox_validity_frame, textvariable=self.bb_end_var, width=7).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(bbox_validity_frame, text="Set", command=self.set_bbox_end_frame).grid(row=1, column=2, padx=2, pady=2)

        ttk.Button(bbox_frame, text="Add Bounding Box", command=self.add_bounding_box).pack(pady=5)
        ttk.Button(bbox_frame, text="Clear Current Points", command=self.clear_bb_points).pack(pady=2)

        display_frame = ttk.LabelFrame(controls_frame, text="Saved Annotations")
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.annotations_tree = ttk.Treeview(display_frame, columns=('File', 'Clip', 'BBoxes'), show='headings')
        self.annotations_tree.heading('File', text='File')
        self.annotations_tree.heading('Clip', text='Clip/Center Frame')
        self.annotations_tree.heading('BBoxes', text='# BBoxes')
        self.annotations_tree.column('File', width=120)
        self.annotations_tree.column('Clip', width=120, anchor=tk.CENTER)
        self.annotations_tree.column('BBoxes', width=60, anchor=tk.CENTER)
        self.annotations_tree.pack(fill=tk.BOTH, expand=True)
        self.annotations_tree.bind('<<TreeviewSelect>>', self.on_annotation_select)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit? This will export annotations to 'annotations_export.json'."):
            self.stop_thread.set()
            if self.video_thread: self.video_thread.join()
            self.export_annotations_to_json(show_success_message=False) # Don't show success on quit
            if self.cap: self.cap.release()
            self.root.destroy()

    # --- Video Handling and Threaded Playback (Producer/Consumer Model) ---
    
    def _load_video(self, video_idx):
        if not (0 <= video_idx < len(self.file_names)): return False
        
        self.stop_thread.set()
        if self.video_thread and self.video_thread.is_alive(): self.video_thread.join()
        if self.cap: self.cap.release()
        
        self.current_video_idx = video_idx
        video_path = os.path.join(self.base_path, self.file_names[self.current_video_idx])
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open video: {video_path}")
            return False
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.original_frame_dims = (width, height)

        self._update_gui_for_new_video()
        
        self.stop_thread.clear()
        self.command_queue.put(('seek', 0))
        self.video_thread = threading.Thread(target=self._video_producer_loop, daemon=True)
        self.video_thread.start()
        self._gui_consumer_loop()
        return True

    def _video_producer_loop(self):
        """(RUNS ON BACKGROUND THREAD) Decodes frames and puts them in a queue."""
        while not self.stop_thread.is_set():
            # Handle commands from the GUI thread using a queue
            try:
                command, value = self.command_queue.get_nowait()
                if command == 'seek':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
                    with FRAME_QUEUE.mutex: FRAME_QUEUE.queue.clear()
                    # After seeking, immediately produce one frame for display
                    ret, frame = self.cap.read()
                    if ret:
                        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        # Use block=True to ensure the seek frame gets through
                        FRAME_QUEUE.put((current_pos, frame), block=True, timeout=1)
            except queue.Empty:
                pass # No commands
            except (queue.Full, AttributeError): # Catch if queue is full or cap is released
                continue 

            if not self.is_playing or not self.cap or not self.cap.isOpened():
                time.sleep(0.02) # Sleep briefly to prevent high CPU usage when paused
                continue

            start_time = time.time()
            ret, frame = self.cap.read()
            if ret:
                current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                try:
                    FRAME_QUEUE.put((current_pos, frame), block=False)
                except queue.Full:
                    time.sleep(0.01) 
                    continue
            else:
                self.is_playing = False
                continue

            elapsed = time.time() - start_time
            delay = (1.0 / self.fps / self.playback_speed) - elapsed
            if delay > 0: time.sleep(delay)

    def _gui_consumer_loop(self):
        """(RUNS ON GUI THREAD) Gets frames from queue and displays them."""
        try:
            frame_num, frame_data = FRAME_QUEUE.get_nowait()
            self._update_display(frame_num, frame_data)
        except queue.Empty:
            pass
        finally:
            self.root.after(30, self._gui_consumer_loop)

    def _update_display(self, frame_num, frame_data):
        """(RUNS ON GUI THREAD) Renders a single frame."""
        self.frame_var.set(frame_num)
        display_frame = frame_data.copy()

        active_bboxes = self._get_active_bboxes_for_frame(frame_num)
        for bbox_poly in active_bboxes:
            cv2.polylines(display_frame, [np.array(bbox_poly, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        if self.is_drawing_bb and self.current_bb_points:
            points_np = np.array(self.current_bb_points, dtype=np.int32)
            for point in self.current_bb_points:
                cv2.circle(display_frame, point, 5, (0, 0, 255), -1)
            
            is_closed = len(self.current_bb_points) == 4
            cv2.polylines(display_frame, [points_np], isClosed=is_closed, color=(255, 200, 0), thickness=2)

        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # --- DYNAMIC RESIZING LOGIC ---
        container_w = self.video_container_frame.winfo_width()
        container_h = self.video_container_frame.winfo_height()
        
        # Default to a tiny image if container isn't ready
        resized_img = Image.new('RGB', (1, 1))

        if container_w > 1 and container_h > 1:
            orig_w, orig_h = self.original_frame_dims
            if orig_w > 0 and orig_h > 0:
                # Calculate the scale to fit the video within the container while maintaining aspect ratio
                scale = min(container_w / orig_w, container_h / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)

                resized_img = img_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        self.photo = ImageTk.PhotoImage(image=resized_img)
        self.video_label.config(image=self.photo)
        self.video_label.image = self.photo
        self._update_frame_label()

    def seek_frame(self, value):
        """(RUNS ON GUI THREAD) Requests a seek."""
        if not self.cap: return
        self.is_playing = False
        self.play_pause_btn.config(text="▶ Play")
        self.command_queue.put(('seek', int(float(value))))

    # --- Mouse and Bounding Box Logic ---
    
    def on_video_click(self, event):
        if not self.is_drawing_bb: return
            
        if len(self.current_bb_points) < 4:
            # --- SIMPLIFIED DYNAMIC COORDINATE CALCULATION ---
            # event.x/y are relative to the label, which shows the resized video.
            label_w = event.widget.winfo_width()
            label_h = event.widget.winfo_height()
            orig_w, orig_h = self.original_frame_dims

            if orig_w == 0 or orig_h == 0 or label_w <= 1 or label_h <= 1: return 

            # Scale from the label's coordinate system to the original video's.
            scale_x = orig_w / label_w
            scale_y = orig_h / label_h

            orig_x = int(event.x * scale_x)
            orig_y = int(event.y * scale_y)
            
            self.current_bb_points.append((orig_x, orig_y))
            self.seek_frame(self.frame_var.get())

    def add_bounding_box(self):
        if len(self.current_bb_points) != 4:
            messagebox.showwarning("Incomplete BBox", "A bounding box requires exactly 4 points.")
            return
        if self.current_clip_id is None:
            messagebox.showerror("Error", "Please save a temporal clip first before adding bounding boxes to it.")
            return

        start_f, end_f = self.bb_start_var.get(), self.bb_end_var.get()
        if not (self.start_frame <= start_f <= end_f <= self.end_frame):
            messagebox.showwarning("Invalid Range", "BBox validity must be within the temporal clip's range.")
            return
            
        if self.current_clip_id not in self.bounding_boxes:
            self.bounding_boxes[self.current_clip_id] = []
        
        x_coords = [p[0] for p in self.current_bb_points]
        y_coords = [p[1] for p in self.current_bb_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        normalized_rect = [
            (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)
        ]

        bbox_count = len(self.bounding_boxes[self.current_clip_id])
        bbox_info = {
            'id': f"bbox_{bbox_count}",
            'start_frame': start_f,
            'end_frame': end_f,
            'keyframes': {
                start_f: normalized_rect,
                end_f: normalized_rect
            }
        }
        self.bounding_boxes[self.current_clip_id].append(bbox_info)
        
        messagebox.showinfo("Success", f"Bounding box added for frames {start_f}-{end_f}.")
        self.clear_bb_points()
        self.update_treeview()

    def _get_active_bboxes_for_frame(self, frame_num):
        polygons = []
        if self.current_clip_id in self.bounding_boxes:
            for bbox_info in self.bounding_boxes[self.current_clip_id]:
                if bbox_info['start_frame'] <= frame_num <= bbox_info['end_frame']:
                    keyframes = sorted(bbox_info['keyframes'].keys())
                    if not keyframes: continue
                    
                    if frame_num <= keyframes[0]:
                        polygons.append(bbox_info['keyframes'][keyframes[0]])
                        continue
                    if frame_num >= keyframes[-1]:
                        polygons.append(bbox_info['keyframes'][keyframes[-1]])
                        continue

                    prev_kf, next_kf = -1, -1
                    for i in range(len(keyframes) - 1):
                        if keyframes[i] <= frame_num <= keyframes[i+1]:
                            prev_kf, next_kf = keyframes[i], keyframes[i+1]
                            break
                    
                    if prev_kf != -1 and prev_kf != next_kf:
                        t = (frame_num - prev_kf) / (next_kf - prev_kf)
                        prev_box = np.array(bbox_info['keyframes'][prev_kf])
                        next_box = np.array(bbox_info['keyframes'][next_kf])
                        interp_box = prev_box + t * (next_box - prev_box)
                        polygons.append(interp_box.astype(int).tolist())
        return polygons

    # --- Annotation Saving and Loading ---
    
    def save_temporal_annotation(self):
        current_file = self.file_names[self.current_video_idx]
        clip_id = f"{current_file}_{self.start_frame}_{self.end_frame}"
        
        # CHANGED: Handle replacing a point annotation
        if self.active_point_id:
            found_index = -1
            for i, ann in enumerate(self.annotations):
                if ann.get('id') == self.active_point_id:
                    found_index = i
                    break
            
            if found_index != -1:
                new_annotation = { 'id': clip_id, 'file_name': current_file, 'start_frame': self.start_frame, 'end_frame': self.end_frame, 'type': 'clip' }
                self.annotations[found_index] = new_annotation
                self.current_clip_id = clip_id
                self.active_point_id = None
                messagebox.showinfo("Updated", f"point annotation updated to clip [{self.start_frame}-{self.end_frame}].")
                self.update_treeview()
                return

        # Fallback to existing logic if not replacing a point
        if any(ann['id'] == clip_id for ann in self.annotations):
            messagebox.showinfo("Info", "This temporal annotation already exists.")
            self.current_clip_id = clip_id
            return

        annotation = { 'id': clip_id, 'file_name': current_file, 'start_frame': self.start_frame, 'end_frame': self.end_frame, 'type': 'clip' }
        self.annotations.append(annotation)
        self.current_clip_id = clip_id
        messagebox.showinfo("Saved", f"Temporal clip [{self.start_frame}-{self.end_frame}] selected. You can now add BBoxes.")
        self.update_treeview()
        
    def export_annotations_to_json(self, show_success_message=True):
        # Filter out 'point' annotations, only save full clips.
        export_data = []
        for ann in self.annotations:
            if ann.get('type') == 'point':
                continue
            clip_id = ann['id']
            export_data.append({
                'id': ann['id'],
                'file_name': ann['file_name'],
                'start_frame': ann['start_frame'],
                'end_frame': ann['end_frame'],
                'bounding_boxes': self.bounding_boxes.get(clip_id, [])
            })
        
        if not export_data:
            messagebox.showwarning("No Annotations", "There are no full clip annotations to save.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Annotations As"
        )
        if not save_path:
            return

        try:
            with open(save_path, 'w') as f:
                json.dump(export_data, f, indent=4)
            if show_success_message:
                messagebox.showinfo("Save Successful", f"Annotations successfully saved to\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not write to file: {e}")

    def load_annotations_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv")])
        if not file_path: return

        try:
            new_annotations_count = 0
            if file_path.lower().endswith('.csv'):
                # --- CSV Loading Logic ---
                annotated_videos = {ann['file_name'] for ann in self.annotations}
                
                with open(file_path, mode='r', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    for row in reader:
                        if not row: continue
                        video_basename = row[0]
                        matching_filename = None
                        for fname in self.file_names:
                            if os.path.splitext(fname)[0] == video_basename:
                                matching_filename = fname
                                break
                        
                        if not matching_filename:
                            print(f"Warning: No matching video file found for CSV entry '{video_basename}'")
                            continue
                        
                        if matching_filename in annotated_videos:
                            print(f"Info: Skipping CSV entry for '{matching_filename}' as it's already annotated.")
                            continue
                        
                        center_frames = [int(f) for f in ast.literal_eval(row[1]) if f]
                        for center in center_frames:
                            point_id = f"{matching_filename}_point_{center}"
                            if not any(a.get('id') == point_id for a in self.annotations):
                                self.annotations.append({
                                    'id': point_id,
                                    'file_name': matching_filename,
                                    'center_frame': center,
                                    'type': 'point'
                                })
                                new_annotations_count += 1
                messagebox.showinfo("Success", f"Loaded {new_annotations_count} new point annotations from CSV file for unannotated videos.")

            elif file_path.lower().endswith('.json'):
                # --- Existing JSON Loading Logic ---
                with open(file_path, 'r') as f:
                    loaded_data = json.load(f)
                
                if isinstance(loaded_data, list):
                    for ann in loaded_data:
                        if not isinstance(ann, dict): continue
                        if 'id' in ann and not any(a['id'] == ann.get('id') for a in self.annotations):
                            self.annotations.append({k: v for k, v in ann.items() if k != 'bounding_boxes'})
                            self.bounding_boxes[ann['id']] = ann.get('bounding_boxes', [])
                            new_annotations_count += 1
                    messagebox.showinfo("Success", f"Loaded {new_annotations_count} annotations from application file.")

                elif isinstance(loaded_data, dict):
                    for video_basename, video_info in loaded_data.items():
                        matching_filename = None
                        for fname in self.file_names:
                            if os.path.splitext(fname)[0] == video_basename:
                                matching_filename = fname
                                break
                        if not matching_filename: continue

                        labels = video_info.get('labels', [])
                        in_anomaly = False
                        start = 0
                        for i, label in enumerate(labels):
                            if label == 1.0 and not in_anomaly:
                                in_anomaly = True; start = i
                            elif label == 0.0 and in_anomaly:
                                in_anomaly = False; end = i - 1
                                clip_id = f"{matching_filename}_{start}_{end}"
                                if not any(ann['id'] == clip_id for ann in self.annotations):
                                    self.annotations.append({'id': clip_id, 'file_name': matching_filename, 'start_frame': start, 'end_frame': end, 'type': 'clip'})
                                    self.bounding_boxes[clip_id] = [] 
                                    new_annotations_count += 1
                        if in_anomaly:
                            end = len(labels) - 1
                            clip_id = f"{matching_filename}_{start}_{end}"
                            if not any(ann['id'] == clip_id for ann in self.annotations):
                                self.annotations.append({'id': clip_id, 'file_name': matching_filename, 'start_frame': start, 'end_frame': end, 'type': 'clip'})
                                self.bounding_boxes[clip_id] = []
                                new_annotations_count += 1
                    messagebox.showinfo("Success", f"Loaded {new_annotations_count} new temporal annotations from ground truth file.")
                else:
                    raise TypeError("Unsupported JSON format.")
            
            self.update_treeview()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or parse annotations file: {str(e)}")


    # --- GUI Update Helpers and Callbacks ---

    def on_annotation_select(self, event):
        """Called when an item in the annotations Treeview is selected."""
        selected_items = self.annotations_tree.selection()
        if not selected_items: return

        item_values = self.annotations_tree.item(selected_items[0], 'values')
        file_name, clip_frames_str = item_values[0], item_values[1]
        
        if '-' in clip_frames_str:
            # It's a full clip annotation
            try:
                start_str, end_str = clip_frames_str.split('-')
                start_f, end_f = int(start_str), int(end_str)
                self._handle_selection(file_name, 'clip', start_f, end_f)
            except (ValueError, IndexError):
                print(f"Could not parse clip frames from Treeview: {item_values}")
        else:
            # It's a point annotation
            try:
                center_f = int(clip_frames_str)
                self._handle_selection(file_name, 'point', center_f)
            except (ValueError, IndexError):
                print(f"Could not parse center frame from Treeview: {item_values}")

    def _handle_selection(self, file_name, selection_type, *args):
        """Unified handler for any selection type."""
        def apply_logic():
            if selection_type == 'clip':
                self._apply_annotation_selection(args[0], args[1], file_name)
            elif selection_type == 'point':
                self._apply_point_selection(args[0], file_name)

        if file_name != self.file_names[self.current_video_idx]:
            try:
                video_idx = self.file_names.index(file_name)
                self._load_video(video_idx)
                self.root.after(100, apply_logic)
            except ValueError:
                messagebox.showerror("Error", f"Could not find the video file '{file_name}' in the loaded directory.")
        else:
            apply_logic()

    def _apply_point_selection(self, center_frame, file_name):
        """Helper to apply a point selection."""
        self.seek_frame(center_frame)
        self.current_clip_id = None
        # CHANGED: Keep track of the selected point ID to replace it later
        self.active_point_id = f"{file_name}_point_{center_frame}"
        self.start_frame_label.config(text="Start: -")
        self.end_frame_label.config(text="End: -")
        print(f"Jumped to point frame: {center_frame}")

    def _apply_annotation_selection(self, start_f, end_f, file_name):
        """Helper function to apply the selected annotation's state to the GUI."""
        self.start_frame = start_f
        self.end_frame = end_f
        self.start_frame_label.config(text=f"Start: {self.start_frame}")
        self.end_frame_label.config(text=f"End: {self.end_frame}")

        self.bb_start_var.set(start_f)
        self.bb_end_var.set(end_f)

        self.current_clip_id = f"{file_name}_{start_f}_{end_f}"
        # CHANGED: Clear the point ID since we've selected a full clip
        self.active_point_id = None
        self.seek_frame(start_f)
        print(f"Selected clip: {self.current_clip_id}")

    def _update_gui_for_new_video(self):
        self.video_info_label.config(text=f"{self.file_names[self.current_video_idx]}")
        self.frame_scale.config(to=self.total_frames - 1 if self.total_frames > 0 else 0)
        self.frame_var.set(0)
        self.start_frame = 0
        self.end_frame = self.total_frames - 1 if self.total_frames > 0 else 0
        self.start_frame_label.config(text=f"Start: {self.start_frame}")
        self.end_frame_label.config(text=f"End: {self.end_frame}")
        self.current_clip_id = None
        self.active_point_id = None # Clear active point on new video
        self.clear_bb_points()

    def update_treeview(self):
        for i in self.annotations_tree.get_children(): self.annotations_tree.delete(i)
        for ann in self.annotations:
            if ann.get('type') == 'point':
                clip_display = ann['center_frame']
                num_boxes = 'N/A'
            else: # It's a 'clip'
                clip_display = f"{ann['start_frame']}-{ann['end_frame']}"
                clip_id = ann['id']
                num_boxes = len(self.bounding_boxes.get(clip_id, []))
            
            self.annotations_tree.insert('', tk.END, values=(ann['file_name'], clip_display, num_boxes))

    def _update_frame_label(self):
        self.current_frame_label.config(text=f"Frame: {self.frame_var.get()} / {self.total_frames - 1}")

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_pause_btn.config(text="❚❚ Pause" if self.is_playing else "▶ Play")

    def prev_video(self):
        if self.current_video_idx > 0: self._load_video(self.current_video_idx - 1)
            
    def next_video(self):
        if self.current_video_idx < len(self.file_names) - 1: self._load_video(self.current_video_idx + 1)

    def update_speed(self, value):
        self.playback_speed = float(value)
        self.speed_label.config(text=f"Speed: {self.playback_speed:.1f}x")

    def set_start_frame(self):
        self.start_frame = self.frame_var.get()
        self.start_frame_label.config(text=f"Start: {self.start_frame}")

    def set_end_frame(self):
        self.end_frame = self.frame_var.get()
        self.end_frame_label.config(text=f"End: {self.end_frame}")

    def set_bbox_start_frame(self):
        """Sets the bounding box start frame to the current frame."""
        self.bb_start_var.set(self.frame_var.get())

    def set_bbox_end_frame(self):
        """Sets the bounding box end frame to the current frame."""
        self.bb_end_var.set(self.frame_var.get())

    def toggle_bb_drawing(self):
        self.is_drawing_bb = self.draw_bb_var.get()
        if self.is_drawing_bb and self.current_clip_id:
            self.bb_start_var.set(self.start_frame)
            self.bb_end_var.set(self.end_frame)
        self.clear_bb_points()

    def clear_bb_points(self):
        self.current_bb_points = []
        self.seek_frame(self.frame_var.get())

    def run(self):
        if self.file_names:
            self._load_video(0)
        self.root.mainloop()

# --- Main Execution ---
if __name__ == "__main__":
    try:
        video_directory = 'videos'
        if not os.path.isdir(video_directory):
            os.makedirs(video_directory)
            messagebox.showinfo("Setup", f"Created directory '{video_directory}'. Please place your videos inside and restart.")
            file_names = []
        else:
            supported = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
            file_names = sorted([f for f in os.listdir(video_directory) if f.lower().endswith(supported)])

        if not file_names:
            if os.path.isdir(video_directory):
                messagebox.showwarning("No Videos", f"No video files found in the '{video_directory}' directory.")
        else:
            app = VideoAnnotator(file_names, video_directory)
            app.run()

    except Exception as e:
        messagebox.showerror("Fatal Error", f"An unexpected error occurred: {e}")
