import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
import os
import gc
from datetime import datetime, timedelta
import psutil  

# code patch kurnia
import requests

# Jetson Nano dashboard base URL; can be overridden via env var
FISHDASH_SERVER = os.environ.get("FISHDASH_SERVER", "http://100.76.99.48:8000")

def _send_event(direction: str, conf: float, track_id=None, cls: str="fish"):
    """Send an event to the FishDash server (/event) — best-effort, non-blocking."""
    try:
        r = requests.post(
            f"{FISHDASH_SERVER}/event",
            json={
                "direction": direction,
                "conf": float(conf) if conf is not None else None,
                "track_id": int(track_id) if track_id is not None else None,
                "cls": cls,
            },
            timeout=1.5
        )
        r.raise_for_status()
        print(f"[INFO] Event sent to {FISHDASH_SERVER}: {direction} id={track_id} conf={conf:.2f}")
    except Exception as e:
        print(f"[WARN] Failed to send event to {FISHDASH_SERVER}: {e}")
# code patch kurnia end

class SimpleFishCounter:
    def __init__(self, model_path="best.pt"):
        self.model = YOLO(model_path)
        self.line_points = None
        self.line_type = "vertical"
        self.drawing = False
        self.temp_point = None
        self.config_file = "live_camera_line_config.json"
        
        # Counting variables
        self.in_count = 0
        self.out_count = 0
        self.track_history = {}
        self.counted_ids = set()
        
        # Memory management and monitoring for long time runs
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 600  # Clean up every 10 minutes
        self.max_inactive_time = 60  # Remove tracks inactive for 1 min
        self.track_last_seen = {}  # Track when each ID was last seen
        self.start_time = time.time()
        
        # Status reporting
        self.last_status_time = time.time()
        self.status_interval = 600  # Status report every 10 minutes
        self.frame_count = 0
        
        # Performance monitoring
        self.fps_history = []
        self.memory_usage_history = []
        
    def get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  
        
    def cleanup_memory(self):
        """Clean up old tracks and perform garbage collection"""
        current_time = time.time()
        
        print(f"\n[CLEANUP] Starting memory cleanup at {datetime.now().strftime('%H:%M:%S')}")
        
        # Get memory usage before cleanup
        memory_before = self.get_memory_usage()
        
        # Clean up old track histories
        inactive_tracks = []
        for track_id in list(self.track_history.keys()):
            if track_id not in self.track_last_seen:
                inactive_tracks.append(track_id)
                continue
                
            time_since_seen = current_time - self.track_last_seen[track_id]
            if time_since_seen > self.max_inactive_time:
                inactive_tracks.append(track_id)
        
        # Remove inactive tracks
        for track_id in inactive_tracks:
            if track_id in self.track_history:
                del self.track_history[track_id]
            if track_id in self.track_last_seen:
                del self.track_last_seen[track_id]
            # Keep counted_ids to prevent double counting
        
        # Limit the size of counted_ids if it gets too large
        if len(self.counted_ids) > 10000:  
            # Keep only the most recent half
            self.counted_ids = set(list(self.counted_ids)[-5000:])
            print(f"[CLEANUP] Trimmed counted_ids set to prevent large memory usage.")
        
        # Force garbage collection
        gc.collect()
        
        # Get memory usage after cleanup
        memory_after = self.get_memory_usage()
        memory_freed = memory_before - memory_after
        
        print(f"[CLEANUP] Removed {len(inactive_tracks)} inactive tracks")
        print(f"[CLEANUP] Active tracks: {len(self.track_history)}")
        print(f"[CLEANUP] Memory: {memory_before:.1f}MB -> {memory_after:.1f}MB (freed: {memory_freed:.1f}MB)")
        
        self.memory_usage_history.append(memory_after)
        # Keep only last 100 memory readings
        if len(self.memory_usage_history) > 100:
            self.memory_usage_history = self.memory_usage_history[-100:]
    
    def print_detailed_status(self):
        """Print detailed status report"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        runtime_str = str(timedelta(seconds=int(elapsed)))
        
        # Calculate current FPS
        current_fps = self.frame_count / elapsed if elapsed > 0 else 0
        self.fps_history.append(current_fps)
        
        # Keep only last 100 FPS readings for average
        if len(self.fps_history) > 100:
            self.fps_history = self.fps_history[-100:]
        
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        # Memory usage
        current_memory = self.get_memory_usage()
        avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history) if self.memory_usage_history else current_memory
        
        print("\n" + "-"*50)
        print(f"STATUS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*50)
        print(f"Runtime: {runtime_str}")
        print(f"Total Frames: {self.frame_count:,}")
        print(f"Current FPS: {current_fps:.1f} | Average FPS: {avg_fps:.1f}")
        print(f"Fish Count - IN: {self.in_count} | OUT: {self.out_count} | Net: {self.in_count - self.out_count}")
        print(f"Active Tracks: {len(self.track_history)} | Total Counted IDs: {len(self.counted_ids)}")
        print(f"Memory Usage: {current_memory:.1f}MB")
        print("-"*50)
        
    def load_saved_line_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.line_points = config.get('line_points')
                    self.line_type = config.get('line_type', 'vertical')
                    print("Loaded saved line configuration.")
                    return True
            except Exception as e:
                print(f"Could not load config file: {e}")
        return False
    
    def save_line_config(self):
        if self.line_points:
            config = {
                'line_points': self.line_points,
                'line_type': self.line_type,
                'source': 'live_camera',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            try:
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"Line configuration saved to: {self.config_file}")
            except Exception as e:
                print(f"Could not save configuration: {e}")
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.temp_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if self.temp_point and abs(x - self.temp_point[0]) + abs(y - self.temp_point[1]) > 10:
                self.line_points = [self.temp_point, (x, y)]
                dx = abs(x - self.temp_point[0])
                dy = abs(y - self.temp_point[1])
                if dx > dy * 2: 
                    self.line_type = "horizontal"
                elif dy > dx * 2: 
                    self.line_type = "vertical"
                else: 
                    self.line_type = "diagonal"
                print(f"Line set: {self.line_type} from {self.temp_point} to {(x, y)}")

    def setup_counting_line_from_frame(self, frame):
        print("\n" + "="*60)
        print("INTERACTIVE LINE POSITIONING FOR LIVE CAMERA")
        print("="*60)

        if self.load_saved_line_config():
            use_saved = input("Use saved line position? (y/n): ").lower().strip()
            if use_saved != 'n':
                return self.line_points

        window_name = "Draw Counting Line (Press 's' to Save, 'r' to Reset, 'q' to Quit)"
        cv2.imshow(window_name, frame)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\nInstructions:")
        print("  - Click and drag to draw the counting line.")
        print("  - Press 's' to save and continue.")
        print("  - Press 'r' to reset the line.")
        print("  - Press 'q' to quit.")
        
        self.line_points = None
        
        while True:
            display_frame = frame.copy()
            if self.line_points:
                cv2.line(display_frame, self.line_points[0], self.line_points[1], (0, 255, 0), 3)
            
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                raise Exception("Line positioning cancelled.")
            elif key == ord('r'):
                self.line_points = None
                print("Line reset. Draw a new line.")
            elif key == ord('s') and self.line_points:
                cv2.destroyAllWindows()
                self.save_line_config()
                break
            elif key == ord('s') and not self.line_points:
                print("Please draw a line first before saving.")
        
        return self.line_points

    def line_intersection(self, p1, p2):
        if not self.line_points:
            return False
        
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            return 1 if val > 0 else 2
        
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        line_p1, line_p2 = self.line_points
        
        o1 = orientation(p1, p2, line_p1)
        o2 = orientation(p1, p2, line_p2)
        o3 = orientation(line_p1, line_p2, p1)
        o4 = orientation(line_p1, line_p2, p2)
        
        if o1 != o2 and o3 != o4:
            return True
        
        if o1 == 0 and on_segment(p1, line_p1, p2): return True
        if o2 == 0 and on_segment(p1, line_p2, p2): return True
        if o3 == 0 and on_segment(line_p1, p1, line_p2): return True
        if o4 == 0 and on_segment(line_p1, p2, line_p2): return True
        
        return False

    def get_direction(self, p1, p2):
        if not self.line_points:
            return None
            
        line_start, line_end = self.line_points
        line_vec = (line_end[0] - line_start[0], line_end[1] - line_start[1])
        move_vec = (p2[0] - p1[0], p2[1] - p1[1])
        
        cross_product = line_vec[0] * move_vec[1] - line_vec[1] * move_vec[0]
        
        if abs(cross_product) < 1:  
            return None
            
        return "in" if cross_product > 0 else "out"

    def process_frame(self, frame):
        current_time = time.time()
        self.frame_count += 1
        
        # Periodic cleanup
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self.cleanup_memory()
            self.last_cleanup_time = current_time
        
        # Periodic status reporting
        if current_time - self.last_status_time > self.status_interval:
            self.print_detailed_status()
            self.last_status_time = current_time
        
        # Process detections
        results = self.model.track(frame, persist=True, conf=0.1, iou=0.5, classes=[0], tracker="bytetrack.yaml")
        
        annotated_frame = results[0].plot()
        
        total_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        tracked_objects = len(results[0].boxes.id) if results[0].boxes is not None and results[0].boxes.id is not None else 0
        
        # Draw counting line
        if self.line_points:
            cv2.line(annotated_frame, tuple(self.line_points[0]), tuple(self.line_points[1]), (0, 255, 255), 3)
            
            mid_point = ((self.line_points[0][0] + self.line_points[1][0]) // 2, 
                        (self.line_points[0][1] + self.line_points[1][1]) // 2)
            
            # Show detection info
            cv2.putText(annotated_frame, f"Detections: {total_detections}, Tracked: {tracked_objects}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Process tracked objects
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confidences)):
                x_center, y_center, width, height = box
                current_pos = (int(x_center), int(y_center))
                
                # Update last seen time for this track
                self.track_last_seen[track_id] = current_time
                
                # Display track info
                cv2.putText(annotated_frame, f"ID:{track_id} ({conf:.2f})", 
                           (current_pos[0]-30, current_pos[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                
                # Initialize or update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append(current_pos)
                
                # Limit track history length
                if len(self.track_history[track_id]) > 25:
                    self.track_history[track_id] = self.track_history[track_id][-25:]
                
                # Draw track path
                if len(self.track_history[track_id]) > 1:
                    points = np.array(self.track_history[track_id], dtype=np.int32)
                    cv2.polylines(annotated_frame, [points], False, (0, 255, 0), 2)
                
                # Check for line crossings
                if len(self.track_history[track_id]) >= 3 and track_id not in self.counted_ids:
                    for j in range(len(self.track_history[track_id]) - 1):
                        prev_pos = self.track_history[track_id][j]
                        curr_pos = self.track_history[track_id][j + 1]
                        
                        if self.line_intersection(prev_pos, curr_pos):
                            direction = self.get_direction(prev_pos, curr_pos)
                            
                            if direction == "in":
                                self.in_count += 1
                                print(f" Fish {track_id} crossed IN. Total IN: {self.in_count}")
                                # code patch kurnia
                                _send_event('in', conf, track_id, 'fish')
                                # code patch kurnia end
                            elif direction == "out":
                                self.out_count += 1
                                print(f" Fish {track_id} crossed OUT. Total OUT: {self.out_count}")
                                # code patch kurnia
                                _send_event('out', conf, track_id, 'fish')
                                # code patch kurnia end
                            
                            if direction:  # Only add to counted if direction was determined
                                self.counted_ids.add(track_id)
                                
                                # Visual feedback for crossing
                                crossing_point = ((prev_pos[0] + curr_pos[0]) // 2, 
                                                (prev_pos[1] + curr_pos[1]) // 2)
                                cv2.circle(annotated_frame, crossing_point, 10, (0, 0, 255), -1)
                                cv2.putText(annotated_frame, direction.upper(), 
                                          (crossing_point[0] + 15, crossing_point[1]), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                break
        
        return annotated_frame

    def draw_counter_overlay(self, frame):
        """Enhanced overlay with runtime and memory info"""
        h, w = frame.shape[:2]
        
        # Calculate runtime
        runtime = time.time() - self.start_time
        runtime_str = str(timedelta(seconds=int(runtime)))
        
        # Main counter (top right)
        counter_text = f"IN: {self.in_count} | OUT: {self.out_count} | Net: {self.in_count - self.out_count}"
        
        # System info 
        active_tracks = len(self.track_history)
        memory_mb = self.get_memory_usage()
        current_fps = self.frame_count / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 0
        
        info_text = f"Runtime: {runtime_str} | FPS: {current_fps:.1f} | Tracks: {active_tracks} | Mem: {memory_mb:.0f}MB"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw main counter background and text
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(counter_text, font, font_scale, thickness)
        padding = 10
        box_width = text_width + 2 * padding
        box_height = text_height + 2 * padding
        box_x = w - box_width - 10
        box_y = 10
        
        # Counter box
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), 2)
        cv2.putText(frame, counter_text, (box_x + padding, box_y + padding + text_height), 
                   font, font_scale, (0, 0, 0), thickness)
        
        # System info 
        cv2.putText(frame, info_text, (10, h - 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, info_text, (10, h - 20), font, 0.5, (255, 255, 255), 1)
        
        return frame


def run_simple_fish_counter(model_path="best.pt", camera_index=1, save_output=False, max_runtime_hours=None):
    """
    Enhanced fish counter with memory management and detailed monitoring

    Args:
        model_path: Path to YOLO model
        camera_index: Camera index (0 for default)
        save_output: Whether to save video output
        max_runtime_hours: Maximum runtime in hours (current: unlimited)
    """
    print("-"*50)
    print("ENHANCED FISH COUNTER WITH MEMORY MANAGEMENT")
    print("-"*50)

    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera with index {camera_index}")

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    print(f"Camera initialized: {w}x{h} @ {fps}fps")

    # Capture setup frame
    success, setup_frame = cap.read()
    if not success:
        raise IOError("Failed to capture setup frame")

    # Initialize counter
    counter = SimpleFishCounter(model_path)
    print(f"Initial memory usage: {counter.get_memory_usage():.1f}MB")

    # Setup counting line
    region_points = counter.setup_counting_line_from_frame(setup_frame)
    if not region_points:
        print("No counting line configured. Exiting.")
        return

    # Setup video writer
    video_writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_fps = max(1, fps if fps > 0 else 30)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"fish_output_{timestamp}.mp4"
        video_writer = cv2.VideoWriter(output_filename, fourcc, output_fps, (w, h))
        print(f"Recording enabled: {output_filename} @ {output_fps}fps")

    print(f"\nStarting live counting...")
    if max_runtime_hours:
        print(f"Maximum runtime: {max_runtime_hours} hours")
    print("Press 'q' to quit, 's' for immediate status report")
    print("-"*50)

    start_time = time.time()

    try:
        while cap.isOpened():
            current_time = time.time()

            # Check maximum runtime
            if max_runtime_hours and (current_time - start_time) > (max_runtime_hours * 3600):
                print(f"\nReached maximum runtime of {max_runtime_hours} hours")
                break

            success, frame = cap.read()
            if not success:
                print("Frame capture failed")
                break

            # Process frame
            annotated_frame = counter.process_frame(frame)
            final_frame = counter.draw_counter_overlay(annotated_frame)

            # Save video
            if video_writer:
                video_writer.write(final_frame)

            # Display
            cv2.imshow("Fish Counter", final_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Manual status report
                counter.print_detailed_status()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        # Final status report
        elapsed = time.time() - start_time
        avg_fps = counter.frame_count / elapsed if elapsed > 0 else 0
        final_memory = counter.get_memory_usage()

        print("\n" + "-"*50)
        print("FINAL RESULTS")
        print("-"*50)
        print(f"Total Runtime: {str(timedelta(seconds=int(elapsed)))}")
        print(f"Total Frames: {counter.frame_count:,}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Fish Count - IN: {counter.in_count} | OUT: {counter.out_count}")
        print(f"Net Fish Count: {counter.in_count - counter.out_count}")
        print(f"Final Memory Usage: {final_memory:.1f}MB")
        print(f"Active Tracks at End: {len(counter.track_history)}")
        print(f"Total Unique Fish IDs: {len(counter.counted_ids)}")
        print("-"*50)


if __name__ == "__main__":
    try:
        run_simple_fish_counter(
            model_path="best.pt", 
            camera_index=1,
            save_output=True,
            max_runtime_hours=None # unlimited runtime
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
