import json
import os
import queue
import threading
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class Recorder:
    """
    Recorder collects per-step metadata and saves selected frames without
    blocking the main environment loop by delegating image writes to a worker.
    """

    def __init__(self, recordings_root: str, frame_skip: int = 1, quality: str = "high") -> None:
        self.recordings_root = recordings_root
        self.frame_skip = max(1, int(frame_skip))
        self.quality = quality

        self.steps: List[Dict[str, Any]] = []
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=64)
        self.worker_thread: Optional[threading.Thread] = None

        self.frame_count = 0
        self.saved_count = 0
        self.start_time: Optional[float] = None

        self.recording_dir: Optional[str] = None
        self.frames_dir: Optional[str] = None
        self.meta: Dict[str, Any] = {}

        self._running = False
        self._sentinel = object()
        self._lock = threading.Lock()

    def _mask_hud(self, frame):
        hud_h = 32
        color = (104, 136, 252)
        out = frame.copy()
        out[:hud_h, :, :] = np.array(color, dtype=frame.dtype)
        return out

    def start_recording(self, meta: Dict[str, Any]) -> None:
        timestamp = datetime.now().strftime("recording_%Y%m%d_%H%M%S")
        base_dir = os.path.join(self.recordings_root, timestamp)
        recording_dir = base_dir
        suffix = 1
        while os.path.exists(recording_dir):
            recording_dir = f"{base_dir}_{suffix}"
            suffix += 1

        frames_dir = os.path.join(recording_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        self.recording_dir = recording_dir
        self.frames_dir = frames_dir
        self.meta = dict(meta)

        self.steps = []
        self.frame_count = 0
        self.saved_count = 0
        self.start_time = time.time()

        # Reset queue and worker state
        self.queue = queue.Queue(maxsize=64)
        self._running = True

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def record_step(
        self,
        *,
        frame: Optional[np.ndarray],
        action_idx: int,
        action_buttons: List[str],
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]],
        timestep: int,
    ) -> None:
        if self.start_time is None:
            raise RuntimeError("Recording has not been started. Call start_recording first.")
        self.frame_count += 1
        timestamp = time.time() - self.start_time

        should_save = bool(self.frame_count % self.frame_skip == 0 and frame is not None)

        # External schema mappings
        frame_id = int(self.frame_count - 1)  # 0-based, matches external
        ext_names = self._map_to_external_names(list(action_buttons))
        action_binary = self._action_binary_external(ext_names)

        # Detect death (terminal without flag_get)
        death = bool(done) and not bool(info and info.get("flag_get", False))

        step_info: Dict[str, Any] = {
            "frame_id": frame_id,
            "timestamp": float(timestamp),
            "action_code": int(action_idx),
            "action_binary": action_binary,
            "action_names": ext_names if ext_names else ["NONE"],
            "mario_state": "standing",
            "mario_dead": bool(death),
            "frame_saved": bool(should_save),
            "frame_filename": None,
        }

        if should_save and frame is not None and self.frames_dir:
            user = self.meta.get("user_name", "Zy")
            nt_val = 0 if death else int(self.frame_skip)
            filename = f"{user}_f{frame_id}_a{action_idx}_nt{nt_val}.png"
            out_path = os.path.join(self.frames_dir, filename)
            try:
                self.queue.put((frame.copy(), out_path, step_info), timeout=0.01)
            except queue.Full:
                step_info["frame_saved"] = False

        self.steps.append(step_info)

    def stop_recording(self) -> None:
        if not self._running:
            return

        self._running = False
        # Send sentinel so the worker can exit after flushing tasks.
        while True:
            try:
                self.queue.put(self._sentinel, timeout=0.1)
                break
            except queue.Full:
                continue
        self.queue.join()
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        self.worker_thread = None

        total_steps = len(self.steps)
        duration = time.time() - self.start_time if self.start_time is not None else 0.0
        recording_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        game_version = f"Mario Level {self.meta.get('world')}"
        naming_format = self.meta.get("naming_format", "user_fxxx_axxx_ntxxx.png")
        user_name = self.meta.get("user_name", "Zy")

        recording_data = {
            "recording_info": {
                "total_frames": total_steps,
                "duration": float(duration),
                "recording_time": recording_time,
                "game_version": game_version,
                "user_name": user_name,
                "naming_format": naming_format,
                "temperature": float(self.meta.get("temperature", 0.0)),
            },
            "frame_data": self.steps,
        }

        if self.recording_dir:
            data_path = os.path.join(self.recording_dir, "recording_data.json")
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(recording_data, f, indent=2)

            stats = self._compute_statistics()
            stats_path = os.path.join(self.recording_dir, "statistics.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is self._sentinel:
                self.queue.task_done()
                break

            frame, out_path, step_info = item
            try:
                frame_to_save = self._process_frame_quality(self._mask_hud(frame))
                frame_bgr = cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(out_path, frame_bgr)
                if success:
                    step_info["frame_filename"] = os.path.basename(out_path)
                    step_info["frame_saved"] = True
                    with self._lock:
                        self.saved_count += 1
            except Exception:
                step_info["frame_saved"] = False
            finally:
                self.queue.task_done()

    def _process_frame_quality(self, frame: np.ndarray) -> np.ndarray:
        if self.quality == "low":
            scale = 0.5
        elif self.quality == "medium":
            scale = 0.75
        else:
            scale = 1.0

        if scale == 1.0:
            return frame

        height, width = frame.shape[:2]
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _compute_statistics(self) -> Dict[str, Any]:
        # Keep a lightweight stats file aligned with action_code counts
        action_counts = Counter()
        for step in self.steps:
            action_counts[step["action_code"]] += 1

        action_descriptions = self._load_action_descriptions()

        return {
            "action_counts": {str(idx): count for idx, count in sorted(action_counts.items())},
            "action_descriptions": action_descriptions,
            "saved_frames": self.saved_count,
        }

    def _load_action_descriptions(self) -> Dict[str, List[str]]:
        overrides = self.meta.get("actions_override")
        if isinstance(overrides, list) and overrides:
            return {str(idx): buttons for idx, buttons in enumerate(overrides)}
        actions_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Actions.json"))
        if not os.path.exists(actions_path):
            return {}
        try:
            with open(actions_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

        action_list = config.get("actions", [])
        return {str(idx): buttons for idx, buttons in enumerate(action_list)}

    # --- External schema helpers ---
    def _map_to_external_names(self, buttons: List[str]) -> List[str]:
        # Map our Actions.json button labels to external names
        mapping = {
            "A": "jump",
            "B": "action",
            "left": "left",
            "right": "right",
            "down": "down",
            "NOOP": "NONE",
            "NONE": "NONE",
        }
        mapped = []
        for b in buttons:
            mapped.append(mapping.get(b, b))
        # If action was NOOP only, external expects ["NONE"].
        if not mapped:
            mapped = ["NONE"]
        return mapped

    def _action_binary_external(self, action_names: List[str]) -> str:
        order = self.meta.get("buttons_order_ext") or ["action", "jump", "left", "right", "down"]
        index = {name: i for i, name in enumerate(order)}
        mask = 0
        for name in action_names:
            i = index.get(name)
            if i is not None and name != "NONE":
                mask |= (1 << i)
        return f"0b{mask:b}"
