from typing import Dict, List, Tuple, Set
import time
import numpy as np

class DetectionManager:
    def __init__(self, cooldown_seconds: float = 10.0):
        """
        Manages detection cooldowns and prevents repetitive announcements.
        
        Args:
            cooldown_seconds (float): Minimum time between announcements of the same object
        """
        self.cooldown_seconds = cooldown_seconds
        self.last_detection_time: Dict[str, float] = {}
        self.last_announcement: str = ""
        self.last_announcement_time = 0
        self.last_announcement_objects: Set[str] = set()
        self.similarity_threshold = 0.7  # Threshold for considering scenes similar
        
        # Common large objects that shouldn't be classified as "below"
        self.large_objects = {
            'refrigerator', 'oven', 'microwave', 'tv', 'monitor',
            'bed', 'couch', 'chair', 'table', 'desk', 'bookshelf',
            'cabinet', 'shelf', 'counter', 'bench', 'sofa'
        }
    
    def should_announce(self, object_name: str) -> bool:
        """
        Check if an object should be announced based on cooldown.
        
        Args:
            object_name (str): Name of the detected object
            
        Returns:
            bool: True if the object should be announced, False if it's in cooldown
        """
        current_time = time.time()
        
        if object_name not in self.last_detection_time:
            self.last_detection_time[object_name] = current_time
            return True
            
        time_since_last = current_time - self.last_detection_time[object_name]
        
        if time_since_last >= self.cooldown_seconds:
            self.last_detection_time[object_name] = current_time
            return True
            
        return False
    
    def _get_vertical_position(self, bbox: Tuple[float, float, float, float], obj_name: str) -> str:
        """Determine if an object is above, in front of, or below the user."""
        _, y1, _, y2 = bbox
        center_y = (y1 + y2) / 2
        
        # Large objects are always "in front of" unless they're clearly above
        if obj_name in self.large_objects:
            return "in front of" if center_y > 0.3 else "above"
        
        if center_y < 0.4:
            return "above"
        elif center_y > 0.7:
            return "below"
        else:
            return "in front of"
    
    def _get_horizontal_position(self, bbox: Tuple[float, float, float, float]) -> str:
        """Determine if an object is to the left, center, or right."""
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2
        
        if center_x < 0.4:
            return "to your left"
        elif center_x > 0.6:
            return "to your right"
        else:
            return "in front of"
    
    def _group_objects_by_position(self, detections: List[Tuple[str, float, Tuple[float, float, float, float]]]) -> Dict[str, List[str]]:
        """Group objects by their vertical and horizontal positions."""
        groups = {
            "above": [],
            "in_front": [],
            "below": [],
            "left": [],
            "right": []
        }
        
        for name, _, bbox in detections:
            vert_pos = self._get_vertical_position(bbox, name)
            horiz_pos = self._get_horizontal_position(bbox)
            
            if vert_pos == "above":
                groups["above"].append(name)
            elif vert_pos == "below":
                groups["below"].append(name)
            else:  # in front of
                if horiz_pos == "to your left":
                    groups["left"].append(name)
                elif horiz_pos == "to your right":
                    groups["right"].append(name)
                else:
                    groups["in_front"].append(name)
        
        return groups
    
    def _is_scene_similar(self, current_objects: Set[str], previous_objects: Set[str]) -> bool:
        """Check if the current scene is similar to the previous one."""
        if not previous_objects:
            return False
        
        intersection = len(current_objects & previous_objects)
        union = len(current_objects | previous_objects)
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= self.similarity_threshold
    
    def get_spatial_description(self, detections: List[Tuple[str, float, Tuple[float, float, float, float]]]) -> str:
        """
        Generate a natural language description of the scene with spatial relationships.
        
        Args:
            detections: List of (object_name, confidence, bbox) tuples
                      bbox is (x1, y1, x2, y2) normalized coordinates
            
        Returns:
            str: Natural language description of the scene
        """
        current_time = time.time()
        current_objects = {name for name, _, _ in detections}
        
        # Check if we should announce based on time and scene similarity
        if (current_time - self.last_announcement_time < self.cooldown_seconds or
            self._is_scene_similar(current_objects, self.last_announcement_objects)):
            return ""
        
        if not detections:
            return "The scene is empty."
        
        groups = self._group_objects_by_position(detections)
        parts = []
        
        # Describe objects above
        if groups["above"]:
            objects_str = ", ".join(groups["above"])
            parts.append(f"{objects_str} above")
        
        # Describe objects in front
        if groups["in_front"]:
            objects_str = ", ".join(groups["in_front"])
            parts.append(f"{objects_str} in front")
        
        # Describe objects to the left
        if groups["left"]:
            objects_str = ", ".join(groups["left"])
            parts.append(f"{objects_str} around the left")
        
        # Describe objects to the right
        if groups["right"]:
            objects_str = ", ".join(groups["right"])
            parts.append(f"{objects_str} around the right")
        
        # Update last announcement
        self.last_announcement_time = current_time
        self.last_announcement_objects = current_objects
        
        return "I see " + ", ".join(parts) if parts else ""
    
    def reset(self):
        """Reset all cooldown timers."""
        self.last_detection_time.clear()
        self.last_announcement = "" 