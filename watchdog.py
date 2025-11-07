from typing import List, Dict, Set, Tuple
import time
from dataclasses import dataclass
from enum import Enum
import numpy as np

class AlertLevel(Enum):
    INFO = 1
    WARNING = 2
    ALERT = 3

@dataclass
class SceneState:
    objects: Dict[str, Tuple[float, Tuple[float, float, float, float]]]  # object_name -> (confidence, bbox)
    timestamp: float
    is_empty: bool

class WatchdogMonitor:
    def __init__(self, 
                 cooldown_seconds: float = 5.0,
                 empty_scene_cooldown: float = 30.0,
                 min_confidence: float = 0.5):
        self.cooldown_seconds = cooldown_seconds
        self.empty_scene_cooldown = empty_scene_cooldown
        self.min_confidence = min_confidence
        
        # Track scene state
        self.previous_state: SceneState = None
        self.last_alert_time: Dict[str, float] = {}
        self.last_empty_alert_time: float = 0
        self.important_objects = {
            'laptop', 'cell phone', 'backpack', 'handbag', 'suitcase',
            'person', 'bicycle', 'car', 'motorcycle'
        }
        
        # Track persistent objects
        self.persistent_objects: Set[str] = set()
        self.object_persistence_time: Dict[str, float] = {}
        self.min_persistence_time = 10.0  # seconds to consider an object "persistent"
    
    def _get_current_objects(self, detections: List[Tuple[str, float, Tuple[float, float, float, float]]]) -> Dict[str, Tuple[float, Tuple[float, float, float, float]]]:
        """Convert detections to a dictionary of current objects."""
        return {name: (conf, bbox) for name, conf, bbox in detections if conf >= self.min_confidence}
    
    def _is_significant_change(self, current: Dict[str, Tuple[float, Tuple[float, float, float, float]]], 
                             previous: Dict[str, Tuple[float, Tuple[float, float, float, float]]]) -> bool:
        """Check if the scene change is significant enough to warrant an alert."""
        if not previous:
            return bool(current)  # Any objects in empty scene is significant
        
        # Check for new or removed important objects
        current_objects = set(current.keys())
        previous_objects = set(previous.keys())
        
        new_objects = current_objects - previous_objects
        removed_objects = previous_objects - current_objects
        
        # Check if any new or removed objects are important
        return bool(new_objects & self.important_objects) or bool(removed_objects & self.important_objects)
    
    def _update_persistent_objects(self, current_objects: Dict[str, Tuple[float, Tuple[float, float, float, float]]], 
                                 current_time: float):
        """Update tracking of persistent objects."""
        current_set = set(current_objects.keys())
        
        # Update persistence times
        for obj in current_set:
            if obj not in self.object_persistence_time:
                self.object_persistence_time[obj] = current_time
            elif current_time - self.object_persistence_time[obj] >= self.min_persistence_time:
                self.persistent_objects.add(obj)
        
        # Remove objects that are no longer present
        for obj in list(self.persistent_objects):
            if obj not in current_set:
                self.persistent_objects.remove(obj)
                self.object_persistence_time.pop(obj, None)
    
    def _should_alert_empty_scene(self, current_time: float) -> bool:
        """Check if we should alert about an empty scene."""
        return (current_time - self.last_empty_alert_time) >= self.empty_scene_cooldown
    
    def process_scene(self, detections: List[Tuple[str, float, Tuple[float, float, float, float]]]) -> List[Tuple[AlertLevel, str]]:
        """
        Process the current scene and return any alerts that should be announced.
        Returns a list of (AlertLevel, message) tuples.
        """
        current_time = time.time()
        current_objects = self._get_current_objects(detections)
        alerts = []
        
        # Update persistent objects tracking
        self._update_persistent_objects(current_objects, current_time)
        
        # Handle empty scene
        if not current_objects:
            if self._should_alert_empty_scene(current_time):
                alerts.append((AlertLevel.INFO, "Scene is clear."))
                self.last_empty_alert_time = current_time
            return alerts
        
        # If this is the first scene, just store it
        if self.previous_state is None:
            self.previous_state = SceneState(current_objects, current_time, False)
            # Alert for initial person detection
            if 'person' in current_objects:
                alerts.append((AlertLevel.ALERT, "Person detected in the scene."))
            return alerts
        
        # Check for significant changes
        if self._is_significant_change(current_objects, self.previous_state.objects):
            # Check for new objects
            new_objects = set(current_objects.keys()) - set(self.previous_state.objects.keys())
            for obj in new_objects:
                if obj in self.important_objects:
                    if (current_time - self.last_alert_time.get(obj, 0)) >= self.cooldown_seconds:
                        alerts.append((AlertLevel.ALERT, f"New {obj} detected in the scene."))
                        self.last_alert_time[obj] = current_time
            
            # Check for removed objects
            removed_objects = set(self.previous_state.objects.keys()) - set(current_objects.keys())
            for obj in removed_objects:
                if obj in self.important_objects:
                    if (current_time - self.last_alert_time.get(f"removed_{obj}", 0)) >= self.cooldown_seconds:
                        alerts.append((AlertLevel.WARNING, f"{obj.capitalize()} has been removed from the scene."))
                        self.last_alert_time[f"removed_{obj}"] = current_time
        
        # Immediate alert for person detection
        if 'person' in current_objects and 'person' not in self.previous_state.objects:
            alerts.append((AlertLevel.ALERT, "Person detected in the scene."))
        
        # Update previous state
        self.previous_state = SceneState(current_objects, current_time, False)
        return alerts 