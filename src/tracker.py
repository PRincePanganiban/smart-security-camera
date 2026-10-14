"""
tracker.py
----------
TOPIC E - MOTION AND OBJECT TRACKING

A simple centroid tracker. Every moving box gets a center point (centroid).
Centroids in the new frame are matched to the closest centroids from the
previous frame, so the same object keeps the same ID while it moves.
The last positions of each object are stored to draw a movement trail.
"""

from collections import deque
import math


class CentroidTracker:
    def __init__(self, max_distance=90, max_missing=15, trail_length=30):
        """
        max_distance : farthest an object may move between frames and still
                       be considered the same object (pixels)
        max_missing  : frames an object may disappear before it is forgotten
        trail_length : number of past positions kept for the movement trail
        """
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.trail_length = trail_length
        self.next_id = 0
        self.centroids = {}   # id -> (cx, cy)
        self.missing = {}     # id -> frames since last seen
        self.trails = {}      # id -> deque of recent centroids

    @staticmethod
    def _centroid(box):
        x, y, w, h = box
        return (x + w // 2, y + h // 2)

    def _register(self, centroid):
        self.centroids[self.next_id] = centroid
        self.missing[self.next_id] = 0
        self.trails[self.next_id] = deque([centroid], maxlen=self.trail_length)
        self.next_id += 1
        return self.next_id - 1

    def _forget(self, object_id):
        del self.centroids[object_id]
        del self.missing[object_id]
        del self.trails[object_id]

    def update(self, boxes):
        """
        Match the new boxes with known objects.
        Returns a list of (object_id, box) for the objects seen this frame.
        """
        new_centroids = [self._centroid(b) for b in boxes]
        assigned = {}  # box index -> object id

        # Build all (distance, object, box) pairs and match the closest first
        pairs = []
        for oid, old_c in self.centroids.items():
            for i, new_c in enumerate(new_centroids):
                pairs.append((math.dist(old_c, new_c), oid, i))
        pairs.sort()

        used_objects, used_boxes = set(), set()
        for dist, oid, i in pairs:
            if dist > self.max_distance:
                break
            if oid in used_objects or i in used_boxes:
                continue
            assigned[i] = oid
            used_objects.add(oid)
            used_boxes.add(i)
            self.centroids[oid] = new_centroids[i]
            self.missing[oid] = 0
            self.trails[oid].append(new_centroids[i])

        # Objects not matched: count a missed frame, forget them if gone too long
        for oid in list(self.centroids):
            if oid not in used_objects:
                self.missing[oid] += 1
                if self.missing[oid] > self.max_missing:
                    self._forget(oid)

        # Boxes not matched are new objects
        for i, c in enumerate(new_centroids):
            if i not in used_boxes:
                assigned[i] = self._register(c)

        return [(assigned[i], boxes[i]) for i in range(len(boxes))]
