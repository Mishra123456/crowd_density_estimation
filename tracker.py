from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, maxDisappeared=30, maxDistance=150):
        """
        Initializes the centroid tracker.
        :param maxDisappeared: Maximum consecutive frames an object is allowed to be missing before being deregistered.
        :param maxDistance: Maximum pixel distance between centroids to associate an object between frames.
        """
        self.nextObjectID = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        """
        Updates the tracker with new bounding boxes from the current frame.
        rects: list of bounding boxes [(startX, startY, endX, endY)]
        """
        # If no bounding boxes in this frame
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Calculate centroids for current frame boxes
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If no objects are currently tracked, register all inputs
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute Euclidean distance using numpy broadcast
            D = np.linalg.norm(np.array(objectCentroids)[:, None] - inputCentroids, axis=2)

            # Find matching pairs (smallest distances)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                # If the distance is too great, it's not the same person
                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            # Find unassociated tracked objects and increment disappeared counter
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # Register entirely new objects that couldn't match existing tracks
            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects
