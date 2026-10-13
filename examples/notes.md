# Annotation project memory
A manually corrected bounding box invalidates background tracking jobs that started from an older annotation revision. Match results using video ID, frame number, object ID and revision before applying them.

# Dataset quality
Identical class IDs and bounding box coordinates on one frame indicate duplicate annotations. Keep an audit log when removing duplicates.
