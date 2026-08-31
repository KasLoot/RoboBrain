import cv2

# -------------------------
# Configuration
# -------------------------
video_path = "video/failure_part.mp4"
frame_number = 903  # 0-based frame index

# Crop coordinates: (x, y)
top_left = (350, 50)
bottom_right = (650, 300)


# -------------------------
# Open video
# -------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Total frames: {total_frames}")
print(f"Selected frame: {frame_number}")

if frame_number < 0 or frame_number >= total_frames:
    cap.release()
    raise ValueError(
        f"frame_number must be between 0 and {total_frames - 1}"
    )


# -------------------------
# Seek to designated frame
# -------------------------
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

success, frame = cap.read()

cap.release()

if not success:
    raise RuntimeError(f"Could not read frame {frame_number}")


# -------------------------
# Preview original frame
# -------------------------
cv2.imshow(f"Frame {frame_number}", frame)

print("Press any key to continue...")
cv2.waitKey(0)
cv2.destroyAllWindows()


# -------------------------
# Crop selected frame
# -------------------------
x1, y1 = top_left
x2, y2 = bottom_right

cropped = frame[y1:y2, x1:x2]


# -------------------------
# Preview cropped image
# -------------------------
cv2.imshow("Cropped frame", cropped)

print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()


# Optional: save crop
cv2.imwrite("FAIL_DEMO_2.jpg", cropped)