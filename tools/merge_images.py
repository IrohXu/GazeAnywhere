import cv2
import os
import glob

# Path to your images and output video file
image_folder = '/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/tmp'
output_video = 'output_video.mp4'

# Grab all .jpg/.png/etc in the folder and sort them
images = []
for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
    images.extend(glob.glob(os.path.join(image_folder, ext)))
images.sort()

if not images:
    raise ValueError(f'No images found in {image_folder}')

# Read the first image to get dimensions
frame = cv2.imread(images[0])
height, width, _ = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'avc1' on some platforms
video_writer = cv2.VideoWriter(output_video, fourcc, 5.0, (width, height))

# Iterate through images and write to video
for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        print(f'Warning: couldn\'t read {img_path}, skipping.')
        continue
    # If any image has different size, resize it
    if (img.shape[1], img.shape[0]) != (width, height):
        img = cv2.resize(img, (width, height))
    video_writer.write(img)

# Release the writer
video_writer.release()
print(f'Done! Video saved as {output_video}')