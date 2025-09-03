# %%
import cv2
import os
import re

# %%
def extract_numbers(s):
    numbers = list(map(int, re.findall(r"-?\d+", s)))
    while len(numbers) < 3:  # Ensure exactly 3 numbers by padding with 0
        numbers.append(0)
    return tuple(numbers)


# %%
# Set parameters
image_folder = '/home/zihangw/EvoComm/figures/evolve_traj'  # Folder containing images
output_video = '/home/zihangw/EvoComm/figures/bn_ndeme10_edge5_beta0_rep0_mac.mp4'
fps = 4  # Frames per second
# seconds_per_frame = 1

# Get images and sort them
# images = []
# generations_list = []
# for img in os.listdir(image_folder):
#     if not (img.endswith(".png") or img.endswith(".jpg")):
#         raise ValueError(f"All files in {image_folder} must be .png or .jpg files.")
#     images.append(img)
#     generations_list.append(img.split("_")[-1].split(".")[0])
images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
images = sorted(images, key=extract_numbers)
generations_list = [img.split("_")[-1].split(".")[0] for img in images]

# %%
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

#%%
font = cv2.FONT_HERSHEY_SIMPLEX  # Font for text
font_scale = 2  # Size of text
font_thickness = 2  # Thickness of text
text_color = (34, 2, 2)
position = (width - 600, 400)  # Position of text (x, y)


# %%
# Read the first image to get dimensions

# Define video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')  # Codec
# fourcc = cv2.VideoWriter_fourcc(*'H264')  # Codec
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
# frames_per_image = fps * seconds_per_frame

# Write images to video
for i, image in enumerate(images):
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)

    text = f"generation {generations_list[i]}"  # Frame number
    frame_with_text = frame.copy()
    cv2.putText(frame_with_text, text, position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # for _ in range(frames_per_image):
    video.write(frame_with_text)


video.release()
cv2.destroyAllWindows()
print("Video created successfully!")

# %%
