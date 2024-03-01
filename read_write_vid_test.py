import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Function to add text to a frame
def add_text_to_frame(frame, text, position=(50, 50), font_size=30):
    # Convert the frame to PIL Image
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    # Specify the font (default font here)
    font = ImageFont.load_default(font_size)

    # Add text
    draw.text(position, text, font=font, fill=(0, 0, 255))

    # Convert back to OpenCV image and return
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def add_text_to_frame2(frame, text, position=(50, 50), font_scale=1, font_color=(0, 0, 255), thickness=4):
    """
    Adds text to a single frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness)
    return frame
# Open the input video
cap = cv2.VideoCapture('input/basketball.mp4')

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'X264')
print(fps, width, height, fourcc)

# Create a VideoWriter object
out = cv2.VideoWriter('./_test_output.mp4', fourcc, fps, (width, height))

print("OK")

while cap.isOpened():
    print(".",end="", flush=True)
    ret, frame = cap.read()
    if not ret:
        break

    # Add text to frame
    # frame = add_text_to_frame2(frame, "Your Text Here")

    # Write the frame to the output video
    out.write(frame)

# Release everything
cap.release()
print("cap Done")

out.release()
print("out Done")
