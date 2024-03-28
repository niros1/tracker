import cv2
import numpy as np


def reduce_image_resolution(image, scale_percent=50):
    # Calculate the 50 percent of original dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    # # dsize
    dsize = (width, height)
    # resize image
    output = cv2.resize(image, dsize)

    # Downsample the image
    # downsampled = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # # Upsample the image back to original size
    # output = cv2.resize(
    #     downsampled, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    # )

    return output


image_name = "frame9"
img_path = f"output/frames/{image_name}.jpg"
# zoomed_img_path = f"/home/ubuntu/tracker/tracker/output/frames/{image_name}_zoom.jpg"
# Load an image
org_image = cv2.imread(img_path)
low_res_image = reduce_image_resolution(org_image, 30)
# image_copy = image.copy()

# Initialize the list to store rectangle points

rect_endpoint = []

rect_endpoint_tmp = []


# Define the event function
def draw_rectangle(event, x, y, flags, param):
    print(event)
    global rect_endpoint_tmp, rect_endpoint
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Start point: {x}, {y}")
        rect_endpoint_tmp[:] = [x, y]
        start_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        print(f"End point: {x}, {y}")
        rect_endpoint_tmp += [x, y]
        cv2.rectangle(
            org_image,
            (rect_endpoint_tmp[0], rect_endpoint_tmp[1]),
            (rect_endpoint_tmp[2], rect_endpoint_tmp[3]),
            (255, 255, 255),
            2,
        )
        rect_endpoint += rect_endpoint_tmp
        cv2.imshow("image", org_image)


# Create a window and assign the callback
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_rectangle)
cv2.imshow("image", org_image)
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

# # Print the coordinates of the rectangles
# for rect in rect_endpoint:
#     print(f"Start point: {rect}, end point: {rect}")
print(rect_endpoint)
# x_points = rect_endpoint[::4]
# y_points = rect_endpoint[1::4]
lenght = len(rect_endpoint) / 4
for i in range(int(len(rect_endpoint) / 4)):
    idx = i * 4
    print(
        f"Start point: {rect_endpoint[idx]} {rect_endpoint[idx+1]}, END POINT {rect_endpoint[idx+2]} {rect_endpoint[idx+3]}"
    )
