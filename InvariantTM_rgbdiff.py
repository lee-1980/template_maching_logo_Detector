import os
import sys
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from tqdm import tqdm

box_points = []
button_down = False
calculator = 0

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def scale_image(image, percent, maxwh):
    max_width = maxwh[1]
    max_height = maxwh[0]
    max_percent_width = max_width / image.shape[1] * 100
    max_percent_height = max_height / image.shape[0] * 100
    max_percent = 0
    if max_percent_width < max_percent_height:
        max_percent = max_percent_width
    else:
        max_percent = max_percent_height
    if percent > max_percent:
        percent = max_percent
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    result = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return result, percent

def click_and_crop(event, x, y, flags, param):
    global box_points, button_down, calculator
    if (button_down == False) and (event == cv2.EVENT_LBUTTONDOWN):
        button_down = True
        box_points = [(x, y)]
    elif (button_down == True) and (event == cv2.EVENT_MOUSEMOVE):
        image_copy = param.copy()
        point = (x, y)
        cv2.rectangle(image_copy, box_points[0], point, (0, 255, 0), 2)
        cv2.imshow("Template Cropper - Press C to Crop", image_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        button_down = False
        box_points.append((x, y))
        cv2.rectangle(param, box_points[0], box_points[1], (0, 255, 0), 2)
        cv2.imshow("Template Cropper - Press C to Crop", param)

# GUI template cropping tool
def template_crop(image):
    clone = image.copy()
    cv2.namedWindow("Template Cropper - Press C to Crop")
    param = image
    cv2.setMouseCallback("Template Cropper - Press C to Crop", click_and_crop, param)
    while True:
        cv2.imshow("Template Cropper - Press C to Crop", image)
        key = cv2.waitKey(1)
        if key == ord("c"):
            cv2.destroyAllWindows()
            break

    cropped_region = clone
    if len(box_points) == 2:
        cropped_region = clone[box_points[0][1]:box_points[1][1], box_points[0][0]:box_points[1][0]]
    return cropped_region


# Resize Image

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def invariantMatchTemplate(rgbimage, rgbtemplate, method, matched_thresh, rgbdiff_thresh, rot_range, rot_interval, scale_range, scale_interval, rm_redundant, minmax):
    """
    rgbimage: RGB image where the search is running.
    rgbtemplate: RGB searched template. It must be not greater than the source image and have the same data type.
    method: [String] Parameter specifying the comparison method
    matched_thresh: [Float] Setting threshold of matched results(0~1).
    rgbdiff_thresh: [Float] Setting threshold of average RGB difference between template and source image.
    rot_range: [Integer] Array of range of rotation angle in degrees. Example: [0,360]
    rot_interval: [Integer] Interval of traversing the range of rotation angle in degrees.
    scale_range: [Integer] Array of range of scaling in percentage. Example: [50,200]
    scale_interval: [Integer] Interval of traversing the range of scaling in percentage.
    rm_redundant: [Boolean] Option for removing redundant matched results based on the width and height of the template.
    minmax:[Boolean] Option for finding points with minimum/maximum value.

    Returns: List of satisfied matched points in format [[point.x, point.y], angle, scale].
    """
    img_gray = rgbimage
    template_gray = rgbtemplate
    image_maxwh = img_gray.shape
    height, width = template_gray.shape
    all_points = []
    if minmax == False:
        for next_angle in range(rot_range[0], rot_range[1], rot_interval):
            for next_scale in range(scale_range[0], scale_range[1], scale_interval):
                scaled_template_gray, actual_scale = scale_image(template_gray, next_scale, image_maxwh)
                if next_angle == 0:
                    rotated_template = scaled_template_gray
                else:
                    rotated_template = rotate_image(scaled_template_gray, next_angle)
                if method == "TM_CCOEFF":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCOEFF)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCOEFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCOEFF_NORMED)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCORR":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCORR)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCORR_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCORR_NORMED)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_SQDIFF":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_SQDIFF)
                    satisfied_points = np.where(matched_points <= matched_thresh)
                elif method == "TM_SQDIFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_SQDIFF_NORMED)
                    satisfied_points = np.where(matched_points <= matched_thresh)
                else:
                    raise MethodError("There's no such comparison method for template matching.")
                for pt in zip(*satisfied_points[::-1]):
                    all_points.append([pt, next_angle, actual_scale])
    else:
        for next_angle in range(rot_range[0], rot_range[1], rot_interval):
            for next_scale in range(scale_range[0], scale_range[1], scale_interval):
                scaled_template_gray, actual_scale = scale_image(template_gray, next_scale, image_maxwh)

                if next_angle == 0:
                    rotated_template = scaled_template_gray
                else:
                    rotated_template = rotate_image(scaled_template_gray, next_angle)
                if method == "TM_CCOEFF":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCOEFF)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                elif method == "TM_CCOEFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                elif method == "TM_CCORR":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCORR)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                elif method == "TM_CCORR_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_CCORR_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val])
                elif method == "TM_SQDIFF":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_SQDIFF)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if min_val <= matched_thresh:
                        all_points.append([min_loc, next_angle, actual_scale, min_val])
                elif method == "TM_SQDIFF_NORMED":
                    matched_points = cv2.matchTemplate(img_gray,rotated_template,cv2.TM_SQDIFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if min_val <= matched_thresh:
                        all_points.append([min_loc, next_angle, actual_scale, min_val])
                else:
                    raise MethodError("There's no such comparison method for template matching.")
        if method == "TM_CCOEFF":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCOEFF_NORMED":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCORR":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCORR_NORMED":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_SQDIFF":
            all_points = sorted(all_points, key=lambda x: x[3])
        elif method == "TM_SQDIFF_NORMED":
            all_points = sorted(all_points, key=lambda x: x[3])
    if rm_redundant == True:
        lone_points_list = []
        visited_points_list = []
        for point_info in all_points:
            point = point_info[0]
            scale = point_info[2]
            all_visited_points_not_close = True
            if len(visited_points_list) != 0:
                for visited_point in visited_points_list:
                    if ((abs(visited_point[0] - point[0]) < (width * scale / 100)) and (abs(visited_point[1] - point[1]) < (height * scale / 100))):
                        all_visited_points_not_close = False
                if all_visited_points_not_close == True:
                    lone_points_list.append(point_info)
                    visited_points_list.append(point)
            else:
                lone_points_list.append(point_info)
                visited_points_list.append(point)
        points_list = lone_points_list
    else:
        points_list = all_points
    color_filtered_list = []
    # template_channels = cv2.mean(rgbtemplate)
    # template_channels = np.array([template_channels[0], template_channels[1], template_channels[2]])
    for point_info in points_list:
        color_filtered_list.append([point_info[0], point_info[1], point_info[2], point_info[3]])
        # point = point_info[0]
        # cropped_img = rgbimage[point[1]:point[1]+height, point[0]:point[0]+width]
        # cropped_channels = cv2.mean(cropped_img)
        # cropped_channels = np.array([cropped_channels[0], cropped_channels[1], cropped_channels[2]])
        # diff_observation = cropped_channels - template_channels
        # total_diff = np.sum(np.absolute(diff_observation))
        # print(total_diff)
        # if total_diff < rgbdiff_thresh:
    return color_filtered_list


def main():
    templateMatchModes = [
        "TM_SQDIFF",
        "TM_SQDIFF_NORMED",
        "TM_CCORR",
        "TM_CCORR_NORMED",
        "TM_CCOEFF",
        "TM_CCOEFF_NORMED",
    ]

    # Configure the Logo and Images Path
    print("Keep in mind that only jpg format is accepted")
    print()
    print("Which logo would you like to detect? Write the logo folder name.")
    print()

    os_paths = []
    logoname = ''
    base_template_width = 200

    while True:
        logoname = input()
        if not os.path.exists('logo/' + logoname):
            print("Path: " + 'logo/' + logoname + "is not existed")
        else:
            os_paths = [('logo/' + logoname + '/' + entry) for entry in os.listdir('logo/' + logoname) if (".jpg" in entry) or (".jpeg" in entry)]
            if len(os_paths) > 0:
                break
            else:
                print("File Path: " + 'logo/' + logoname)
                print("Logo file you input is not existed in logo folder!")


    print(os_paths)
    print("Detected logo files: " + str(len(os_paths)))
    print("Loading image files in image folder....")

    dir_path = 'images'
    arr_logo_detection = []

    for entry in tqdm(os.listdir(dir_path)):

        if not (os.path.isfile(os.path.join(dir_path, entry)) and (".jpg" in entry)):
            continue

        img_rgb = cv2.imread(os.path.join(dir_path, entry), cv2.IMREAD_GRAYSCALE)
        points_list = []
        for os_path in os_paths:
            cropped_template_rgb = cv2.imread(os_path, cv2.IMREAD_GRAYSCALE)
            cropped_template_rgb = image_resize(cropped_template_rgb , width=base_template_width)
            print(os_path)
            # cropped_template_rgb = template_crop(template_rgb)
            cropped_template_rgb = np.array(cropped_template_rgb)
            # cropped_template_gray = cv2.cvtColor(cropped_template_rgb, cv2.COLOR_RGB2GRAY)
            height, width = cropped_template_rgb.shape

            points_list = points_list + invariantMatchTemplate(img_rgb, cropped_template_rgb, templateMatchModes[5], 0.6, 600,
                                                 [0, 10], 10,
                                                 [30, 200], 1, True, True)
            if len(points_list) > 0:
                break
        print(points_list)
        if len(points_list) > 0:
            arr_logo_detection.append([entry, 1])
        else:
            arr_logo_detection.append([entry, 0])
        # fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        # plt.gcf().canvas.manager.set_window_title('Template Matching Results')
        # ax.imshow(img_rgb, cmap='gray')
        # centers_list = []
        # for point_info in points_list:
        #     point = point_info[0]
        #     angle = point_info[1]
        #     scale = point_info[2]
        #     centers_list.append([point, scale])
        #     plt.scatter(point[0] + (width / 2) * scale / 100, point[1] + (height / 2) * scale / 100, s=20, color="red")
        #     plt.scatter(point[0], point[1], s=20, color="green")
        #     rectangle = patches.Rectangle((point[0], point[1]), width * scale / 100, height * scale / 100, color="red",
        #                                   alpha=0.50, label='Matched box')
        #     box = patches.Rectangle((point[0], point[1]), width * scale / 100, height * scale / 100, color="green",
        #                             alpha=0.50, label='Bounding box')
        #     transform = mpl.transforms.Affine2D().rotate_deg_around(point[0] + width / 2 * scale / 100,
        #                                                             point[1] + height / 2 * scale / 100,
        #                                                             angle) + ax.transData
        #     rectangle.set_transform(transform)
        #     ax.add_patch(rectangle)
        #     ax.add_patch(box)
        #     plt.legend(handles=[rectangle, box])
        # # plt.grid(True)
        #
        # ax2.imshow(img_rgb, cmap='gray')
        # for point_info in centers_list:
        #     point = point_info[0]
        #     scale = point_info[1]
        #     plt.scatter(point[0] + width / 2 * scale / 100, point[1] + height / 2 * scale / 100, s=20, color="red")
        #
        # plt.show()

    pd.DataFrame(arr_logo_detection).to_csv('csv/' + logoname +'.csv', index_label="No", header=['ImageName', 'LogoDetect'])

if __name__ == "__main__":
    main()