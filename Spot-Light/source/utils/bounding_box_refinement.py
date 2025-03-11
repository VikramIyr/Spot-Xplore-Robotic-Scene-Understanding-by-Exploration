import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, Bounds


def cost (x, *args):
    """
    cost function for bounding box refinement, to be minimized
    @param x: bounding box coordinates in the form of [x1, y1, x2, y2]
    @param args: distance transform (heatmap), lambda
    @return: sclar cost
    """
    x1, y1, x2, y2 = x.astype(int)
    dist = args[0][0]
    l = args[0][1]
    area0 = args[0][2]
    center0 = args[0][3]

    top_edge = dist[y1, x1:x2 + 1]
    bottom_edge = dist[y2, x1:x2 + 1]
    left_edge = dist[y1:y2 + 1, x1]
    right_edge = dist[y1:y2 + 1, x2]

    max_cost_top_edge = top_edge.size
    max_cost_bottom_edge = bottom_edge.size
    max_cost_left_edge = left_edge.size
    max_cost_right_edge = right_edge.size

    # cost_distance = np.average(top_edge) + np.average(bottom_edge) + np.average(left_edge) + np.average(right_edge)

    cost_distance = np.array((np.sum(top_edge)/max_cost_top_edge, np.sum(bottom_edge)/max_cost_bottom_edge,
                     np.sum(left_edge)/max_cost_left_edge, np.sum(right_edge)/max_cost_right_edge)).mean()

    squaredness = (np.minimum(x2-x1, y2-y1)/
                   np.maximum(x2-x1, y2-y1))

    cost_squaredness = 1 - squaredness

    # area = (x2 - x1) * (y2 - y1)
    # if area > area0:
    #     area_cost = (area - area0) / area
    # else:
    #     area_cost = 0
    #
    # center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    # d_max = np.sqrt(dist.shape[0] ** 2 + dist.shape[1] ** 2)
    # center_cost = np.linalg.norm(center - center0) / d_max

    cost = cost_distance + l*cost_squaredness

    return cost


def refine_bounding_box(image: np.ndarray, bounding_box: np.ndarray, vis_block: bool = False, l: float = 0.5):
    """
    refine bounding box using distance transform local, non-convex, constrained optimization
    @param image: close-up image of the object
    @param bounding_box: initial bounding box in the form of [x1, y1, x2, y2]
    @param l: trade-off parameter between distance and squaredness cost
    @return: refined bounding box in the form of [x1, y1, x2, y2]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=10)
    line_image = np.zeros_like(gray)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)

    # build heatmap
    dist = cv2.distanceTransform(~line_image.astype(np.uint8), cv2.DIST_L2, 0)
    dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # optimize bounding box
    x0 = np.array([bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]])
    area0 = (x0[2] - x0[0]) * (x0[3] - x0[1])
    center0 = np.array([(x0[0] + x0[2]) / 2, (x0[1] + x0[3]) / 2])
    bounds = Bounds([0, 0, 0, 0], [image.shape[1]-1, image.shape[0]-1, image.shape[1]-1, image.shape[0]-1])
    res = minimize(cost, x0, method='nelder-mead', args=[dist, l, area0, center0],
                   options={'xatol': 1e-8, 'disp': False}, bounds=bounds)
    x_opt = res.x.astype(int)
    x0 = x0.astype(int)

    if vis_block:
        canv = copy.deepcopy(image)
        canv = cv2.cvtColor(canv, cv2.COLOR_BGR2RGB)
        cv2.rectangle(canv, (x_opt[0], x_opt[1]), (x_opt[2], x_opt[3]), (0, 255, 0), 2)
        cv2.rectangle(canv, (x0[0], x0[1]), (x0[2], x0[3]), (0, 0, 255), 2)
        plt.imshow(canv)
        plt.show()
        plt.figure(figsize=(20, 20))
        plt.imshow(line_image)
        plt.show()
        plt.figure(figsize=(20, 20))
        plt.imshow(dist)
        plt.show()

    return x_opt


if __name__ == '__main__':


    bb = np.array([130, 40, 450, 360]) #xyxy
    image = cv2.imread("/home/cvg-robotics/tim_ws/test_bounding_box_refinement.png")
    refine_bounding_box(image, bb, vis_block=True)