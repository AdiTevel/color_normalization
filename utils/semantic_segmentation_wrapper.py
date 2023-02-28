import numpy as np
# import rospy
import cv2
# from std_msgs.msg import Float32, Int8

from tevelnnservice.semantic_segmentation.SemanticSegmentationClient import SemanticSegmentationClient


class SemanticSegmentationWrapper:

    def __init__(self):
        self._client = SemanticSegmentationClient()
        # self._ros_stats_call_publisher = rospy.Publisher('/profiling/vision/semantic_segmentation/call', Int8, queue_size=1)
        # self._ros_stats_time_publisher = rospy.Publisher('/profiling/vision/semantic_segmentation/time', Float32, queue_size=1)
        self.width_full_image = -1  # TODO(assaf)
        self.height_full_image = -1
        self.number_of_labels = None
        self.radius_scaling_in_batch = 20  # TODO(assaf)
        self.ratio_radius_in_pixel_score = 0.7  # should be between 0-1
        self.calculate_connected_space = False
        self.calculate_scores = False
        self.calculate_full_image = True
        self.calculate_branch_size_between_blobs = False
        self.background_index_in_mask = 0  # TODO(assaf)
        self.fruit_index_in_mask = 1
        self.branch_index_in_mask = 2
        self.leaves_index_in_mask = 3

    def infer_mask(self, rgb, circles=None, batch_segmentation=False):
        self.width_full_image = rgb.shape[1]
        self.height_full_image = rgb.shape[0]
        # self._ros_stats_call_publisher.publish(1)
        # start_time = rospy.get_time()
        masks, scores, mask_channels, radius_scaling_in_batch, times, original_position_in_rgb, network_name = self._client.infer_mask(rgb, circles, batch_segmentation)
        # end_time = rospy.get_time()
        # self._ros_stats_time_publisher.publish(end_time - start_time)
        self.number_of_labels = mask_channels
        self.radius_scaling_in_batch = radius_scaling_in_batch
        full_image, full_image_binary_only_fruit, masks_data = self._post_process_masks(masks, circles, radius_scaling_in_batch, original_position_in_rgb, scores)
        return full_image, full_image_binary_only_fruit, masks_data

    def infer_mask_without_scores(self, rgb, circles=None, batch_segmentation=False):
        self.width_full_image = rgb.shape[1]
        self.height_full_image = rgb.shape[0]
        # self._ros_stats_call_publisher.publish(1)
        # start_time = rospy.get_time()
        masks, scores, mask_channels, radius_scaling_in_batch, times, original_position_in_rgb, network_name = self._client.infer_mask(rgb, circles, batch_segmentation)
        # end_time = rospy.get_time()
        # self._ros_stats_time_publisher.publish(end_time - start_time)
        self.number_of_labels = mask_channels
        self.radius_scaling_in_batch = radius_scaling_in_batch
        full_image, full_image_binary_only_fruit, masks_data = self._post_process_masks(masks, circles, radius_scaling_in_batch, original_position_in_rgb, scores)
        return full_image, full_image_binary_only_fruit, masks_data

    def _post_process_masks(self, masks, circles, radius_scaling_in_batch, original_position_of_mask_in_rgb, scores):
        # mask is already in the relevant size of the rgb, however he is bigger than the bbox. the radios used before network was original_radios + self.radius_scaling_in_batch
        output_color_mask = np.zeros((self.height_full_image, self.width_full_image, 3), dtype=np.uint8)  # empty full image
        output_binary_mask_fruit = np.zeros((self.height_full_image, self.width_full_image), dtype=bool)  # empty full image
        res = []

        if self.number_of_labels is None or len(circles) == 0:
            return output_color_mask, output_binary_mask_fruit, res

        for bbox, mask, org_pos, score in zip(circles, masks, original_position_of_mask_in_rgb, scores):
            bbox_in_mask = self._get_bbox_in_mask(mask, bbox, org_pos)
            mask_cropped = self._crop_mask_according_bbox(mask, bbox_in_mask)

            data = {"bbox": bbox,
                    "mask": mask,
                    "original_position_of_mask_in_rgb": org_pos,
                    "radius_increment": radius_scaling_in_batch,
                    "mask_cropped_according_bbox": mask_cropped,
                    "bbox_in_mask": bbox_in_mask
                    }

            if self.calculate_scores:
                data["all_mask_ratios"] = self._get_ratios_around_bbox(mask, bbox_in_mask)
                data["middle_of_bbox_ratios"] = self._get_ratios_around_bbox(mask, (bbox_in_mask[0], bbox_in_mask[1], bbox_in_mask[2] * self.ratio_radius_in_pixel_score))
                data["pixel_score"] = self._get_pixel_score(score, bbox)
                if self.calculate_connected_space:
                    data.update(self._calculate_connected_spaces_in_mask(self._crop_mask_according_bbox(mask, bbox_in_mask))
                                )

            if self.calculate_full_image:
                self._insert_mask_to_image(output_color_mask, output_binary_mask_fruit, bbox, mask_cropped,
                                           radius_scaling_in_batch)

            res.append(data)
        return output_color_mask, output_binary_mask_fruit, res

    def _post_process_masks_without_scores(self, masks, circles, radius_scaling_in_batch, original_position_of_mask_in_rgb, scores):
        # mask is already in the relevant size of the rgb, however he is bigger than the bbox. the radios used before network was original_radios + self.radius_scaling_in_batch
        output_color_mask = np.zeros((self.height_full_image, self.width_full_image, 3), dtype=np.uint8)  # empty full image
        output_binary_mask_fruit = np.zeros((self.height_full_image, self.width_full_image), dtype=bool)  # empty full image
        res = []

        if self.number_of_labels is None or len(circles) == 0:
            return output_color_mask, output_binary_mask_fruit, res

        for bbox, mask, org_pos, score in zip(circles, masks, original_position_of_mask_in_rgb, scores):
            bbox_in_mask = self._get_bbox_in_mask(mask, bbox, org_pos)
            data = {"bbox": bbox,
                    "mask": mask,
                    "original_position_of_mask_in_rgb": org_pos,
                    "radius_increment": radius_scaling_in_batch,
                    "mask_cropped_according_bbox": self._crop_mask_according_bbox(mask, bbox_in_mask),
                    "bbox_in_mask": bbox_in_mask
                    }

            res.append(data)
        return output_color_mask, output_binary_mask_fruit, res

    def _get_nipple_radius_in_pixels(self):
        return 5   # TODO(assaf)

    def _get_pixel_score(self, scores, bbox):
        """
        calculate pixel score from origin mask.
        >>> s = SemanticSegmentationWrapper()
        >>> s._get_pixel_score(np.concatenate([np.zeros((64,80,1)), np.ones([64,80,1])],2),(300,400,20))
        1.0
        >>> s._get_pixel_score(np.concatenate([np.ones([64,80,1]), np.zeros((64,80,1))],2),(300,400,20))
        0.0
        """
        ratio = bbox[2] / (bbox[2] + self.radius_scaling_in_batch)
        score_shape_0, score_shape_1, channels = scores.shape
        w, h = ratio * score_shape_0, ratio * score_shape_1
        x_c, y_c = score_shape_0 / 2, score_shape_1 / 2
        score_crop = scores[int(x_c-w/2):int(x_c+w/2), int(y_c-h/2):int(y_c+h/2), :]
        mask_crop = np.argmax(score_crop, axis=2).astype(np.uint8)
        pixel_score = np.sum(mask_crop == self.fruit_index_in_mask) / (w * h)
        return pixel_score


    def _get_ratios(self, mask):
        """
        calculate the ratios of the different labels in the mask
        >>> s = SemanticSegmentationWrapper()
        >>> s.number_of_labels = 2
        >>> m = np.array([np.zeros(5), np.ones(5)])
        >>> s._get_ratios(m)
        [0.5, 0.5]
        >>> s.number_of_labels = 4
        >>> m = np.concatenate((m, [np.ones(5)*2, np.ones(5)*3]), axis=0)
        >>> s._get_ratios(m)
        [0.25, 0.25, 0.25, 0.25]
        >>> s.number_of_labels = 5
        >>> m = np.concatenate((m, np.ones((4,1)) * 4), axis=1)
        >>> s._get_ratios(m)
        [0.20833333333333334, 0.20833333333333334, 0.20833333333333334, 0.20833333333333334, 0.16666666666666666]
        """
        ratio = [None] * self.number_of_labels
        unique_indexes_all_mask, counts_all_mask = np.unique(mask, return_counts=True)
        size_of_mask = mask.shape[0] * mask.shape[1]
        unique_i = 0
        for i in range(self.number_of_labels):
            if len(unique_indexes_all_mask) == unique_i or unique_indexes_all_mask[unique_i] != i:
                ratio[i] = 0
            else:
                ratio[i] = counts_all_mask[unique_i] / size_of_mask
                unique_i = unique_i + 1
        return ratio

    def _get_ratios_around_bbox(self, mask, bbox):
        # bbox here should be pixel values in the mask
        mask_cropped = self._crop_mask_according_bbox(mask, bbox)
        return self._get_ratios(mask_cropped)

    def _find_branch_size_between_points(self, mask, center1, center2):
        if center1[0] > center2[0]:
            x_max = center1[0]
            x_min = center2[0]
            y_max = center1[1]
            y_min = center2[1]
        else:
            x_max = center2[0]
            x_min = center1[0]
            y_max = center2[1]
            y_min = center1[1]

        cnt_total = 0
        cnt_branch = 0
        for x in range(x_min+1, x_max):
            y = self._get_y_of_x(x, x_min, y_min, x_max, y_max)
            cnt_total += 1
            if mask[x, y]:
                cnt_branch += 1

        return cnt_branch, cnt_total

    @staticmethod
    def _get_y_of_x(x, x1, y1, x2, y2):
        """
        returns y given x on a straight line between two points (x1, y1) to (x2, y2).
        >>> s = SemanticSegmentationWrapper()
        >>> s._get_y_of_x(4,1,1,8,8)
        4
        >>> s._get_y_of_x(10,1,1,8,8)
        10
        """
        return int((x - x1) / (x2 - x1) * (y2 - y1) + y1)

    def _insert_mask_to_image(self, image_color, image_binary_only_fruit, bbox, mask, radius_scaling_in_batch):
        # bbox values here should be pixel value in the original image.
        xc, yc, r = bbox
        r = r #+ radius_scaling_in_batch
        x1, y1 = int(np.clip(xc - r, 0, self.width_full_image - 1)), int(np.clip(yc - r, 0, self.height_full_image - 1))
        x2, y2 = int(np.clip(xc + r, 0, self.width_full_image - 1)), int(np.clip(yc + r, 0, self.height_full_image - 1))
        image_binary_only_fruit[y1:y2, x1:x2] = mask == self.fruit_index_in_mask
        if self.number_of_labels == 2:
            mask_colors = np.stack([mask * 1, mask * 1, mask * 255]).transpose(1, 2, 0)
        else:
            mask_colors = cv2.applyColorMap(1 + mask * 30, cv2.COLORMAP_HSV)
        image_color[y1:y2, x1:x2, :] = mask_colors

    def _calculate_connected_spaces_in_mask(self, mask):
        data = {}
        mask_only_fruit = mask == self.fruit_index_in_mask
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(mask_only_fruit.astype(np.uint8), 4, cv2.CV_32S)
        data["number_of_fruit_blobs"] = num_labels - 1

        # three biggest blobs, including background
        biggest_blob = -1, -1, -1     # area, index, label
        second_biggest = -1, -1, -1   # area, index, label
        for i in range(num_labels):
            label = sum(mask_only_fruit[labels == i]) / stats[i, cv2.CC_STAT_AREA]
            if label == self.background_index_in_mask or stats[i, cv2.CC_STAT_AREA] == 0:
                continue
            if stats[i, cv2.CC_STAT_AREA] > biggest_blob[0]:
                second_biggest = biggest_blob
                biggest_blob = stats[i, cv2.CC_STAT_AREA], i, label
            elif stats[i, cv2.CC_STAT_AREA] > second_biggest[0]:
                second_biggest = stats[i, cv2.CC_STAT_AREA], i, label

        data["biggest_blob_size"] = biggest_blob[0]
        data["biggest_blob_label"] = biggest_blob[2]
        data["biggest_blob_centroid"] = centroids[biggest_blob[1]]
        data["distance_between_biggest_blob_centroid_to_middle_mask"] = self._calculate_distance_between_pixel_and_mid_mask(centroids[biggest_blob[1]], mask)
        data["biggest_blob_ratios"] = \
            self._get_ratios_around_bbox(mask,
                                         (centroids[biggest_blob[1]][0],
                                          centroids[biggest_blob[1]][1],
                                          self._get_nipple_radius_in_pixels()
                                          )
                                         )
        if not second_biggest[0] == -1:
            data["second_biggest_size"] = second_biggest[0]
            data["second_blob_label"] = second_biggest[2]
            data["second_biggest_centroid"] = centroids[second_biggest[1]]
            data["second_biggest_ratios"] = \
                self._get_ratios_around_bbox(mask,
                                             (centroids[second_biggest[1]][0],
                                              centroids[second_biggest[1]][1],
                                              self._get_nipple_radius_in_pixels()
                                              )
                                             )

        if self.calculate_branch_size_between_blobs and self.number_of_labels == 4:
            pixels_between_with_branch, pixels_between_total = \
                self._find_branch_size_between_points(mask == self.branch_index_in_mask,
                                                      data["bigger_blob_centroid"],
                                                      data["second_bigger_blob_centroid"])
            data["branch_between_centroids_ratio"] = pixels_between_with_branch / pixels_between_total


        return data

    @staticmethod
    def _crop_mask_according_xc_yc_r(mask, xc, yc, r):
        xc, yc, r = xc, yc, r
        x1, y1 = int(np.clip(xc - r, 0, mask.shape[1] - 1)), int(np.clip(yc - r, 0, mask.shape[0] - 1))
        x2, y2 = int(np.clip(xc + r, 0, mask.shape[1] - 1)), int(np.clip(yc + r, 0, mask.shape[0] - 1))
        return mask[y1:y2, x1:x2]

    def _crop_mask_according_bbox(self, mask, bbox):
        return self._crop_mask_according_xc_yc_r(mask, bbox[0], bbox[1], bbox[2])

    @staticmethod
    def _calculate_distance_between_pixel_and_mid_mask(pixel, mask):
        """
        pixel is (x,y) while mask. shape is y,x
        """
        return np.sqrt((pixel[0] - mask.shape[1])**2 + (pixel[1] - mask.shape[0])**2)

    def _get_bbox_in_mask(self, mask, bbox, original_position_of_mask_in_rgb):
        xc_mask, yc_mask = self._get_middle_of_bbox_in_mask(mask, bbox, original_position_of_mask_in_rgb)
        return xc_mask, yc_mask, bbox[2]

    def _get_middle_of_bbox_in_mask(self, mask, bbox, original_position_of_mask_in_rgb):
        """
        >>> s = SemanticSegmentationWrapper()
        >>> s._get_middle_of_bbox_in_mask(np.zeros([64,64]), (100,100,20), ((60,60),(140,140)))
        (32, 32)
        >>> s._get_middle_of_bbox_in_mask(np.zeros([64,64]), (100,100,20), ((60,60),(120,140)))
        (42, 32)
        >>> s._get_middle_of_bbox_in_mask(np.zeros([64,64]), (100,100,20), ((60,80), (140,140)))
        (32, 21)
        >>> s._get_middle_of_bbox_in_mask(np.zeros([48,64]), (100,100,20), ((60,60), (140,140)))
        (24, 32)
        """
        xc, yc, r = bbox
        x1, y1 = original_position_of_mask_in_rgb[0]
        x2, y2 = original_position_of_mask_in_rgb[1]
        xc_mask = self._get_middle_pixel_in_line(x1, xc, x2, mask.shape[1])
        yc_mask = self._get_middle_pixel_in_line(y1, yc, y2, mask.shape[0])
        return int(xc_mask), int(yc_mask)

    @staticmethod
    def _get_middle_pixel_in_line(x1, xc, x2, mask_shape_in_x):
        """
        >>> SemanticSegmentationWrapper._get_middle_pixel_in_line(100, 150, 200, 64)
        32.0
        >>> SemanticSegmentationWrapper._get_middle_pixel_in_line(125, 150, 200, 64)
        21.333333333333336
        >>> SemanticSegmentationWrapper._get_middle_pixel_in_line(100, 105, 125, 64)
        12.8
        >>> SemanticSegmentationWrapper._get_middle_pixel_in_line(100, 105, 125, 90)
        18.0
        """
        return mask_shape_in_x / (x2 - x1) * (xc - x1)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
