
from tevelnnservice.semantic_segmentation.SemanticSegmentationClient import SemanticSegmentationClient
import numpy as np
class SemanticSegmentationWrapper:
    def __init__(self):

        self._client = SemanticSegmentationClient()

    def infer(self,rgb,circles = None,batch_segmentation =False):
        masks, scores, mask_channels, radius_scaling_in_batch, times, original_position_in_rgb, network_name = \
            self._client.infer_mask(rgb, circles, batch_segmentation)
        mask_in_image=[]
        local_mask_array = []
        for pos,local_mask in zip(original_position_in_rgb,masks):
            local_mask = local_mask[20:-20,20:-20]
            local_mask = np.logical_or(local_mask == 1, local_mask == 3)
            local_mask_array.append(local_mask)
            mask = np.zeros((rgb.shape[0],rgb.shape[1]))
            mask[pos[0][1]+20:pos[1][1]-20,pos[0][0]+20:pos[1][0]-20] =  np.logical_or(local_mask ==1,local_mask ==3)

            mask_in_image.append(mask)

        return local_mask_array,mask_in_image
    def _post_process_masks(self, masks, circles, radius_scaling_in_batch, original_position_of_mask_in_rgb, scores):
        # mask is already in the relevant size of the rgb, however he is bigger than the bbox. the radios used before network was original_radios + self.radius_scaling_in_batch
        output_color_mask = np.zeros((self.height_full_image, self.width_full_image, 3),
                                     dtype=np.uint8)  # empty full image
        output_binary_mask_fruit = np.zeros((self.height_full_image, self.width_full_image),
                                            dtype=bool)  # empty full image
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

            if self.calculate_scores:
                data["all_mask_ratios"] = self._get_ratios_around_bbox(mask, bbox_in_mask)
                data["middle_of_bbox_ratios"] = self._get_ratios_around_bbox(mask, (
                bbox_in_mask[0], bbox_in_mask[1], bbox_in_mask[2] * self.ratio_radius_in_pixel_score))
                data["pixel_score"] = self._get_pixel_score(score, bbox)
                if self.calculate_connected_space:
                    data.update(
                        self._calculate_connected_spaces_in_mask(self._crop_mask_according_bbox(mask, bbox_in_mask))
                        )

            if self.calculate_full_image:
                self._insert_mask_to_image(output_color_mask, output_binary_mask_fruit, mask, org_pos)

            res.append(data)
        return output_color_mask, output_binary_mask_fruit, res

    def _post_process_masks_without_scores(self, masks, circles, radius_scaling_in_batch,
                                           original_position_of_mask_in_rgb, scores):
        # mask is already in the relevant size of the rgb, however he is bigger than the bbox. the radios used before network was original_radios + self.radius_scaling_in_batch
        output_color_mask = np.zeros((self.height_full_image, self.width_full_image, 3),
                                     dtype=np.uint8)  # empty full image
        output_binary_mask_fruit = np.zeros((self.height_full_image, self.width_full_image),
                                            dtype=bool)  # empty full image
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

            if self.calculate_full_image:
                self._insert_mask_to_image(output_color_mask, output_binary_mask_fruit, mask, org_pos)

            res.append(data)
        return output_color_mask, output_binary_mask_fruit, res

if __name__=='__main__':
    import cv2
    fn = '/home/tevel/Pictures/snapshot.png'
    img = cv2.imread(fn)
    bbox= np.array([100,100,8])
    seg_obj = SemanticSegmentationWrapper()
    out = seg_obj.infer(img)
    print (out)
