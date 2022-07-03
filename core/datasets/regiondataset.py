import cv2
import torch
import random
import numpy as np
import imgaug.augmenters as iaa
import json


from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class RegionDataset(Dataset):
    def __init__(self, dirnames, classes, image_size, image_patterns, mask_pattern, is_textline=False, ignore_field = [],
                h_factor=None, w_factor=None, is_pad_to_square=False, transforms=None, require_transforms=None):
        super(RegionDataset, self).__init__()
        self.classes = classes
        self.ignore_field = ignore_field
        self.image_size = image_size
        self.transforms = transforms if transforms else []
        self.require_transforms = require_transforms if require_transforms else []
        if is_pad_to_square:
            self.pad_to_square = iaa.PadToSquare(position='right-bottom')
        self.is_textline = is_textline
        self.is_pad_to_square = is_pad_to_square
        self.h_factor = h_factor
        self.w_factor = w_factor
        
        image_paths, label_paths = [], []

        for dirname in dirnames:
            for image_extent in image_patterns:
                image_paths.extend(list(Path(dirname).glob('**/*{}'.format(image_extent))))

            label_paths.extend(list(Path(dirname).glob('**/*{}'.format(mask_pattern))))

        image_paths = natsorted(image_paths, key=lambda x: x.stem)
        label_paths = natsorted(label_paths, key=lambda x: x.stem)

        self.data_pairs = [(image, label)for image, label in zip(image_paths, label_paths) if image.stem == label.stem]
        
        print('--' * 20)
        print(f'[___Dataset Info___]: {Path(dirnames[0]).parent.stem} - {len(self.data_pairs)} pairs.')

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.data_pairs[idx]
        #image, mask = cv2.imread(str(image_path)), cv2.imread(str(mask_path))
        image = cv2.imread(str(image_path))
        mask = self._get_mask(str(mask_path), image.shape)

        if image.shape != mask.shape:
            print(image_path)
            raise ValueError('image and mask does not have the same shape.')

        image_info = [str(image_path), image.shape[1::-1]]
        if self.is_pad_to_square:
            image = self.pad_to_square(image=image)
            mask = self.pad_to_square(image=mask)
        mask = SegmentationMapsOnImage(mask, image.shape)

        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, mask = transform(image=image, segmentation_maps=mask)

        for require_transform in self.require_transforms:
            image, mask = require_transform(image=image, segmentation_maps=mask)

        mask = mask.get_arr()

        image, mask = cv2.resize(image, dsize=self.image_size), cv2.resize(mask, dsize=self.image_size, interpolation=cv2.INTER_NEAREST)
        
        class_map = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.int64)
        for field in self.classes:
            color = self.classes[field][0]
            color = np.asarray(color)
            color = np.expand_dims(np.expand_dims(color, axis=0), axis=0)
            class_map[(mask == color).prod(axis=-1) == 1] = self.classes[field][1]
            
        image, mask = np.ascontiguousarray(image), np.ascontiguousarray(class_map)
        image, mask = torch.from_numpy(image), torch.from_numpy(mask)
        image = image.permute(2, 0, 1).to(torch.float)
        image = (image - image.mean()) / image.std()
        return image, mask, image_info
    
    def _get_mask(self, json_path, image_shape):
        mask = np.zeros(image_shape, dtype=np.int32)
        with open(json_path) as fp:
            info = json.load(fp)['shapes']
        
        boxes = []
        for field in info:
            points = field['points'] if len(field['points']) == 4 \
                    else self._to_4points(field['points'])
            boxes.append(points)

        is_correct = self.Is_correct_direction(boxes[:20])

        if self.is_textline:
            keys = list(self.classes.keys())
            keys.remove('BG')
            color = self.classes[keys[0]][0]
            for field in info:
                if field['label'] in self.ignore_field: continue
                if field['label'] == 'BARCODE' and 'value' not in field: continue
                points = field['points'] if len(field['points']) == 4 \
                    else self._to_4points(field['points'])
                if is_correct:
                    if self.h_factor:
                        points = self._reduce_height(self._order_points(points), self.h_factor[0], self.h_factor[1])
                    if self.w_factor:
                        points = self._reduce_width(self._order_points(points), self.w_factor[0], self.w_factor[1])
                else:
                    if self.h_factor:
                        points = self._reduce_width(self._order_points(points), self.h_factor[0], self.h_factor[1])
                    if self.w_factor:
                        points = self._reduce_height(self._order_points(points), self.w_factor[0], self.w_factor[1])
                cv2.fillPoly(img=mask, pts=[np.int32(points)], color=color)
        else:
            for field_cfg in self.classes:
                for field in info:
                    if field_cfg == field['label']:
                        color = self.classes[field_cfg][0]
                        points = field['points'] if len(field['points']) == 4 \
                        else self._to_4points(field['points'])
                        if is_correct:
                            if self.h_factor:
                                points = self._reduce_height(self._order_points(points), self.h_factor[0], self.h_factor[1])
                            if self.w_factor:
                                points = self._reduce_width(self._order_points(points), self.w_factor[0], self.w_factor[1])
                        else:
                            if self.h_factor:
                                points = self._reduce_width(self._order_points(points), self.h_factor[0], self.h_factor[1])
                            if self.w_factor:
                                points = self._reduce_height(self._order_points(points), self.w_factor[0], self.w_factor[1])
                        cv2.fillPoly(img=mask, pts=[np.int32(points)], color=color)

        return mask

    def _reduce_height(self, points, reduce_ratio_1, reduce_ratio_2):
        reduced_points = np.zeros_like(points)
        reduced_points[0] = self._point_on_segment(points[0], points[3], 0.5 * reduce_ratio_1 * self._distance(points[0], points[3]))
        reduced_points[1] = self._point_on_segment(points[1], points[2], 0.5 * reduce_ratio_1 * self._distance(points[1], points[2]))
        reduced_points[2] = self._point_on_segment(points[2], points[1], 0.5 * reduce_ratio_2 * self._distance(points[1], points[2]))
        reduced_points[3] = self._point_on_segment(points[3], points[0], 0.5 * reduce_ratio_2 * self._distance(points[0], points[3]))
        return reduced_points.tolist()

    def _reduce_width(self, points, reduce_ratio_1, reduce_ratio_2):
        reduced_points = np.zeros_like(points)
        reduced_points[0] = self._point_on_segment(points[0], points[1], 0.5 * reduce_ratio_1 * self._distance(points[0], points[1]))
        reduced_points[1] = self._point_on_segment(points[1], points[0], 0.5 * reduce_ratio_1 * self._distance(points[0], points[1]))
        reduced_points[2] = self._point_on_segment(points[2], points[3], 0.5 * reduce_ratio_2 * self._distance(points[2], points[3]))
        reduced_points[3] = self._point_on_segment(points[3], points[2], 0.5 * reduce_ratio_2 * self._distance(points[2], points[3]))
        return reduced_points.tolist()

    def _point_on_segment(self, point1, point2, length_from_point1):
        point1 = np.float32(point1)
        point2 = np.float32(point2)
        alpha = length_from_point1 / self._distance(point1, point2)
        point = alpha * point2 + (1 - alpha) * point1
        return point.round().astype(np.int32).tolist()

    def _distance(self, point1, point2):
        point1 = np.float32(point1)
        point2 = np.float32(point2)
        return np.linalg.norm(point1 - point2)

    def _order_points(self, points):
        '''
        tl = min(points, key=lambda p: p[0] + p[1])
        br = max(points, key=lambda p: p[0] + p[1])
        tr = max(points, key=lambda p: p[0] - p[1])
        bl = min(points, key=lambda p: p[0] - p[1])
        '''
        sorted_points = sorted(points, key=lambda p: p[0] * p[1])
        tl, br = sorted_points[0], sorted_points[-1]
        tr = min(sorted_points[1:-1], key=lambda p: p[1])
        bl = max(sorted_points[1:-1], key=lambda p: p[1])
        return [tl, tr, br, bl]

    def _to_4points(self, points):
        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[1][0], points[1][1]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    def Is_correct_direction(self, boxes):
        vote = 0
        for box in boxes:
            points = self._order_points(box)
            if (self._distance(points[0], points[1]) + 1e-5) / (self._distance(points[0], points[3]) + 1e-5) > 3.0:
                vote += 1
        if vote > len(boxes) // 3:
            return True
        return False
