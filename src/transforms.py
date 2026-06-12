import albumentations as A


LANDMARK_FLIP_ORDER = [1, 0, 2, 4, 3]

class FaceHorizontalFlip(A.HorizontalFlip):
    def apply_to_keypoints(self, keypoints, **params):
        keypoints = super().apply_to_keypoints(keypoints, **params)
        n_faces = len(keypoints) // 5
        reordered = []
        for i in range(n_faces):
            face_kps = keypoints[i*5:(i+1)*5]
            reordered.extend([face_kps[j] for j in LANDMARK_FLIP_ORDER])
        return reordered


TRANSFORM_REGISTRY = {
    "horizontal_flip": lambda **kw: FaceHorizontalFlip(**kw),
    "resize": lambda **kw: A.Resize(**kw),
    # TODO
    # "random_crop": lambda **kw: A.RandomCrop(**kw),
    # "rotate": lambda **kw: A.Rotate(**kw),
    "color_jitter": lambda **kw: A.ColorJitter(**kw),
    "random_brightness_contrast": lambda **kw: A.RandomBrightnessContrast(**kw),
    "hue_saturation_value": lambda **kw: A.HueSaturationValue(**kw),
    "gauss_noise": lambda **kw: A.GaussNoise(**kw),
    "blur": lambda **kw: A.Blur(**kw),
    "normalize": lambda **kw: A.Normalize(**kw),
    "image_compression": lambda **kw: A.ImageCompression(**kw),
}


def build_transforms(cfg, split):
    steps = []

    for item in cfg['transforms'][split]:
        name = item['name']
        kwargs = {k: v for k, v in item.items() if k != 'name'}
        steps.append(TRANSFORM_REGISTRY[name](**kwargs))

    return A.Compose(
        steps,
        bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']),
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    )
