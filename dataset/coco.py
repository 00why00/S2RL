"""COCO feature extraction and S²-RL prompt datasets."""

import os
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


COCO_dict = {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    "4": "airplane",
    "5": "bus",
    "6": "train",
    "7": "truck",
    "8": "boat",
    "9": "traffic light",
    "10": "fire hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench",
    "14": "bird",
    "15": "cat",
    "16": "dog",
    "17": "horse",
    "18": "sheep",
    "19": "cow",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "broccoli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "tv",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush",
    "80": "banner",
    "81": "blanket",
    "82": "bridge",
    "83": "cardboard",
    "84": "counter",
    "85": "curtain",
    "86": "door",
    "87": "wood floor",
    "88": "flower",
    "89": "fruit",
    "90": "gravel",
    "91": "house",
    "92": "light",
    "93": "mirror",
    "94": "net",
    "95": "pillow",
    "96": "platform",
    "97": "playing field",
    "98": "railroad",
    "99": "river",
    "100": "road",
    "101": "roof",
    "102": "sand",
    "103": "sea",
    "104": "shelf",
    "105": "snow",
    "106": "stairs",
    "107": "tent",
    "108": "towel",
    "109": "brick wall",
    "110": "stone wall",
    "111": "tile wall",
    "112": "wood wall",
    "113": "water",
    "114": "window blind",
    "115": "window",
    "116": "tree",
    "117": "fence",
    "118": "ceiling",
    "119": "sky",
    "120": "cabinet",
    "121": "table",
    "122": "floor",
    "123": "pavement",
    "124": "mountain",
    "125": "grass",
    "126": "dirt",
    "127": "paper",
    "128": "food",
    "129": "building",
    "130": "rock",
    "131": "wall",
    "132": "rug"
}


class COCODataset(Dataset):
    def __init__(self, data_root, phase='val'):
        self.data = []
        prefix = 'val2017' if phase == 'val' else 'train2017'
        files = os.listdir(data_root + '/' + prefix + '/')
        for fn in files:
            self.data.append({
                'image': data_root + '/' + prefix + '/' + fn,
                'label': data_root + '/annotations/panoptic_semantic_' + prefix + '/' + fn.replace('.jpg', '.png')
            })

        self.name_dic = {}
        for k in COCO_dict.keys():
            self.name_dic[int(k)] = COCO_dict[k]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_filename, label_filename = item['image'], item['label']
        basename = os.path.basename(image_filename).split('.')[0]

        label = np.array(Image.open(label_filename).resize((512, 512), Image.Resampling.NEAREST))
        image = Image.open(image_filename).resize((512, 512), Image.Resampling.BICUBIC).convert('RGB')

        prompt = ''
        for lb in np.unique(label):
            if lb == 255:
                continue
            prompt += self.name_dic[lb] + ','

        return dict(prompt=prompt, image=image, label=label, basename=basename)


class COCOSampleTextDataset(Dataset):
    def __init__(
        self,
        data_root=os.path.join(os.environ.get("S2RL_DATA_ROOT", "../dataset"), 'ILDistillation/COCO'),
        sample_type=None,
        selected_images=None,
    ):
        self.data = []
        files = os.listdir(os.path.join(data_root, sample_type, 'images', selected_images))
        for fn in files:
            self.data.append({
                'image': os.path.join(data_root, sample_type, 'images', selected_images, fn),
                'label': os.path.join(data_root, sample_type, 'annotations', selected_images, fn.replace('.jpg', '.png'))
            })

        self.name_dic = {}
        for k in COCO_dict.keys():
            self.name_dic[int(k)] = COCO_dict[k]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_filename, label_filename = item['image'], item['label']
        basename = os.path.basename(image_filename).split('.')[0]
        label = np.array(Image.open(label_filename).resize((512, 512), Image.Resampling.NEAREST))

        prompt = ''
        class_ids = []
        for lb in np.unique(label):
            if lb == 255:
                continue
            prompt += self.name_dic[lb] + ','
            class_ids.append(lb)

        return dict(prompt=prompt, class_ids=class_ids)


class COCOAllSampleTextDataset(Dataset):
    def __init__(
        self,
        data_root=os.path.join(os.environ.get("S2RL_DATA_ROOT", "../dataset"), 'COCO'),
        sample_root=os.path.join(os.environ.get("S2RL_DATA_ROOT", "../dataset"), 'ILDistillation/COCO'),
        sample_type=None,
    ):
        self.data = []
        # The policy uses the union of the three main-experiment budgets.
        ratios = ('0.000625', '0.00125', '0.0025')
        files = []
        for ratio in ratios:
            files.extend(glob(os.path.join(sample_root, sample_type, 'images', f"time_*_ratio_{ratio}", "*.jpg")))
        files = sorted(set([os.path.basename(file) for file in files]))
        for fn in files:
            self.data.append({
                'image': os.path.join(data_root, 'train2017', fn),
                'label': os.path.join(data_root, 'annotations/panoptic_semantic_train2017', fn.replace('.jpg', '.png'))
            })

        self.name_dic = {}
        for k in COCO_dict.keys():
            self.name_dic[int(k)] = COCO_dict[k]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_filename, label_filename = item['image'], item['label']
        basename = os.path.basename(image_filename).split('.')[0]
        label = np.array(Image.open(label_filename).resize((512, 512), Image.Resampling.NEAREST))

        prompt = ''
        class_ids = []
        for lb in np.unique(label):
            if lb == 255:
                continue
            prompt += self.name_dic[lb] + ','
            class_ids.append(lb)

        return dict(prompt=prompt, class_ids=class_ids)
