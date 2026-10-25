"""ADE20K feature extraction and S²-RL prompt datasets."""

import os
import random
from glob import glob

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset


ADE20K_dict = {
    '1':'wall',
    '2':'building',
    '3':'sky',
    '4':'floor',
    '5':'tree',
    '6':'ceiling',
    '7':'road',
    '8':'bed',
    '9':'window',
    '10':'grass',
    '11':'cabinet',
    '12':'sidewalk',
    '13':'person',
    '14':'ground',
    '15':'door',
    '16':'table',
    '17':'mountain',
    '18':'plant',
    '19':'curtain',
    '20':'chair',
    '21':'car',
    '22':'water',
    '23':'painting',
    '24':'sofa',
    '25':'shelf',
    '26':'house',
    '27':'sea',
    '28':'mirror',
    '29':'carpet',
    '30':'field',
    '31':'armchair',
    '32':'seat',
    '33':'fence',
    '34':'desk',
    '35':'stone',
    '36':'wardrobe',
    '37':'lamp',
    '38':'bathtub',
    '39':'railing',
    '40':'cushion',
    '41':'pedestal',
    '42':'box',
    '43':'pillar',
    '44':'signboard',
    '45':'dresser',
    '46':'counter',
    '47':'sand',
    '48':'sink',
    '49':'skyscraper',
    '50':'fireplace',
    '51':'refrigerator',
    '52':'grandstand',
    '53':'path',
    '54':'stairs',
    '55':'runway',
    '56':'showcase',
    '57':'pool table',
    '58':'pillow',
    '59':'screen door',
    '60':'stairway',
    '61':'river',
    '62':'bridge',
    '63':'bookcase',
    '64':'blinds',
    '65':'coffee table',
    '66':'toilet',
    '67':'flower',
    '68':'book',
    '69':'hill',
    '70':'bench',
    '71':'countertop',
    '72':'stove',
    '73':'palm',
    '74':'kitchen',
    '75':'computer',
    '76':'swivel chair',
    '77':'boat',
    '78':'bar',
    '79':'arcade machine',
    '80':'hovel',
    '81':'bus',
    '82':'towel',
    '83':'light',
    '84':'truck',
    '85':'tower',
    '86':'chandelier',
    '87':'awning',
    '88':'streetlight',
    '89':'booth',
    '90':'television',
    '91':'airplane',
    '92':'dirt road',
    '93':'apparel',
    '94':'pole',
    '95':'land',
    '96':'bannister',
    '97':'escalator',
    '98':'pouf',
    '99':'bottle',
    '100':'sideboard',
    '101':'poster',
    '102':'stage',
    '103':'van',
    '104':'ship',
    '105':'fountain',
    '106':'conveyor',
    '107':'canopy',
    '108':'washer',
    '109':'plaything',
    '110':'swimming pool',
    '111':'stool',
    '112':'barrel',
    '113':'basket',
    '114':'waterfall',
    '115':'tent',
    '116':'bag',
    '117':'motorbike',
    '118':'cradle',
    '119':'oven',
    '120':'ball',
    '121':'food',
    '122':'stair',
    '123':'tank',
    '124':'brand',
    '125':'microwave',
    '126':'pot',
    '127':'animal',
    '128':'bicycle',
    '129':'lake',
    '130':'dishwasher',
    '131':'silver screen',
    '132':'blanket',
    '133':'sculpture',
    '134':'hood',
    '135':'sconce',
    '136':'vase',
    '137':'traffic light',
    '138':'tray',
    '139':'ashcan',
    '140':'fan',
    '141':'pier',
    '142':'crt screen',
    '143':'plate',
    '144':'monitor',
    '145':'bulletin board',
    '146':'shower',
    '147':'radiator',
    '148':'glass',
    '149':'clock',
    '150':'flag'
}


# Preserve the feature reader's original caption names.
_FEATURE_CLASS_NAMES = {**ADE20K_dict, "131": "screen", "142": "screen"}


class ADE20KBase(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 normalize=True,
                 filter_empty=False,
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip_p = flip_p
        self.normalize = normalize
        if filter_empty:
            self.filter_empty()

    def filter_empty(self):
        filtered_image_paths = []
        for path in self.image_paths:
            path_ = path[:-4]
            assert 'training' in path_, 'path_ should be training'
            path2 = os.path.join(self.data_root, 'annotations/training', path_.split('/')[-1] + '.png')
            pil_image2 = Image.open(path2)
            label = np.array(pil_image2).astype(np.float32)
            class_ids = sorted(np.unique(label.astype(np.uint8)))
            if class_ids[0] == 0:
                class_ids = class_ids[1:]
            if len(class_ids) > 0:
                filtered_image_paths.append(path)
        self.image_paths = filtered_image_paths
        self._length = len(self.image_paths)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path = self.image_paths[i]
        pil_image = Image.open(path)
        if not pil_image.mode == "RGB":
            pil_image = pil_image.convert("RGB")

        path_ = self.image_paths[i][:-4]
        if 'training' in path_:
            path2 = os.path.join(self.data_root, 'annotations/training', path_.split('/')[-1] + '.png')
        else:
            path2 = os.path.join(self.data_root, 'annotations/validation', path_.split('/')[-1] + '.png')
        pil_image2 = Image.open(path2)

        flip = random.random() < self.flip_p
        if self.size is not None:
            pil_image = pil_image.resize((self.size, self.size), resample=self.interpolation)
            pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        if flip:
            pil_image = pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            pil_image2 = pil_image2.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        pil_image = np.array(pil_image).astype(np.uint8)
        if self.normalize:
            example["image"] = (pil_image / 127.5 - 1.0).astype(np.float32)
        else:
            example["image"] = pil_image
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        if class_ids[0] == 0:
            class_ids = class_ids[1:]
        class_ids_final = np.zeros(151)
        text = ''
        for i in range(len(class_ids)):
            text += _FEATURE_CLASS_NAMES[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final
        example["filename"] = path_.split('/')[-1]

        return example


class ADE20KSampleTextDataset(Dataset):
    def __init__(
        self,
        data_root=os.path.join(os.environ.get("S2RL_DATA_ROOT", "../dataset"), 'ILDistillation/ADE20K'),
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
        for k in ADE20K_dict.keys():
            self.name_dic[int(k)] = ADE20K_dict[k]

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
            if lb == 0:
                continue
            prompt += self.name_dic[lb] + ','
            class_ids.append(lb - 1)

        return dict(prompt=prompt, class_ids=class_ids)


class ADE20KAllSampleTextDataset(Dataset):
    def __init__(
        self,
        data_root=os.path.join(os.environ.get("S2RL_DATA_ROOT", "../dataset"), 'ADE20K'),
        sample_root=os.path.join(os.environ.get("S2RL_DATA_ROOT", "../dataset"), 'ILDistillation/ADE20K'),
        sample_type=None,
    ):
        self.data = []
        # The policy uses the union of the three main-experiment budgets.
        ratios = ('0.0025', '0.005', '0.01')
        files = []
        for ratio in ratios:
            files.extend(glob(os.path.join(sample_root, sample_type, 'images', f"time_*_ratio_{ratio}", "*.jpg")))
        files = sorted(set([os.path.basename(file) for file in files]))
        for fn in files:
            self.data.append({
                'image': os.path.join(data_root, 'images/training', fn),
                'label': os.path.join(data_root, 'annotations/training', fn.replace('.jpg', '.png'))
            })

        self.name_dic = {}
        for k in ADE20K_dict.keys():
            self.name_dic[int(k)] = ADE20K_dict[k]

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
            if lb == 0:
                continue
            prompt += self.name_dic[lb] + ','
            class_ids.append(lb - 1)

        return dict(prompt=prompt, class_ids=class_ids)
