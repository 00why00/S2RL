import os
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, DINOv3ViTModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.ade20k import ADE20KBase

base_dir = os.environ.get("S2RL_FEATS_ROOT", "feats")
pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
processor.size = {"height": 896, "width": 896}
dinov3_model = DINOv3ViTModel.from_pretrained(pretrained_model_name)
dinov3_model.eval().cuda()

train_dataset = ADE20KBase(
    data_root=os.path.join(os.environ.get("S2RL_DATA_ROOT", "../dataset"), 'ADE20K'),
    txt_file=os.path.join(os.environ.get("S2RL_DATA_ROOT", "../dataset"), 'ADE20K/ADE20K_train.txt'),
    size=512, flip_p=0., normalize=False
)
def collect_fn(examples):
    image = [example["image"] for example in examples]
    label = [torch.tensor(example["label"]) for example in examples]
    label = torch.stack(label, dim=0)
    return {"image": image, "label": label}
dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collect_fn)

if __name__ == '__main__':
    mean_feat_dict = [[] for _ in range(150)]
    std_feat_dict = [[] for _ in range(150)]
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            images = [Image.fromarray(img) for img in batch["image"]]
            labels = batch["label"].cuda() - 1
            inputs = processor(images=images, return_tensors="pt").to(dinov3_model.device)
            hidden_states = dinov3_model(**inputs).last_hidden_state[:, 5:, :]  # remove CLS and REG token
            bs, sqr_len, dim = hidden_states.shape
            height = width = int(sqr_len ** 0.5)
            hidden_states = hidden_states.permute(0, 2, 1).reshape(bs, dim, height, width)
            hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=1)
            labels = torch.nn.functional.interpolate(labels.unsqueeze(1).float(), size=(height, width), mode='nearest').squeeze(1).long()

            for i in range(bs):
                cls_ids = labels[i].unique()
                for cls_id in cls_ids:
                    if cls_id.item() == -1:
                        continue
                    mask = (labels[i] == cls_id)
                    cls_feat = hidden_states[i, :, mask]
                    mean_cls_feat = cls_feat.mean(dim=1, keepdim=True).T
                    std_cls_feat = cls_feat.std(dim=1, keepdim=True, unbiased=False).T
                    mean_cls_feat = torch.nn.functional.normalize(mean_cls_feat, p=2, dim=1)
                    mean_feat_dict[cls_id].append(mean_cls_feat)
                    std_feat_dict[cls_id].append(std_cls_feat)

    for cls_id in range(150):
        mean_feat_dict[cls_id] = torch.cat(mean_feat_dict[cls_id], dim=0)
        std_feat_dict[cls_id] = torch.cat(std_feat_dict[cls_id], dim=0)
        save_path = os.path.join(base_dir, "dinov3_ms", f"mean_feats_{cls_id}.pt")
        torch.save(mean_feat_dict[cls_id], save_path)
        save_path = os.path.join(base_dir, "dinov3_ms", f"std_feats_{cls_id}.pt")
        torch.save(std_feat_dict[cls_id], save_path)
