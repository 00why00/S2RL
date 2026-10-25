import glob
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, DINOv3ViTModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.coco import COCODataset

base_dir = os.environ.get("S2RL_FEATS_ROOT", "feats")
pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
processor.size = {"height": 896, "width": 896}
dinov3_model = DINOv3ViTModel.from_pretrained(pretrained_model_name)
dinov3_model.eval().cuda()

train_dataset = COCODataset( data_root=os.path.join(os.environ.get("S2RL_DATA_ROOT", "../dataset"), 'COCO'), phase="train")
def collect_fn(examples):
    image = [example["image"] for example in examples]
    label = [torch.tensor(example["label"]) for example in examples]
    label = torch.stack(label, dim=0)
    return {"image": image, "label": label}
dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collect_fn)


def save_chunk(buffer_mean, chunk_id):
    chunk_save_path = os.path.join(base_dir, "dinov3_coco", f"chunk_{chunk_id}.pt")
    torch.save(buffer_mean, chunk_save_path)
    print(f"  [Info] Saved chunk {chunk_id} to {chunk_save_path}, clearing memory...")

if __name__ == '__main__':
    mean_feat_dict = [[] for _ in range(133)]
    chunk_counter = 0
    with torch.inference_mode():
        for idx, batch in enumerate(tqdm(dataloader)):
            labels = batch["label"].cuda()
            inputs = processor(images=batch["image"], return_tensors="pt").to(dinov3_model.device)
            hidden_states = dinov3_model(**inputs).last_hidden_state[:, 5:, :]  # remove CLS and REG token
            bs, sqr_len, dim = hidden_states.shape
            height = width = int(sqr_len ** 0.5)
            hidden_states = hidden_states.permute(0, 2, 1).reshape(bs, dim, height, width)
            hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=1)
            labels = torch.nn.functional.interpolate(labels.unsqueeze(1).float(), size=(height, width), mode='nearest').squeeze(1).long()

            for i in range(bs):
                cls_ids = labels[i].unique()
                for cls_id in cls_ids:
                    if cls_id.item() == 255:
                        continue
                    mask = (labels[i] == cls_id)
                    cls_feat = hidden_states[i, :, mask]
                    mean_cls_feat = cls_feat.mean(dim=1, keepdim=True).T
                    mean_cls_feat = torch.nn.functional.normalize(mean_cls_feat, p=2, dim=1)
                    mean_feat_dict[cls_id].append(mean_cls_feat)

            if (idx + 1) % 3000 == 0:
                save_chunk(mean_feat_dict, chunk_counter)
                mean_feat_dict = [[] for _ in range(133)]
                chunk_counter += 1

        save_chunk(mean_feat_dict, chunk_counter)

    final_aggregation = [[] for _ in range(133)]
    chunk_files = sorted(glob.glob(os.path.join(base_dir, "dinov3_coco", "chunk_*.pt")))
    for c_file in tqdm(chunk_files, desc="Loading Chunks"):
        chunk_data = torch.load(c_file)

        for cls_id in range(133):
            final_aggregation[cls_id].extend(chunk_data[cls_id])

        del chunk_data  # free memory

    for cls_id in range(133):
        all_means = torch.cat(final_aggregation[cls_id], dim=0)
        save_path = os.path.join(base_dir, "dinov3_coco", f"mean_feats_{cls_id}.pt")
        torch.save(all_means, save_path)
