import argparse
import os.path
import shutil
import sys
from collections import defaultdict

import torch
from PIL import Image
from accelerate import PartialState
from accelerate.utils import gather_object
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.coco import COCOSampleTextDataset, COCO_dict
from rewards.calculate import calculate, set_device, calculate_set


def collect_fn(examples):
    prompts = [example["prompt"] for example in examples]
    class_ids = [example["class_ids"] for example in examples]
    return prompts, {"class_ids": class_ids}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='COCO', choices=['COCO'])
parser.add_argument("--finetune_model_path", required=True, type=str)
parser.add_argument("--sample_type", type=str, default='uniform')
parser.add_argument("--guidance_scale", type=float, default=7.0)
parser.add_argument("--selected_images", type=str, default="time_1_ratio_0.000625")
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
print(args)

distributed_state = PartialState()
if args.dataset == "ADE20K":
    raise NotImplementedError
elif args.dataset == "COCO":
    sample_dataset = COCOSampleTextDataset(sample_type=args.sample_type, selected_images=args.selected_images)
    dataloader = DataLoader(sample_dataset, batch_size=1, shuffle=False, collate_fn=collect_fn)
    name2id = {name: int(cid) for cid, name in COCO_dict.items()}
    prompt_list = []
    for idx, (prompt, prompt_metadata) in enumerate(dataloader):
        prompt_list.append([idx, prompt])
    image_save_path = os.path.join(os.path.join(os.environ.get("S2RL_DATA_ROOT", "../dataset"), 'ILDistillation/COCO'), f"{args.save_path}", f"images/{args.selected_images}")
    temp_save_path = os.path.join(os.path.join(os.environ.get("S2RL_DATA_ROOT", "../dataset"), 'ILDistillation/COCO'), "TEMP")
    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(temp_save_path, exist_ok=True)
else:
    raise ValueError

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16,
)
pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.finetune_model_path)
pipe.transformer = pipe.transformer.merge_and_unload()

proto_counts = {}
pipe.set_progress_bar_config(disable=True)
pipe = pipe.to(distributed_state.device)
generator = torch.Generator(device=distributed_state.device).manual_seed(args.seed + distributed_state.process_index)
sample_reward_fn = calculate(distributed_state.device,{"mask2former_intra_div": 10.0, "eomt_intra_div": 10.0, "eomt_1280_intra_div": 10.0, "mask2former_za": 1.0, "eomt_za": 1.0,  "eomt_1280_za": 1.0})
set_reward_fn = calculate_set(distributed_state.device, {"coco_semantic_ucb_coverage": 10.0})
set_device(distributed_state.device)

print(">>> Phase 1: Generating All Candidates...")
local_metadata = []
with distributed_state.split_between_processes(prompt_list) as split_prompt:
    with torch.inference_mode():
        for idx, prompt in tqdm(split_prompt):
            cls_names = prompt[0].split(",")[:-1]
            cls_ids = [name2id[name.strip()] for name in cls_names]
            batch_images = pipe(
                prompt,
                height=1024,
                width=1024,
                num_inference_steps=50,
                guidance_scale=args.guidance_scale,
                generator=generator,
                num_images_per_prompt=16
            ).images

            sample_reward = sample_reward_fn(batch_images, prompt * len(batch_images), {"class_ids": [cls_ids] * len(batch_images)})[0]['avg']
            for i in range(len(batch_images)):
                save_path = os.path.join(temp_save_path, f"p{idx}_b{i}.jpg")
                batch_images[i].save(save_path)

                local_metadata.append({
                    "prompt_idx": idx,
                    "batch_idx": i,
                    "path": save_path,
                    "prompt": prompt[0],
                    "sample_reward": sample_reward[i].item(),
                })
distributed_state.wait_for_everyone()
all_results = gather_object(local_metadata)

if distributed_state.is_main_process:
    print(f">>> Phase 2: Calculating Global Rewards for {len(all_results)} images...")
    assert len(all_results) == len(prompt_list) * 16

    global_images, global_prompts = [], []
    print("Loading images from disk...")
    for res in tqdm(all_results):
        img = Image.open(res['path']).convert("RGB")
        global_images.append(img)
        global_prompts.append(res['prompt'])

    with torch.inference_mode():
        set_reward = set_reward_fn(global_images, global_prompts, None)[0]['avg']

    print(">>> Phase 3: Selecting Winners...")
    candidates_by_prompt = defaultdict(list)
    for i in range(len(all_results)):
        print(f"Image {i}: Sample Reward: {all_results[i]['sample_reward']:.4f}, Set Reward: {set_reward[i]:.4f}")
        total_score = all_results[i]['sample_reward'] + set_reward[i]
        item = {
            "path": all_results[i]['path'],
            "total_score": total_score
        }
        candidates_by_prompt[all_results[i]['prompt_idx']].append(item)

    for p_idx, candidates in candidates_by_prompt.items():
        candidates.sort(key=lambda x: x['total_score'], reverse=True)
        best_image_path = candidates[0]['path']
        shutil.copy(best_image_path, os.path.join(image_save_path, f"{p_idx}.jpg"))
        print(f"Prompt Idx: {p_idx}, Best Total Score: {candidates[0]['total_score']:.4f}")

    shutil.rmtree(temp_save_path)
    print("Done!")
