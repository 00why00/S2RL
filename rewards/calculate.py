import os
from collections import Counter, defaultdict

import math
import torch
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, DPTForSemanticSegmentation, \
    SegformerForSemanticSegmentation, DINOv3ViTModel, Mask2FormerForUniversalSegmentation
from transformers.models.eomt import EomtForUniversalSegmentation


_model_cache = {
    "dinov3_processor": AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m"),
    "upernet_convl_processor": AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-large"),
    "segformer_processor": AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640"),
    "dpt_processor": AutoImageProcessor.from_pretrained("Intel/dpt-large-ade"),
    "mask2former_processor": AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic"),
    "eomt_processor": AutoImageProcessor.from_pretrained("tue-mps/coco_panoptic_eomt_large_640"),
    "eomt_1280_processor": AutoImageProcessor.from_pretrained("tue-mps/coco_panoptic_eomt_base_640_2x"),
}
_feat_cache = {}
_hook_cache = {}

def set_device(device):
    if "upernet_convl" in _model_cache and _model_cache["upernet_convl"] is not None:
        _model_cache["upernet_convl"].to(device)
    if "upernet_swinl" in _model_cache and _model_cache["upernet_swinl"] is not None:
        _model_cache["upernet_swinl"].to(device)
    if "segformer" in _model_cache and _model_cache["segformer"] is not None:
        _model_cache["segformer"].to(device)
    if "dpt" in _model_cache and _model_cache["dpt"] is not None:
        _model_cache["dpt"].to(device)
    if "clip" in _model_cache and _model_cache["clip"] is not None:
        _model_cache["clip"].model.to(device)
    if "mask2former" in _model_cache and _model_cache["mask2former"] is not None:
        _model_cache["mask2former"].to(device)
    if "eomt" in _model_cache and _model_cache["eomt"] is not None:
        _model_cache["eomt"].to(device)
    if "eomt_1280" in _model_cache and _model_cache["eomt_1280"] is not None:
        _model_cache["eomt_1280"].to(device)

def free_memory():
    if "upernet_convl" in _model_cache and _model_cache["upernet_convl"] is not None:
        _model_cache["upernet_convl"].to("cpu")
    if "upernet_swinl" in _model_cache and _model_cache["upernet_swinl"] is not None:
        _model_cache["upernet_swinl"].to("cpu")
    if "segformer" in _model_cache and _model_cache["segformer"] is not None:
        _model_cache["segformer"].to("cpu")
    if "dpt" in _model_cache and _model_cache["dpt"] is not None:
        _model_cache["dpt"].to("cpu")
    if "clip" in _model_cache and _model_cache["clip"] is not None:
        _model_cache["clip"].model.to("cpu")
    if "mask2former" in _model_cache and _model_cache["mask2former"] is not None:
        _model_cache["mask2former"].to("cpu")
    if "eomt" in _model_cache and _model_cache["eomt"] is not None:
        _model_cache["eomt"].to("cpu")
    if "eomt_1280" in _model_cache and _model_cache["eomt_1280"] is not None:
        _model_cache["eomt_1280"].to("cpu")

def get_output_hook(name):
    def hook(_module, _input, _output):
        if isinstance(_output, tuple):
            _hook_cache[name] = _output[0].detach()
        else:
            _hook_cache[name] = _output.detach()

    return hook

def get_grad_hook(name):
    def hook(_module, grad_in, grad_out):
        if isinstance(grad_out, tuple):
            assert len(grad_out[0].shape) == 3  # [B, N, C]
            _hook_cache[name] = grad_out[0].mean(dim=[1])  # [B, C]
        else:
            assert len(grad_out.shape) == 3  # [B, N, C]
            _hook_cache[name] = grad_out.mean(dim=[1])  # [B, C]

    return hook

def calculate_align(class_ids, pred_labels, num_classes=150):
    rewards = []
    for b in range(pred_labels.shape[0]):
        pred = pred_labels[b]
        gt_class_ids = class_ids[b]
        if len(gt_class_ids) == 0:
            rewards.append(torch.tensor(0.0, device=pred.device))
            continue

        area_pred_label = torch.histc(pred.float(), bins=num_classes, min=0, max=num_classes - 1)
        area_pred_label = area_pred_label / pred.numel()
        pred_cls_ids = torch.nonzero(area_pred_label > 0, as_tuple=False).squeeze(1)
        if pred_cls_ids.numel() == 0:
            rewards.append(torch.tensor(0.0, device=pred.device))
            continue

        coverage = 0.0
        for cls_id in gt_class_ids:
            if cls_id in pred_cls_ids:
                coverage += 1.0
        coverage_score = coverage / (len(gt_class_ids))
        areas = area_pred_label[torch.Tensor(gt_class_ids).to(dtype=torch.int64)]
        area_score = areas.sum().item()
        reward = 0.7 * coverage_score + 0.3 * area_score

        rewards.append(reward)

    return rewards


def calculate_diversity(feats, pred_labels, margin=0.8):
    batch_rewards = []
    for b in range(pred_labels.shape[0]):
        img_feat = feats[b]
        img_label = pred_labels[b]

        reward_per_class = []
        cls_ids = torch.unique(img_label)
        for cls_id in cls_ids:
            cls_id = cls_id.item()
            label_mask = (img_label == cls_id)
            if label_mask.sum().item() < 2:
                continue
            cls_feat = img_feat[:, label_mask].T
            cls_feat = cls_feat[torch.randperm(cls_feat.shape[0])[:min(100, cls_feat.shape[0])]]

            sim = torch.nn.functional.cosine_similarity(cls_feat.unsqueeze(0), cls_feat.unsqueeze(1), dim=2)
            sim = sim[torch.triu(torch.ones_like(sim), diagonal=1).bool()]
            sim = torch.nn.functional.relu(sim - margin).mean()
            reward_per_class.append(-sim)
        batch_rewards.append(torch.stack(reward_per_class).mean())
    return torch.stack(batch_rewards)


def upernet_align_score(device):
    if "upernet_convl" not in _model_cache or _model_cache["upernet_convl"] is None:
        _model_cache["upernet_convl"] = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-large")
    segmentor_processor = _model_cache["upernet_convl_processor"]
    segmentor_model = _model_cache["upernet_convl"]
    segmentor_model.requires_grad_(False)
    segmentor_model.eval()

    @torch.inference_mode()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        images = segmentor_processor(images=images, return_tensors="pt").to(device, dtype=segmentor_model.dtype, non_blocking=True)
        logits = segmentor_model(**images).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)

        rewards = calculate_align(metadata["class_ids"], pred_labels)
        return rewards, {}

    return _fn

def segformer_b5_align_score(device):
    if "segformer" not in _model_cache or _model_cache["segformer"] is None:
        _model_cache["segformer"] = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    segmentor_processor = _model_cache["segformer_processor"]
    segmentor_model = _model_cache["segformer"]
    segmentor_model.requires_grad_(False)
    segmentor_model.eval()

    @torch.inference_mode()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        images = segmentor_processor(images=images, return_tensors="pt").to(device, dtype=segmentor_model.dtype, non_blocking=True)
        logits = segmentor_model(**images).logits
        logits = torch.nn.functional.interpolate(
            logits, size=(512, 512), mode="bicubic", align_corners=False
        )
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)

        rewards = calculate_align(metadata["class_ids"], pred_labels)
        return rewards, {}

    return _fn

def dpt_align_score(device):
    if "dpt" not in _model_cache or _model_cache["dpt"] is None:
        _model_cache["dpt"] = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")
    segmentor_processor = _model_cache["dpt_processor"]
    segmentor_model = _model_cache["dpt"]
    segmentor_model.requires_grad_(False)
    segmentor_model.eval()

    @torch.inference_mode()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        images = segmentor_processor(images=images, return_tensors="pt").to(device, dtype=segmentor_model.dtype, non_blocking=True)
        logits = segmentor_model(**images).logits
        logits = torch.nn.functional.interpolate(
            logits, size=(512, 512), mode="bicubic", align_corners=False
        )
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)

        rewards = calculate_align(metadata["class_ids"], pred_labels)
        return rewards, {}

    return _fn

def mask2former_align_score(device):
    if "mask2former" not in _model_cache or _model_cache["mask2former"] is None:
        _model_cache["mask2former"] = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
    segmentor_processor = _model_cache["mask2former_processor"]
    segmentor_model = _model_cache["mask2former"]
    segmentor_model.requires_grad_(False)
    segmentor_model.eval()

    @torch.inference_mode()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        images = segmentor_processor(images=images, return_tensors="pt").to(device, dtype=segmentor_model.dtype, non_blocking=True)
        outputs = segmentor_model(**images)
        pred_labels = segmentor_processor.post_process_semantic_segmentation(outputs, target_sizes=[(512, 512)] * len(prompts))
        pred_labels = torch.stack(pred_labels, dim=0)

        rewards = calculate_align(metadata["class_ids"], pred_labels, num_classes=133)
        return rewards, {}

    return _fn

def eomt_align_score(device):
    if "eomt" not in _model_cache or _model_cache["eomt"] is None:
        _model_cache["eomt"] = EomtForUniversalSegmentation.from_pretrained("tue-mps/coco_panoptic_eomt_large_640")
    segmentor_processor = _model_cache["eomt_processor"]
    segmentor_model = _model_cache["eomt"]
    segmentor_model.requires_grad_(False)
    segmentor_model.eval()

    @torch.inference_mode()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        images = segmentor_processor(images=images, return_tensors="pt").to(device, dtype=segmentor_model.dtype, non_blocking=True)
        outputs = segmentor_model(**images)
        pred_labels = segmentor_processor.post_process_semantic_segmentation(outputs, target_sizes=[(512, 512)] * len(prompts))
        pred_labels = torch.stack(pred_labels, dim=0)

        rewards = calculate_align(metadata["class_ids"], pred_labels, num_classes=133)
        return rewards, {}

    return _fn

def eomt_1280_align_score(device):
    if "eomt_1280" not in _model_cache or _model_cache["eomt_1280"] is None:
        _model_cache["eomt_1280"] = EomtForUniversalSegmentation.from_pretrained("tue-mps/coco_panoptic_eomt_base_640_2x")
    segmentor_processor = _model_cache["eomt_1280_processor"]
    segmentor_model = _model_cache["eomt_1280"]
    segmentor_model.requires_grad_(False)
    segmentor_model.eval()

    @torch.inference_mode()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        images = segmentor_processor(images=images, return_tensors="pt").to(device, dtype=segmentor_model.dtype, non_blocking=True)
        outputs = segmentor_model(**images)
        pred_labels = segmentor_processor.post_process_semantic_segmentation(outputs, target_sizes=[(512, 512)] * len(prompts))
        pred_labels = torch.stack(pred_labels, dim=0)

        rewards = calculate_align(metadata["class_ids"], pred_labels, num_classes=133)
        return rewards, {}

    return _fn


def upernet_diversity_score(device):
    if "upernet_convl" not in _model_cache or _model_cache["upernet_convl"] is None:
        _model_cache["upernet_convl"] = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-large")
        _model_cache["upernet_convl"].decode_head.fpn_bottleneck.register_forward_hook(get_output_hook("upernet-div"))
    segmentor_processor = _model_cache["upernet_convl_processor"]
    segmentor_model = _model_cache["upernet_convl"]
    segmentor_model.requires_grad_(False)
    segmentor_model.eval()

    @torch.inference_mode()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        images = segmentor_processor(images=images, return_tensors="pt").to(device, dtype=segmentor_model.dtype, non_blocking=True)
        logits = segmentor_model(**images).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        feats = _hook_cache["upernet-div"]
        bs, ch, hf, wf = feats.shape
        pred_labels = torch.nn.functional.interpolate(pred_labels.unsqueeze(1).float(), size=(hf, wf), mode="nearest").long().squeeze(1)

        rewards = calculate_diversity(feats, pred_labels, margin=0.8)
        return rewards, {}

    return _fn

def segformer_b5_diversity_score(device):
    if "segformer" not in _model_cache or _model_cache["segformer"] is None:
        _model_cache["segformer"] = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
        _model_cache["segformer"].decode_head.dropout.register_forward_hook(get_output_hook("segformer-div"))
    segmentor_processor = _model_cache["segformer_processor"]
    segmentor_model = _model_cache["segformer"]
    segmentor_model.requires_grad_(False)
    segmentor_model.eval()

    @torch.inference_mode()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        images = segmentor_processor(images=images, return_tensors="pt").to(device, dtype=segmentor_model.dtype, non_blocking=True)
        logits = segmentor_model(**images).logits
        logits = torch.nn.functional.interpolate(logits, size=(512, 512), mode="bicubic", align_corners=False)
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        feats = _hook_cache["segformer-div"]
        bs, ch, hf, wf = feats.shape
        pred_labels = torch.nn.functional.interpolate(pred_labels.unsqueeze(1).float(), size=(hf, wf), mode="nearest").long().squeeze(1)

        rewards = calculate_diversity(feats, pred_labels, margin=0.8)
        return rewards, {}

    return _fn

def dpt_diversity_score(device):
    if "dpt" not in _model_cache or _model_cache["dpt"] is None:
        _model_cache["dpt"] = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")
        _model_cache["dpt"].head.head[3].register_forward_hook(get_output_hook("dpt-div"))
    segmentor_processor = _model_cache["dpt_processor"]
    segmentor_model = _model_cache["dpt"]
    segmentor_model.requires_grad_(False)
    segmentor_model.eval()

    @torch.inference_mode()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        images = segmentor_processor(images=images, return_tensors="pt").to(device, dtype=segmentor_model.dtype, non_blocking=True)
        logits = segmentor_model(**images).logits
        logits = torch.nn.functional.interpolate(logits, size=(512, 512), mode="bicubic", align_corners=False)
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        feats = _hook_cache["dpt-div"]
        bs, ch, hf, wf = feats.shape
        pred_labels = torch.nn.functional.interpolate(pred_labels.unsqueeze(1).float(), size=(hf, wf), mode="nearest").long().squeeze(1)

        rewards = calculate_diversity(feats, pred_labels, margin=0.8)
        return rewards, {}

    return _fn

def mask2former_diversity_score(device):
    if "mask2former" not in _model_cache or _model_cache["mask2former"] is None:
        _model_cache["mask2former"] = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
    segmentor_processor = _model_cache["mask2former_processor"]
    segmentor_model = _model_cache["mask2former"]
    segmentor_model.requires_grad_(False)
    segmentor_model.eval()

    @torch.inference_mode()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        images = segmentor_processor(images=images, return_tensors="pt").to(device, dtype=segmentor_model.dtype, non_blocking=True)
        outputs = segmentor_model(**images)
        feats = outputs.encoder_last_hidden_state
        bs, ch, hf, wf = feats.shape
        pred_labels = segmentor_processor.post_process_semantic_segmentation(outputs, target_sizes=[(hf, wf)] * len(prompts))
        pred_labels = torch.stack(pred_labels, dim=0)

        rewards = calculate_diversity(feats, pred_labels)
        return rewards, {}

    return _fn

def eomt_diversity_score(device):
    if "eomt" not in _model_cache or _model_cache["eomt"] is None:
        _model_cache["eomt"] = EomtForUniversalSegmentation.from_pretrained("tue-mps/coco_panoptic_eomt_large_640")
    segmentor_processor = _model_cache["eomt_processor"]
    segmentor_model = _model_cache["eomt"]
    segmentor_model.requires_grad_(False)
    segmentor_model.eval()

    @torch.inference_mode()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        images = segmentor_processor(images=images, return_tensors="pt").to(device, dtype=segmentor_model.dtype, non_blocking=True)
        outputs = segmentor_model(**images)
        feats = outputs.last_hidden_state[:, 205:, :]  # remove CLS and REG token
        bs, sqr_len, dim = feats.shape
        hf = wf = int(sqr_len ** 0.5)
        feats = feats.permute(0, 2, 1).reshape(bs, dim, hf, wf)
        pred_labels = segmentor_processor.post_process_semantic_segmentation(outputs, target_sizes=[(hf, wf)] * len(prompts))
        pred_labels = torch.stack(pred_labels, dim=0)

        rewards = calculate_diversity(feats, pred_labels)
        return rewards, {}

    return _fn

def eomt_1280_diversity_score(device):
    if "eomt_1280" not in _model_cache or _model_cache["eomt_1280"] is None:
        _model_cache["eomt_1280"] = EomtForUniversalSegmentation.from_pretrained("tue-mps/coco_panoptic_eomt_base_640_2x")
    segmentor_processor = _model_cache["eomt_1280_processor"]
    segmentor_model = _model_cache["eomt_1280"]
    segmentor_model.requires_grad_(False)
    segmentor_model.eval()

    @torch.inference_mode()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]
        images = segmentor_processor(images=images, return_tensors="pt").to(device, dtype=segmentor_model.dtype, non_blocking=True)
        outputs = segmentor_model(**images)
        feats = outputs.last_hidden_state[:, 205:, :]  # remove CLS and REG token
        bs, sqr_len, dim = feats.shape
        hf = wf = int(sqr_len ** 0.5)
        feats = feats.permute(0, 2, 1).reshape(bs, dim, hf, wf)
        pred_labels = segmentor_processor.post_process_semantic_segmentation(outputs, target_sizes=[(hf, wf)] * len(prompts))
        pred_labels = torch.stack(pred_labels, dim=0)

        rewards = calculate_diversity(feats, pred_labels)
        return rewards, {}

    return _fn


def calculate(device, score_dict):
    score_functions = {
        "upernet_za": upernet_align_score,
        "segformer_b5_za": segformer_b5_align_score,
        "dpt_za": dpt_align_score,
        "mask2former_za": mask2former_align_score,
        "eomt_za": eomt_align_score,
        "eomt_1280_za": eomt_1280_align_score,
        "upernet_intra_div": upernet_diversity_score,
        "segformer_b5_intra_div": segformer_b5_diversity_score,
        "dpt_intra_div": dpt_diversity_score,
        "mask2former_intra_div": mask2former_diversity_score,
        "eomt_intra_div": eomt_diversity_score,
        "eomt_1280_intra_div": eomt_1280_diversity_score,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device)

    def _fn(images, prompts, metadata):
        total_scores = []
        score_details = {}

        for _score_name, _weight in score_dict.items():
            scores, rewards = score_fns[_score_name](images, prompts, metadata)
            score_details[_score_name] = scores
            weighted_scores = [_weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details['avg'] = total_scores
        return score_details, {}

    return _fn


def semantic_ucb_coverage_score(device):
    from dataset.ade20k import ADE20K_dict

    name2id = {name: int(cid) - 1 for cid, name in ADE20K_dict.items()}
    if "dinov3" not in _model_cache or _model_cache["dinov3"] is None:
        _model_cache["dinov3"] = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
    dinov3_processor = _model_cache["dinov3_processor"]
    dinov3_processor.size = {"height": 896, "width": 896}
    dinov3_model = _model_cache["dinov3"]
    dinov3_model.eval()

    if "upernet_convl" not in _model_cache or _model_cache["upernet_convl"] is None:
        _model_cache["upernet_convl"] = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-large")
    upernet_processor = _model_cache["upernet_convl_processor"]
    upernet_model = _model_cache["upernet_convl"]
    upernet_model.eval()

    if "segformer" not in _model_cache or _model_cache["segformer"] is None:
        _model_cache["segformer"] = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    segformer_processor = _model_cache["segformer_processor"]
    segformer_model = _model_cache["segformer"]
    segformer_model.eval()

    if "dpt" not in _model_cache or _model_cache["dpt"] is None:
        _model_cache["dpt"] = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")
    dpt_processor = _model_cache["dpt_processor"]
    dpt_model = _model_cache["dpt"]
    dpt_model.eval()

    if "cluster_stats" not in _feat_cache:
        _feat_cache["cluster_stats"] = {}

    alpha_balance = 1.0
    beta_explore = 0.5

    ref_cls_feats = {}
    for i in range(150):
        ref_cls_feat = os.path.join(os.path.join(os.environ.get("S2RL_FEATS_ROOT", "feats"), 'dinov3_kmeans'), f"feats_{i}.pt")
        ref_cls_feats[i] = torch.load(ref_cls_feat).to(device)
        _feat_cache["cluster_stats"][i] = {
            "hard_counts": torch.zeros(ref_cls_feats[i].shape[0], device=device),
            "soft_counts": torch.zeros(ref_cls_feats[i].shape[0], device=device),
        }

    def _fn(images, prompts, metadata):
        dinov3_model.to(device)

        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]

        is_eval = True if len(images) % (8 * 16) != 0 else False

        batch_size = 1
        img2cls = []
        with torch.inference_mode():
            for jj in range(0, len(images), batch_size):
                batch_images = images[jj:jj + batch_size]

                batch_inputs = upernet_processor(images=batch_images, return_tensors="pt").to(device, dtype=upernet_model.dtype)
                batch_logits = upernet_model(**batch_inputs).logits
                batch_logits = torch.nn.functional.interpolate(batch_logits, size=(512, 512), mode="bicubic", align_corners=False)
                batch_probs = torch.nn.functional.softmax(batch_logits, dim=1)
                batch_upernet_pred_labels = torch.argmax(batch_probs, dim=1)

                batch_inputs = segformer_processor(images=batch_images, return_tensors="pt").to(device, dtype=segformer_model.dtype)
                batch_logits = segformer_model(**batch_inputs).logits
                batch_logits = torch.nn.functional.interpolate(batch_logits, size=(512, 512), mode="bicubic", align_corners=False)
                batch_probs = torch.nn.functional.softmax(batch_logits, dim=1)
                batch_segformer_pred_labels = torch.argmax(batch_probs, dim=1)

                batch_inputs = dpt_processor(images=batch_images, return_tensors="pt").to(device, dtype=dpt_model.dtype)
                batch_logits = dpt_model(**batch_inputs).logits
                batch_logits = torch.nn.functional.interpolate(batch_logits, size=(512, 512), mode="bicubic", align_corners=False)
                batch_probs = torch.nn.functional.softmax(batch_logits, dim=1)
                batch_dpt_pred_labels = torch.argmax(batch_probs, dim=1)

                votes = torch.stack([batch_upernet_pred_labels, batch_segformer_pred_labels, batch_dpt_pred_labels], dim=1)
                batch_pred_labels, _ = torch.mode(votes, dim=1)
                agreement = (votes == batch_pred_labels.unsqueeze(1)).long().sum(dim=1)
                batch_pred_labels[agreement < 2] = batch_upernet_pred_labels[agreement < 2]

                batch_inputs = dinov3_processor(images=batch_images, return_tensors="pt").to(device, dtype=dinov3_model.dtype)
                batch_feats = dinov3_model(**batch_inputs).last_hidden_state[:, 5:, :]
                bs, seq_len, dim = batch_feats.shape
                height = weight = int(seq_len ** 0.5)
                batch_feats = batch_feats.permute(0, 2, 1).view(bs, dim, height, weight)
                batch_pred_labels = torch.nn.functional.interpolate(batch_pred_labels.unsqueeze(1).float(), size=(height, weight), mode="nearest").long().squeeze(1)
                batch_feats = torch.nn.functional.normalize(batch_feats, p=2, dim=1)

                for b in range(batch_size):
                    prompt = prompts[jj * batch_size + b]
                    img_feat = batch_feats[b]
                    img_label = batch_pred_labels[b]
                    cls_names = prompt.split(",")[:-1]
                    cls_ids = [name2id[name.strip()] for name in cls_names]
                    cls_feats = {}
                    for cls_id in cls_ids:
                        label_mask = (img_label == cls_id)
                        if label_mask.sum() == 0:
                            continue
                        cls_feat = img_feat[:, label_mask].mean(dim=1, keepdim=True).T
                        cls_feat = torch.nn.functional.normalize(cls_feat, p=2, dim=1)
                        cls_feats[cls_id] = cls_feat
                    img2cls.append(cls_feats)
        dinov3_model.to("cpu")

        final_reward = [0.0] * len(images)
        cluster_stats = _feat_cache["cluster_stats"]
        img_matches = {}
        batch_hard_counts = defaultdict(Counter)
        batch_soft_counts = defaultdict(lambda : defaultdict(float))
        for idx, cls_feats in enumerate(img2cls):
            img_matches[idx] = []
            for cls_id, gen_feat in cls_feats.items():
                ref_feat = ref_cls_feats[cls_id]
                sim = torch.mm(gen_feat, ref_feat.t())
                best_sim, best_idx = sim.max(dim=1)
                best_sim, best_idx = max(0.0, best_sim.item()), best_idx.item()
                img_matches[idx].append((cls_id, best_sim, best_idx))
                batch_hard_counts[cls_id][best_idx] += 1
                batch_soft_counts[cls_id][best_idx] += best_sim

        for idx, matches in img_matches.items():
            img_scores = []
            obj_count_in_prompt = max(len(prompts[idx].split(",")[:-1]), 1)
            for (cls_id, best_sim, best_idx) in matches:
                current_batch_count = batch_soft_counts[cls_id][best_idx]
                history_count = cluster_stats[cls_id]["soft_counts"][best_idx]
                total_count = (history_count + current_batch_count) if not is_eval else current_batch_count

                history_cls_count = cluster_stats[cls_id]["soft_counts"].sum().item()
                batch_cls_count = sum(batch_soft_counts[cls_id].values())
                total_cls_count = (history_cls_count + batch_cls_count) if not is_eval else batch_cls_count
                avg_cls_count = total_cls_count / ref_cls_feats[cls_id].shape[0]

                scarcity_factor = ((avg_cls_count + 1.0) / (total_count + 1.0)) ** alpha_balance
                exploration_bonus = beta_explore * math.sqrt(math.log(max(total_cls_count, 1)) / (1.0 + total_count))
                reward = (best_sim * scarcity_factor) + exploration_bonus
                img_scores.append(reward)

            if img_scores:
                final_reward[idx] = sum(img_scores) / obj_count_in_prompt
            else:
                final_reward[idx] = 0.0

        if not is_eval:
            for cls_id, proto_dict in batch_hard_counts.items():
                for p_id, count in proto_dict.items():
                    cluster_stats[cls_id]["hard_counts"][p_id] += count
                    cluster_stats[cls_id]["soft_counts"][p_id] += batch_soft_counts[cls_id][p_id]
        else:
            print("EVAL semantic UCB coverage score, not updating cluster stats.")

        return final_reward, {}

    return _fn

def coco_semantic_ucb_coverage_score(device):
    from dataset.coco import COCO_dict

    name2id = {name: int(cid) for cid, name in COCO_dict.items()}
    if "dinov3" not in _model_cache or _model_cache["dinov3"] is None:
        _model_cache["dinov3"] = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
    dinov3_processor = _model_cache["dinov3_processor"]
    dinov3_processor.size = {"height": 896, "width": 896}
    dinov3_model = _model_cache["dinov3"]
    dinov3_model.eval()

    if "mask2former" not in _model_cache or _model_cache["mask2former"] is None:
        _model_cache["mask2former"] = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
    mask2former_processor = _model_cache["mask2former_processor"]
    mask2former_model = _model_cache["mask2former"]
    mask2former_model.eval()

    if "eomt" not in _model_cache or _model_cache["eomt"] is None:
        _model_cache["eomt"] = EomtForUniversalSegmentation.from_pretrained("tue-mps/coco_panoptic_eomt_large_640")
    eomt_processor = _model_cache["eomt_processor"]
    eomt_model = _model_cache["eomt"]
    eomt_model.eval()

    if "eomt_1280" not in _model_cache or _model_cache["eomt_1280"] is None:
        _model_cache["eomt_1280"] = EomtForUniversalSegmentation.from_pretrained("tue-mps/coco_panoptic_eomt_base_640_2x")
    eomt_1280_processor = _model_cache["eomt_1280_processor"]
    eomt_1280_model = _model_cache["eomt_1280"]
    eomt_1280_model.eval()

    if "cluster_stats" not in _feat_cache:
        _feat_cache["cluster_stats"] = {}

    alpha_balance = 1.0
    beta_explore = 0.5

    ref_cls_feats = {}
    for i in range(133):
        ref_cls_feat = os.path.join(os.path.join(os.environ.get("S2RL_FEATS_ROOT", "feats"), 'dinov3_coco_kmeans'), f"feats_{i}.pt")
        ref_cls_feats[i] = torch.load(ref_cls_feat).to(device)
        _feat_cache["cluster_stats"][i] = {
            "hard_counts": torch.zeros(ref_cls_feats[i].shape[0], device=device),
            "soft_counts": torch.zeros(ref_cls_feats[i].shape[0], device=device),
        }

    def _fn(images, prompts, metadata):
        dinov3_model.to(device)

        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            images = [Image.fromarray(image) for image in images]

        is_eval = True if len(images) % (8 * 16) != 0 else False

        batch_size = 1
        img2cls = []
        with torch.inference_mode():
            for jj in range(0, len(images), batch_size):
                batch_images = images[jj:jj + batch_size]

                batch_inputs = dinov3_processor(images=batch_images, return_tensors="pt").to(device, dtype=dinov3_model.dtype)
                batch_feats = dinov3_model(**batch_inputs).last_hidden_state[:, 5:, :]
                bs, seq_len, dim = batch_feats.shape
                height = weight = int(seq_len ** 0.5)
                batch_feats = batch_feats.permute(0, 2, 1).view(bs, dim, height, weight)
                batch_feats = torch.nn.functional.normalize(batch_feats, p=2, dim=1)

                batch_inputs = mask2former_processor(images=batch_images, return_tensors="pt").to(device, dtype=mask2former_model.dtype)
                batch_outputs = mask2former_model(**batch_inputs)
                batch_pred_labels = mask2former_processor.post_process_semantic_segmentation(batch_outputs, target_sizes=[(height, weight)] * len(batch_images))
                batch_mask2former_pred_labels = torch.stack(batch_pred_labels, dim=0)

                batch_inputs = eomt_processor(images=batch_images, return_tensors="pt").to(device, dtype=eomt_model.dtype)
                batch_outputs = eomt_model(**batch_inputs)
                batch_pred_labels = eomt_processor.post_process_semantic_segmentation(batch_outputs, target_sizes=[(height, weight)] * len(batch_images))
                batch_eomt_pred_labels = torch.stack(batch_pred_labels, dim=0)

                batch_inputs = eomt_1280_processor(images=batch_images, return_tensors="pt").to(device, dtype=eomt_1280_model.dtype)
                batch_outputs = eomt_1280_model(**batch_inputs)
                batch_pred_labels = eomt_1280_processor.post_process_semantic_segmentation(batch_outputs, target_sizes=[(height, weight)] * len(batch_images))
                batch_eomt_1280_pred_labels = torch.stack(batch_pred_labels, dim=0)

                votes = torch.stack([batch_mask2former_pred_labels, batch_eomt_pred_labels, batch_eomt_1280_pred_labels], dim=1)
                batch_pred_labels, _ = torch.mode(votes, dim=1)
                agreement = (votes == batch_pred_labels.unsqueeze(1)).long().sum(dim=1)
                batch_pred_labels[agreement < 2] = batch_eomt_1280_pred_labels[agreement < 2]

                for b in range(batch_size):
                    prompt = prompts[jj * batch_size + b]
                    img_feat = batch_feats[b]
                    img_label = batch_pred_labels[b]
                    cls_names = prompt.split(",")[:-1]
                    cls_ids = [name2id[name.strip().replace(' - ', '-')] for name in cls_names]
                    cls_feats = {}
                    for cls_id in cls_ids:
                        label_mask = (img_label == cls_id)
                        if label_mask.sum() == 0:
                            continue
                        cls_feat = img_feat[:, label_mask].mean(dim=1, keepdim=True).T
                        cls_feat = torch.nn.functional.normalize(cls_feat, p=2, dim=1)
                        cls_feats[cls_id] = cls_feat
                    img2cls.append(cls_feats)
        dinov3_model.to("cpu")

        final_reward = [0.0] * len(images)
        cluster_stats = _feat_cache["cluster_stats"]
        img_matches = {}
        batch_hard_counts = defaultdict(Counter)
        batch_soft_counts = defaultdict(lambda : defaultdict(float))
        for idx, cls_feats in enumerate(img2cls):
            img_matches[idx] = []
            for cls_id, gen_feat in cls_feats.items():
                ref_feat = ref_cls_feats[cls_id]
                sim = torch.mm(gen_feat, ref_feat.t())
                best_sim, best_idx = sim.max(dim=1)
                best_sim, best_idx = max(0.0, best_sim.item()), best_idx.item()
                img_matches[idx].append((cls_id, best_sim, best_idx))
                batch_hard_counts[cls_id][best_idx] += 1
                batch_soft_counts[cls_id][best_idx] += best_sim

        for idx, matches in img_matches.items():
            img_scores = []
            obj_count_in_prompt = max(len(prompts[idx].split(",")[:-1]), 1)
            for (cls_id, best_sim, best_idx) in matches:
                current_batch_count = batch_soft_counts[cls_id][best_idx]
                history_count = cluster_stats[cls_id]["soft_counts"][best_idx]
                total_count = (history_count + current_batch_count) if not is_eval else current_batch_count

                history_cls_count = cluster_stats[cls_id]["soft_counts"].sum().item()
                batch_cls_count = sum(batch_soft_counts[cls_id].values())
                total_cls_count = (history_cls_count + batch_cls_count) if not is_eval else batch_cls_count
                avg_cls_count = total_cls_count / ref_cls_feats[cls_id].shape[0]

                scarcity_factor = ((avg_cls_count + 1.0) / (total_count + 1.0)) ** alpha_balance
                exploration_bonus = beta_explore * math.sqrt(math.log(max(total_cls_count, 1)) / (1.0 + total_count))
                reward = (best_sim * scarcity_factor) + exploration_bonus
                img_scores.append(reward)

            if img_scores:
                final_reward[idx] = sum(img_scores) / obj_count_in_prompt
            else:
                final_reward[idx] = 0.0

        if not is_eval:
            for cls_id, proto_dict in batch_hard_counts.items():
                for p_id, count in proto_dict.items():
                    cluster_stats[cls_id]["hard_counts"][p_id] += count
                    cluster_stats[cls_id]["soft_counts"][p_id] += batch_soft_counts[cls_id][p_id]
        else:
            print("EVAL semantic UCB coverage score, not updating cluster stats.")

        return final_reward, {}

    return _fn


def calculate_set(device, score_dict):
    score_functions = {
        "semantic_ucb_coverage": semantic_ucb_coverage_score,
        "coco_semantic_ucb_coverage": coco_semantic_ucb_coverage_score,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device)

    def _fn(images, prompts, metadata):
        total_scores = []
        score_details = {}

        for _score_name, _weight in score_dict.items():
            scores, reward_metadata = score_fns[_score_name](images, prompts, metadata)
            score_details[_score_name] = scores
            weighted_scores = [_weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details['avg'] = total_scores
        return score_details, reward_metadata

    return _fn
