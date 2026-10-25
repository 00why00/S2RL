import contextlib
import datetime
import hashlib
import os
import random
import tempfile
import time
from collections import defaultdict
from concurrent import futures
from functools import partial
from typing import Optional, Union, List, Dict, Any

import math
import numpy as np
import swanlab
import torch
import tqdm
from PIL import Image
from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration, broadcast_object_list
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor, is_compiled_module
from ml_collections import config_flags
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader, Sampler

from dataset.ade20k import ADE20KSampleTextDataset, ADE20KAllSampleTextDataset
from dataset.coco import COCOAllSampleTextDataset, COCOSampleTextDataset
from pipelines.ema import EMAModuleWrapper
from rewards import calculate
from rewards.calculate import free_memory, set_device
from utils.stat_tracking import PerPromptStatTracker, calculate_zero_std_ratio

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base_sd3.py", "Training configuration.")

logger = get_logger(__name__)

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k  # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank  # Current replica rank
        self.seed = seed  # Random seed for synchronization

        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])

            # Return current replica's sample indices
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2 ** 31)  # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

def create_set_generator(per_prompts, base_seed):
    generators = []
    for idx in range(per_prompts):
        seed = base_seed + idx
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_layer_guidance_scale: float = 2.8,
    noise_level: float = 0.7,
    return_prev_sample_mean: bool = False
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    ).float()

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []
    all_prev_latents_mean = []

    # 7. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            # noise_pred = noise_pred.to(prompt_embeds.dtype)
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                t.unsqueeze(0),
                latents.float(),
                noise_level=noise_level,
            )

            all_latents.append(latents)
            all_log_probs.append(log_prob)
            all_prev_latents_mean.append(prev_latents_mean)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if return_prev_sample_mean:
        return image, all_latents, all_log_probs, all_prev_latents_mean
    return image, all_latents, all_log_probs


def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: Optional[str] = 'sde',
    return_sqrt_dt: Optional[bool] = False,
):
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma

    if sde_type == 'sde':
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level

        # our sde
        prev_sample_mean = sample * (1 + std_dev_t ** 2 / (2 * sigma) * dt) + model_output * (1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
            - torch.log(std_dev_t * torch.sqrt(-1 * dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

    elif sde_type == 'cps':
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)  # sigma_t in paper
        pred_original_sample = sample - sigma * model_output  # predicted x_0 in paper
        noise_estimate = sample + model_output * (1 - sigma)  # predicted x_1 in paper
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev ** 2 - std_dev_t ** 2)

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        # remove all constants
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_sqrt_dt:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1 * dt)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]))
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
                noise_pred_uncond
                + config.sample.guidance_scale
                * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    # compute the log prob of next_latents given latents under the current model
    return sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
        return_sqrt_dt=config.rationorm
    )

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, set_reward_fn,
         executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([config.sample.negative_prompt], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    image_list, prompt_list = [], []
    for test_batch in tqdm(
        test_dataloader,
        desc="Eval: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)
        # The last batch may not be full batch_size
        if len(prompt_embeds) < len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
        # yield to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)

        images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device))
        image_list.append(images_gather)
        prompt_ids = tokenizers[0](prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
        prompt_ids_gather = accelerator.gather(prompt_ids).cpu().numpy()
        prompts_gather = pipeline.tokenizer.batch_decode(prompt_ids_gather, skip_special_tokens=True)
        prompt_list.extend(prompts_gather)
    image_list = torch.cat(image_list, dim=0)
    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}

    reward_container = [None]
    if accelerator.is_main_process:
        set_rewards = set_reward_fn(image_list, prompt_list, metadata=None)[0]
        for key, value in set_rewards.items():
            set_rewards[key] = torch.as_tensor(value, device=accelerator.device).cpu()
        reward_container[0] = set_rewards
    set_rewards = broadcast_object_list(reward_container, from_process=0)[0]
    for key, value in set_rewards.items():
        all_rewards[key + '_set'] = value.numpy()

    last_batch_rewards_gather = {}
    for key, value in all_rewards.items():
        last_batch_rewards_gather[key] = torch.as_tensor(value[-len(prompts_gather):], device=accelerator.device).cpu().numpy()  # noqa

    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            images_gather = images_gather.cpu().numpy()  # noqa
            num_samples = min(15, len(images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [prompts_gather[index] for index in sample_indices]  # noqa
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            accelerator.log({
                "eval_images": [
                    swanlab.Image(
                        os.path.join(tmpdir, f"{idx}.jpg"),
                        caption=f"{prompt:.1000} | " + " | ".join(
                            f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                    )
                    for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                ],
                **{f"eval/eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
            }, step=global_step)
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def collect_fn(examples):
    prompts = [example["prompt"] for example in examples]
    class_ids = [example["class_ids"] for example in examples]
    return prompts, {"class_ids": class_ids}

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="swanlab",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="sd3-set-grpo",
            config=config.to_dict(),
            init_kwargs={"swanlab": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)  # noqa

    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20×8=160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    reward_fn = getattr(calculate, 'calculate')(accelerator.device, config.reward_fn)
    set_reward_fn = getattr(calculate, 'calculate_set')(accelerator.device, config.set_reward_fn)
    if config.dataset_type != "all_sample" or config.sample_type != "uniform":
        raise ValueError("The main experiment requires all_sample prompts from Uniform subsets.")
    if config.dataset == "ade20k":
        train_dataset = ADE20KAllSampleTextDataset(sample_type=config.sample_type)
        test_dataset = ADE20KSampleTextDataset(sample_type=config.sample_type, selected_images=config.selected_images)
    elif config.dataset == "coco":
        train_dataset = COCOAllSampleTextDataset(sample_type=config.sample_type)
        test_dataset = COCOSampleTextDataset(sample_type=config.sample_type, selected_images=config.selected_images)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # Create an infinite-loop DataLoader
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42
    )

    # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=collect_fn,
    )

    # Create a regular DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=collect_fn,
        shuffle=False,
        num_workers=8,
    )

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([config.sample.negative_prompt], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)  # noqa

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=1)

    # Train!
    samples_per_epoch = (config.sample.train_batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch)
    total_train_batch_size = (config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps)
    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        set_device(accelerator.device)
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, set_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        image_list, prompt_list, prompt_metadata_list = [], [], []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)
            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)
            prompt_ids = tokenizers[0](prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch * 10000 + i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    if config.rationorm:
                        images, latents, log_probs, prev_sample_mean = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            generator=generator,
                            return_prev_sample_mean=True
                        )
                    else:
                        images, latents, log_probs = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            generator=generator
                        )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
            if config.rationorm:
                prev_sample_mean = torch.stack(prev_sample_mean, dim=1)
            else:
                prev_sample_mean = log_probs

            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.train_batch_size, 1)  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
            # yield to make sure reward computation starts
            time.sleep(0)

            samples.append({
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "timesteps": timesteps,
                "latents": latents[:, :-1],  # each entry is the latent before timestep t
                "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                "prev_sample_mean": prev_sample_mean,
                "log_probs": log_probs,
                "rewards": rewards,
                "set_rewards": {},
            })

            images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device))
            image_list.append(images_gather)
            prompt_ids = tokenizers[0](prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
            prompt_ids_gather = accelerator.gather(prompt_ids).cpu().numpy()
            prompts_gather = pipeline.tokenizer.batch_decode(prompt_ids_gather, skip_special_tokens=True)
            prompt_list.extend(prompts_gather)
        image_list = torch.cat(image_list, dim=0)

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        reward_container = [None]
        if accelerator.is_main_process:
            set_rewards, reward_metadata = set_reward_fn(image_list, prompt_list, metadata=None)
            set_rewards = {key: torch.as_tensor(value, device=accelerator.device).float().cpu() for key, value in set_rewards.items()}
            reward_container[0] = set_rewards

        free_memory()
        set_rewards = broadcast_object_list(reward_container, from_process=0)[0]
        set_rewards = {k: v.reshape(len(samples), -1, *v.shape[1:]).to(accelerator.device) for k, v in set_rewards.items()}
        local_shape = samples[0]["rewards"]['avg'].shape[0]
        for reward_name in set_rewards:
            split_reward = torch.split(set_rewards[reward_name], 1, dim=0)
            for sample, split in zip(samples, split_reward):
                sample["set_rewards"][reward_name] = split.squeeze(0)[accelerator.process_index * local_shape:(accelerator.process_index + 1) * local_shape]

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [prompts[i] for i in sample_indices]  # noqa
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]  # noqa
                sampled_set_rewards = [set_rewards['avg'][-1][i] for i in sample_indices]  # noqa

                accelerator.log({
                    "images": [
                        swanlab.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | avg: {avg_reward:.2f} | avg set: {avg_set_reward:.2f}",
                        )
                        for idx, (prompt, avg_reward, avg_set_reward) in enumerate(zip(sampled_prompts, sampled_rewards, sampled_set_rewards))
                    ],
                }, step=global_step, )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"] + samples["set_rewards"]["avg"]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["all_avg"] = samples["rewards"]["ori_avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_set_rewards = {key: accelerator.gather(value) for key, value in samples["set_rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        gathered_set_rewards = {key: value.cpu().numpy() for key, value in gathered_set_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            accelerator.log({
                "epoch": epoch,
                **{f"train/reward_{key}": value.mean() for key, value in gathered_rewards.items() if
                   '_strict_accuracy' not in key and '_accuracy' not in key},
                **{f"train/set_reward_{key}": value.mean() for key, value in gathered_set_rewards.items() if
                     '_strict_accuracy' not in key and '_accuracy' not in key},
            }, step=global_step)

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, gathered_rewards['all_avg'])  # noqa
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                accelerator.log({
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                    "reward_std_mean": reward_std_mean,
                }, step=global_step)
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['all_avg'] - gathered_rewards['all_avg'].mean()) / (gathered_rewards['all_avg'].std() + 1e-4)

        # regather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            accelerator.log({
                "adv/mean": samples["advantages"].mean().item(),
                "adv/std": samples["advantages"].std().item(),
            }, step=global_step)
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["set_rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)

        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            accelerator.log({
                "actual_batch_size": mask.sum().item() // config.sample.num_batches_per_epoch,
            }, step=global_step)
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat([train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]])
                    pooled_embeds = torch.cat([train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]])
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            if config.rationorm:
                                prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                            else:
                                prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        if config.rationorm:
                                            _, _, prev_sample_mean_ref, _, _ = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                                        else:
                                            _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        if config.rationorm:
                            sigma_t = std_dev_t.mean()
                            # Reduce latent dimensions while keeping one correction per sample.
                            ratio_mean_bias = (prev_sample_mean - sample["prev_sample_mean"][:, j]).pow(2).mean(
                                dim=tuple(range(1, prev_sample_mean.ndim))
                            )
                            ratio_mean_bias = ratio_mean_bias / (2 * (sqrt_dt.mean() * sigma_t) ** 2)
                            ratio = torch.exp((log_prob - sample["log_probs"][:, j] + ratio_mean_bias) * (sqrt_dt.mean() * sigma_t))
                        else:
                            ratio = torch.exp(log_prob - sample["log_probs"][:, j])

                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        if config.rationorm:
                            policy_loss = policy_loss / (sqrt_dt.mean() ** 2)

                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["ratio"].append(ratio)
                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                        info["clipfrac_gt_one"].append(torch.mean((ratio - 1.0 > config.train.clip_range).float()))
                        info["clipfrac_lt_one"].append(torch.mean((1.0 - ratio > config.train.clip_range).float()))
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients

        epoch += 1
        if epoch >= config.max_epoch:
            break


if __name__ == "__main__":
    app.run(main)
