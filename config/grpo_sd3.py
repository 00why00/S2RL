import os

from importlib.machinery import SourceFileLoader

base = SourceFileLoader("base", os.path.join(os.path.dirname(__file__), "base_sd3.py")).load_module()


def base_config():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.use_lora = True
    config.dataset = 'ade20k'
    config.dataset_type = 'all_sample'
    config.sample_type = 'uniform'
    config.selected_images = 'time_1_ratio_0.005'
    config.max_epoch = 9999

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # rewards
    config.per_prompt_stat_tracking = True
    return config

def ade20k():
    gpu_number = 2
    config = base_config()
    config.dataset_type = 'all_sample'
    config.sample_type = 'uniform'
    config.selected_images = 'time_1_ratio_0.005'
    config.max_epoch = 9999

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 7
    config.resolution = 512

    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = int(16 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 5  # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.negative_prompt = ""

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.0
    config.sample.global_std = False
    config.train.ema = True

    config.save_freq = 30  # epoch
    config.eval_freq = 30
    config.save_dir = 'outputs/ade20k'

    config.reward_fn = {
        "upernet_za": 1.0,
        "segformer_b5_za": 1.0,
        "dpt_za": 1.0,
        "upernet_intra_div": 5.0,
        "segformer_b5_intra_div": 5.0,
        "dpt_intra_div": 5.0,
    }
    config.set_reward_fn = {
        "semantic_ucb_coverage": 3.0,
    }

    config.train.clip_range = 2e-6
    config.train.highclip_range = 2e-6
    config.train.learning_rate = 1e-4
    config.rationorm = True

    config.per_prompt_stat_tracking = True
    return config


def coco():
    gpu_number = 2
    config = base_config()
    config.dataset = 'coco'
    config.dataset_type = 'all_sample'
    config.sample_type = 'uniform'
    config.selected_images = 'time_1_ratio_0.000625'
    config.max_epoch = 9999

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 7
    config.resolution = 512

    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 10  # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.negative_prompt = ""

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.0
    config.sample.global_std = False
    config.train.ema = True

    config.save_freq = 30  # epoch
    config.eval_freq = 30
    config.save_dir = 'outputs/coco'

    config.reward_fn = {
        "mask2former_za": 1.0,
        "mask2former_intra_div": 10.0,
        "eomt_za": 1.0,
        "eomt_intra_div": 10.0,
        "eomt_1280_za": 1.0,
        "eomt_1280_intra_div": 10.0,
    }
    config.set_reward_fn = {
        "coco_semantic_ucb_coverage": 5.0,
    }

    config.train.clip_range = 2e-6
    config.train.highclip_range = 2e-6
    config.train.learning_rate = 1e-4
    config.rationorm = True

    config.per_prompt_stat_tracking = True
    return config


def get_config(name):
    return globals()[name]()
