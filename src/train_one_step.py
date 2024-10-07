# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import random
import torch.autograd.profiler as profiler

from models import VDT_models
#from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from mask_generator import VideoMaskGenerator
from validation import Evaluate_dataset
from DataLoader import data_load_index_main

import copy


import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

torch.set_num_threads(32)

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """

    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)

    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def get_sigmas(noise_scheduler_copy, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device= timesteps.device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to( timesteps.device)
    timesteps = timesteps.to( timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    print(torch.cuda.is_available())
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    #device = rank % torch.cuda.device_count()
    device = args.device_id
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if '*' in args.dataset and not (args.dataset == 'TaxiBJ13_48' and 'TaxiBJ' in args.few_data_name):
        data_replace = len(args.dataset.split('*'))
    else:
        data_replace = args.dataset
        if args.dataset == 'TaxiBJ13_48' and 'TaxiBJ' in args.few_data_name:
            data_replace = 6

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"experiments/{args.norm_type}-{data_replace}-{args.diffusion_steps}-{args.num_inference_steps}-{model_string_name}-{args.t_patch_len}_{args.stride}-Mask_{args.mask_type}-Prompt_{args.is_prompt}-{args.prompt_content}-{args.machine}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    else:
        logger = create_logger(None)

    # Create model:
    #assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = 32
   
    model = VDT_models[args.model](
        input_size=latent_size, args=args
    ).to(device)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[args.device_id], find_unused_parameters=True)

    #diffusion = create_diffusion(timestep_respacing="", diffusion_steps=args.diffusion_steps,predict_xstart=bool(args.pred_xstart))  # default: 1000 steps, linear noise schedule
    #vae = AutoencoderKL.from_pretrained(f"stabilityai/zhuzhukeji/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr = 1e-5)

    data_ST, train_index, test_index, val_index, test_index_sub, val_index_sub, scaler = data_load_index_main(args)

    sampler = DistributedSampler(
        train_index,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )

    loader = DataLoader(
        train_index,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        #num_workers=args.num_workers,
        #pin_memory=True,
        drop_last=False
    )

    #logger.info(f"Dataset contains {len(train_index):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    early_stop = 0
    min_rmse = 1e9
    useful_iter = 0
    start_time = time()
    mask_index = [int(i) for i in args.mask_type.split('-')]

    f = open(f"{experiment_dir}/result.txt",'a')
    f.write('start training!\n\n')
    f.close()

    logger.info(f"Training for {args.epochs} epochs...")

    # Load scheduler and models
    #noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained('/data2/zhengyu/workspace/sd3', subfolder="scheduler")
    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps = args.diffusion_steps, shift=3.0, use_dynamic_shifting=False, base_shift=0.5, max_shift=1.15)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    for epoch in range(0, args.epochs):
        random.seed(epoch)
        random.shuffle(train_index)
        model.train()
        logger.info(f"Beginning epoch {epoch}...")
        for iiiiindex, (name, batch_index, matrix, subgraphs) in enumerate(train_index):
            batch, ts = data_ST[name]
            x = torch.stack([batch[i:i+args.seq_len] for i in batch_index[:,0]]).unsqueeze(dim=2).to(device)
            timestamps = torch.stack([ts[i:i+args.seq_len] for i in batch_index[:,0]]).to(device)
            mask_id = random.sample(mask_index,1)[0]

            model_kwargs = dict()
            generator = VideoMaskGenerator((x.shape[-4], x.shape[-2], x.shape[-1]), pred_len = args.pred_len, his_len = args.his_len)
            
            if args.mask_random==0:
                mask = generator(x.shape[0], device, idx=mask_id, seed=520)
            else:
                mask = generator(x.shape[0], device, idx=mask_id)

            model_kwargs['mask'] = mask
            model_kwargs['mask_idx'] = mask_id
            model_kwargs['hour_num'] = name.split('_')[-1]
            model_kwargs['timestamps'] = timestamps
            model_kwargs['data_name'] = name
            model_kwargs['node_split'] = subgraphs
            model_kwargs['topo'] = matrix

            # Sample noise that we'll add to the latents
            model_input = x.clone()
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device)

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = get_sigmas(noise_scheduler_copy, timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

            noisy_model_input = noisy_model_input.reshape(x.shape)
            assert x.shape == noisy_model_input.shape == mask.unsqueeze(dim=2).shape, f"Shapes do not match: {x.shape}, {noisy_model_input.shape}, {mask.shape}"

            noisy_model_input = x * (1-mask.unsqueeze(dim=2)) + noisy_model_input * mask.unsqueeze(dim=2)

            model_pred = model(noisy_model_input, timesteps, **model_kwargs)

            # Preconditioning of the model outputs.
            if args.precondition_outputs:
                model_pred = model_pred * (-sigmas) + noisy_model_input

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

            # flow matching loss
            if args.precondition_outputs:
                target = model_input
            else:
                target = noise - model_input

            if args.with_prior_preservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute prior loss
                prior_loss = torch.mean(
                    (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                        target_prior.shape[0], -1
                    ),
                    1,
                )
                prior_loss = prior_loss.mean()

            # Compute regular loss.
            loss = torch.mean(
                (weighting.float() * (((model_pred.float() - target.float()))*mask.unsqueeze(dim=2)) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()

            if args.with_prior_preservation:
                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss
            opt.zero_grad()
            #with profiler.profile(with_stack=True, use_cuda=torch.cuda.is_available()) as prof:
            loss.backward()
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            #exit()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

        if epoch % 1 == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = (end_time - start_time)/60.0
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.6f}, Train Sec: {steps_per_sec:.3f} min")
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()

        if epoch % 20 == 0:
            model_path = f"{checkpoint_dir}/{epoch}.pth"
            torch.save(model.module.state_dict(), model_path)
            model.module.load_state_dict(torch.load(model_path))

        # Save DiT checkpoint:
        if epoch % args.ckpt_every == 0 and epoch >0:
            flag = 0
            val_rmse,num = 0.0, 0.0
            random.seed(epoch)
            with torch.no_grad():
                for index, dataset_name in enumerate(args.dataset.split('*')):
                    flag==1
                    result = Evaluate_dataset(noise_scheduler, val_index_sub, model, args, data_ST, dataset_name, index, device, experiment_dir=experiment_dir)
                    rmse = np.mean(list(result.values()))
                    val_rmse += rmse
                    num += 1

            scheduler.step(val_rmse/num)

            if val_rmse/num < min_rmse and epoch < args.epochs-1:
                min_rmse = rmse
                useful_iter += 1
                early_stop = 0
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    
                    #checkpoint_path = f"{checkpoint_dir}/checkpoint_best.pt"
                    # torch.save(checkpoint, checkpoint_path)
                    model_path = f"{checkpoint_dir}/model_best.pth"
                    torch.save(model.module.state_dict(), model_path)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
                    dist.barrier()


    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    #parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="experiments_all_task")
    parser.add_argument("--pretrained_file_path", type=str, default='')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--machine", type=str, default="LM2")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--few_data_name", type=str, default='')
    parser.add_argument("--is_spatial", type=int, default=1)
    parser.add_argument("--fft", type=int, default=0)
    parser.add_argument("--fft_thred", type=int, default=0)
    parser.add_argument("--best_epoch", type=int, default=0)
    parser.add_argument("--diffusion_steps", type=int, default=200)
    parser.add_argument("--token_gcn", type=int, default=0)
    parser.add_argument("--mask_random", type=int, default=0)
    parser.add_argument("--pred_xstart", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--norm_type", type=str, default='MinMax',choices = ['standard','MinMax'])
    parser.add_argument("--mask_id", type=int, default=0)
    parser.add_argument("--mask_type", type=str, default='0-1-2-3-4-5')
    parser.add_argument("--dataset", type=str, default="TaxiBJ13_48")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--t_patch_len", type=int, default=2)
    parser.add_argument("--time_patch", type=int, default=1)
    parser.add_argument("--batch_size_taxibj", type=int, default=128)
    parser.add_argument("--batch_size_Pop", type=int, default=48)
    parser.add_argument("--batch_size_nyc", type=int, default=128)
    parser.add_argument("--batch_size_graph_large", type=int, default=64)
    parser.add_argument("--batch_size_graph_small", type=int, default=128)
    parser.add_argument("--batch_ratio", type=float, default=1.0)
    parser.add_argument("--multi_patch_size", type=str, default='2-2-100')
    parser.add_argument("--few_ratio", type=float, default=1.0)
    parser.add_argument("--few_data", type=str, default='')
    parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-S/2")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--global_batch_size", type=int, default=8)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=10)
    parser.add_argument("--num_memory", type=int, default=512)
    parser.add_argument("--prompt_content", type=str, default='')
    parser.add_argument("--is_prompt", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--his_len", type=int, default=12)
    parser.add_argument("--reso", type=int, default=2)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        required=False,
        help="`",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )

    args = parser.parse_args()
    
    print('start main!')
    main(args)