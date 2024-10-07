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

from models import VDT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from mask_generator import VideoMaskGenerator
from test import Evaluate_dataset
from DataLoader import data_load_index_main

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
    # if dist.get_rank() == 0:  # real logger
    #     logging.basicConfig(
    #         level=logging.INFO,
    #         format='[\033[34m%(asctime)s\033[0m] %(message)s',
    #         datefmt='%Y-%m-%d %H:%M:%S',
    #         handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    #     )
    #     logger = logging.getLogger(__name__)
    # else:  # dummy logger (does nothing)
    #     logger = logging.getLogger(__name__)
    #     logger.addHandler(logging.NullHandler())

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

#################################################################################
#                                 Model selection                               #
#################################################################################


def VDT_L_2(**kwargs):
    return VDT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def VDT_S_2(**kwargs):
    return VDT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def VDT_S_2(**kwargs):
    return VDT(depth=6, hidden_size=256, patch_size=2, num_heads=4, num_frames=24, **kwargs)



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

    if '*' in args.dataset:
        data_replace = len(args.dataset.split('*'))
    else:
        data_replace = args.dataset

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        experiment_dir = f"{args.results_dir}/{args.norm_type}-{data_replace}-{args.diffusion_steps}-{model_string_name}-{args.t_patch_len}_{args.stride}-Mask_{args.mask_type}-Prompt_{args.is_prompt}-{args.prompt_content}-{args.machine}-Xstart{args.pred_xstart}"
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

    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=args.diffusion_steps,predict_xstart=bool(args.pred_xstart))  # default: 1000 steps, linear noise schedule
    #vae = AutoencoderKL.from_pretrained(f"stabilityai/zhuzhukeji/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr = 1e-5)

    # # Setup data:
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    # ])
    #dataset = ImageFolder(args.data_path, transform=transform)

    # x = torch.load("test_video.pt").to(device)
    # x = torch.randn([6, 16, 3, 32, 32])
    # B, T, C, H, W = x.shape
    # args.num_frames = T
    # dataset = x.cuda()

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

    if args.mode == 'testing':
        #for epoch in [200,220,240,280,300,160,400,480]:
        #for epoch in [300,160,400,480]:
        for epoch in [280, 400]:
            model_path = f"{checkpoint_dir}/{epoch}.pth"
            print(model_path)
            if os.path.exists(model_path):
                model.module.load_state_dict(torch.load(model_path,map_location=torch.device('cuda:{}'.format(device))))
                print('load model!')
                #f = open(f"{experiment_dir}/result_epoch.txt",'a')
                #f = open(f"{experiment_dir}/result_epoch_side.txt",'a')
                f = open(f"{experiment_dir}/result_epoch_side_2.txt",'a')
                f.write('\n------------------------------------------------\n')
                f.write(f"epoch:{epoch}\n")
                with torch.no_grad():
                    for index, dataset_name in enumerate(args.dataset.split('*')):
                        result = Evaluate_dataset(test_index, model, diffusion, args, data_ST, dataset_name, index, device, replicate=20,experiment_dir=experiment_dir)
                        for mask_i in result:
                            if args.norm_type == 'MinMax':
                                mae = (scaler[dataset_name]._max - scaler[dataset_name]._min) / 2 * result[mask_i][1]
                                rmse = (scaler[dataset_name]._max -scaler[dataset_name]._min) / 2 * result[mask_i][0]
                            else:
                                mae = result[mask_i][1] * scaler[dataset_name]._std
                                rmse = result[mask_i][0] * scaler[dataset_name]._std
                            f.write(f"{dataset_name}, mask_index:{mask_i}, rmse:{rmse}, mae:{mae}\n")
                            print(f"{dataset_name}, mask_index:{mask_i}, rmse:{rmse}, mae:{mae}\n")
                        print('\n')
                f.close()
            else:
                breakpoint()
        exit()

    for epoch in range(0, args.epochs):
        random.seed(epoch)
        random.shuffle(train_index)
        logger.info(f"Beginning epoch {epoch}...")
        #for x in loader:
            #x = x.to(device)
            #y = y.to(device)
            # with torch.no_grad():
            #     # Map input images to latent space + normalize latents:
            #     x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        for iiiiindex, (name, batch_index, matrix, subgraphs) in enumerate(train_index):
            batch, ts = data_ST[name]
            x = torch.stack([batch[i:i+args.seq_len] for i in batch_index[:,0]]).unsqueeze(dim=2).to(device)

            timestamps = torch.stack([ts[i:i+args.seq_len] for i in batch_index[:,0]]).to(device)

            mask_id = random.sample(mask_index,1)[0]

            if mask_id != 7:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            else:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0] * args.seq_len,), device=device)

            model_kwargs = dict()
            generator = VideoMaskGenerator((x.shape[-4], x.shape[-2], x.shape[-1]), pred_len = args.pred_len, his_len = args.his_len)
            mask = generator(x.shape[0], device, idx=mask_id)

            diffusion.training = True

            if mask_id ==7:
                x = x.reshape(-1,1,1,x.shape[-2],x.shape[-1])
                mask = mask.reshape(-1,1,mask.shape[-2],mask.shape[-1])
                timestamps = timestamps.reshape(-1,1,2)

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs={'timestamps':timestamps,'hour_num':name.split('_')[-1]}, mask=mask, args=args, mask_idx = mask_id)

            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

        if train_steps % 1 == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.6f}, Train Steps/Sec: {steps_per_sec:.2f}")
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()

        if epoch % 20 == 0:
            # checkpoint = {
            #     "model": model.module.state_dict(),
            #     "ema": ema.state_dict(),
            #     "opt": opt.state_dict(),
            #     "args": args
            # }
            #checkpoint_path = f"{checkpoint_dir}/{epoch}.pt"
            #torch.save(checkpoint, checkpoint_path)
            model_path = f"{checkpoint_dir}/{epoch}.pth"
            torch.save(model.module.state_dict(), model_path)
            model.module.load_state_dict(torch.load(model_path))

        # Save DiT checkpoint:
        if epoch % args.ckpt_every == 0 and epoch >= 200 or epoch == args.epochs-1:
            flag = 0
            val_rmse,num = 0.0, 0.0
            random.seed(epoch)
            with torch.no_grad():
                for index, dataset_name in enumerate(args.dataset.split('*')):
                    pp = random.random()
                    if pp<0.3:
                        flag==1
                        result = Evaluate_dataset(val_index_sub, model, diffusion, args, data_ST, dataset_name, index, device, experiment_dir=experiment_dir)
                        rmse = np.mean(list(result.values()))
                        val_rmse += rmse
                        num += 1
                if flag==0:
                    dataset_select = random.sample(args.dataset.split('*'))[0]
                    result = Evaluate_dataset(val_index_sub, model, diffusion, args, data_ST, dataset_select, index, device, experiment_dir=experiment_dir)
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
                    
                    # checkpoint_path = f"{checkpoint_dir}/checkpoint_best.pt"
                    # torch.save(checkpoint, checkpoint_path)
                    model_path = f"{checkpoint_dir}/model_best.pth"
                    torch.save(model.module.state_dict(), model_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()

                # if useful_iter >= 0:
                #     useful_iter = 0
                #     f = open(f"{experiment_dir}/result.txt",'a')
                #     f.write(f"training_steps:{train_steps}, epoch:{epoch}\n")
                #     for index, dataset_name in enumerate(args.dataset.split('*')):
                #         mae, rmse = Evaluate_dataset(test_index_sub, model, diffusion, args, data_ST, dataset_name, index, device)

                #         if args.norm_type == 'MinMax':
                #             mae = (scaler[dataset_name]._max - scaler[dataset_name]._min) / 2 * mae
                #             rmse = (scaler[dataset_name]._max -scaler[dataset_name]._min) / 2 * rmse
                #         else:
                #             mae = mae * scaler[dataset_name]._std
                #             rmse = rmse * scaler[dataset_name]._std

                #         f.write(f"dataset:{dataset_name}, rmse:{rmse}, mae:{mae}\n")
                    
                #     f.write('\n')
                #     f.close()

            else:
                useful_iter = 0
                early_stop += 1

                if early_stop > args.early_stop:
                    logger.info(f"early stop!")

                    # load best model
                    checkpoint = torch.load(f"{checkpoint_dir}/checkpoint_best.pt")
                    model.module.load_state_dict(checkpoint['model'])
                    f = open(f"{experiment_dir}/result.txt",'a')
                    f.write('------------------------------------------------\n')
                    f.write(f"final evaluation:{train_steps}, epoch:{epoch}\n")
                    with torch.no_grad():
                        for index, dataset_name in enumerate(args.dataset.split('*')):
                            result = Evaluate_dataset(test_index, model, diffusion, args, data_ST, dataset_name, index, device, replicate=20,experiment_dir=experiment_dir)
                            for mask_i in result:
                                if args.norm_type == 'MinMax':
                                    mae = (scaler[dataset_name]._max - scaler[dataset_name]._min) / 2 * result[mask_i][1]
                                    rmse = (scaler[dataset_name]._max -scaler[dataset_name]._min) / 2 * result[mask_i][0]
                                else:
                                    mae = result[mask_i][1] * scaler[dataset_name]._std
                                    rmse = result[mask_i][0] * scaler[dataset_name]._std
                                f.write(f"{dataset_name}, mask_index:{mask_i}, rmse:{rmse}, mae:{mae}\n")
                                print(f"{dataset_name}, mask_index:{mask_i}, rmse:{rmse}, mae:{mae}\n")
                            print('\n')
                            f.close()
                            
                    cleanup()
                    exit()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    #parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="experiments_all_task")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--machine", type=str, default="LM2")
    parser.add_argument("--diffusion_steps", type=int, default=200)
    parser.add_argument("--pred_xstart", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--norm_type", type=str, default='MinMax',choices = ['standard','MinMax'])
    parser.add_argument("--mask_id", type=int, default=0)
    parser.add_argument("--mask_type", type=str, default='0-1-2-5')
    parser.add_argument("--dataset", type=str, default="TaxiBJ13_48")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--t_patch_len", type=int, default=2)
    parser.add_argument("--time_patch", type=int, default=0)
    parser.add_argument("--batch_size_taxibj", type=int, default=128)
    parser.add_argument("--batch_size_chy", type=int, default=64)
    parser.add_argument("--batch_size_nyc", type=int, default=128)
    parser.add_argument("--batch_ratio", type=float, default=1.0)
    parser.add_argument("--multi_patch_size", type=str, default='2-2-100')
    parser.add_argument("--few_ratio", type=float, default=1.0)
    parser.add_argument("--few_data", type=str, default='')
    parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-S/2")
    #parser.add_argument("--image_size", type=int, choices=[32, 64], default=32)
    #parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--global_batch_size", type=int, default=8)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--num_memory", type=int, default=512)
    parser.add_argument("--prompt_content", type=str, default='')
    parser.add_argument("--is_prompt", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--his_len", type=int, default=12)
    parser.add_argument("--reso", type=int, default=2)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)


    args = parser.parse_args()
    if args.time_patch==0:
        args.stride = 1
        args.t_patch_len = 1
    if args.machine=='LM2':
        os.environ['MASTER_ADDR'] = '101.6.69.35'
        os.environ['MASTER_PORT'] = '{}'.format(6042+args.device_id)
    if args.machine == 'LM1' or 'LM1' in args.machine:
        os.environ['MASTER_ADDR'] = '101.6.69.60'
        os.environ['MASTER_PORT'] = '{}'.format(5234+args.device_id)
    if args.machine == 'DL4':
        os.environ['MASTER_ADDR'] = '101.6.69.111'
        os.environ['MASTER_PORT'] = '{}'.format(6234+args.device_id)
    
    print('start main!')
    main(args)