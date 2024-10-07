import torch
from mask_generator import VideoMaskGenerator
import torch.nn.functional as F
import time

def retrieve_timesteps(
    scheduler,
    num_inference_steps,
    device,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def Evaluate_dataset(scheduler, loader, model, args, data_ST, dataset, index,device, replicate=6,experiment_dir=''):

    print('Evaluation!')

    model.eval()

    f = open(f"{experiment_dir}/val.txt",'a')
    f.write('\n-------------------------------')
    f.close()

    mask_index = [int(i) for i in args.mask_type.split('-')]

    mask_result = {}

    if replicate<10:
        mask_index = [0, 2, 8]

    for mask_i in mask_index:

        mean_pred_all, mean_real_all = [], []

        for repli in range(replicate):

            pred_all, real_all, mask_all = [], [], []

            data, ts = data_ST[dataset]

            matrix = loader[index][1]
            subgraphs = loader[index][2]

            for iindex, batch in enumerate(loader[index][0]):

                raw_x = torch.stack([data[i:i+args.seq_len] for i in batch[:,0]]).unsqueeze(dim=2).to(device)

                timestamps = torch.stack([ts[i:i+args.seq_len] for i in batch[:,0]]).to(device)
                generator = VideoMaskGenerator((raw_x.shape[-4], raw_x.shape[-2], raw_x.shape[-1]),pred_len = args.pred_len, his_len = args.his_len)
                mask = generator(raw_x.shape[0], device, idx=mask_i,seed=520)

                timesteps, _ = retrieve_timesteps(scheduler, args.num_inference_steps, device)
                latents = torch.randn(raw_x.shape).to(raw_x)

                assert raw_x.shape == mask.unsqueeze(dim=2).shape == latents.shape

                for i, t in enumerate(timesteps):

                    latents = raw_x * (1-mask.unsqueeze(dim=2)) + latents * mask.unsqueeze(dim=2)
                    latent_model_input = latents
                    vec_t = torch.ones((latent_model_input.shape[0],), device=latents.device) * t 
                    model_kwargs={'timestamps':timestamps,'hour_num':dataset.split('_')[-1],'mask':mask, 'data_name':dataset, 'node_split':subgraphs, 'topo':matrix}
                    noise_pred = model(latent_model_input, vec_t, **model_kwargs)
                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    #print(latents.shape)

                pred = latents * mask.unsqueeze(dim=2)
                
                real = raw_x * mask.unsqueeze(dim=2)

                assert pred.shape == real.shape

                pred_all.append(pred)
                real_all.append(real)
                mask_all.append(mask)

            pred_all = torch.cat(pred_all,dim=0)
            real_all = torch.cat(real_all,dim=0)
            mask = torch.cat(mask_all,dim=0)

            mean_pred_all.append(pred_all.unsqueeze(dim=0))
            mean_real_all.append(real_all.unsqueeze(dim=0))

            mean_pred_cal = torch.mean(torch.cat(mean_pred_all,dim=0), dim=0).squeeze(dim=1)
            mean_real_cal = torch.mean(torch.cat(mean_real_all,dim=0), dim=0).squeeze(dim=1)

            mean_pred_cal = mean_pred_cal.squeeze(dim=2)
            mean_real_cal = mean_real_cal.squeeze(dim=2)

            assert mean_pred_cal.shape == mean_real_cal.shape == mask.shape

            mae = (torch.abs(mean_pred_cal*mask-mean_real_cal*mask).sum()/mask.sum()).item()
            rmse = (torch.sqrt(((mean_pred_cal*mask-mean_real_cal*mask)**2).sum()/mask.sum())).item()

            if replicate>=10:
                print('mask_index:{}, repli:{}, mae:{}, rmse:{}'.format(mask_i, repli, mae, rmse))
                f = open(f"{experiment_dir}/result.txt",'a')
                f.write('mask_index:{}, repli:{}, mae:{}, rmse:{}\n'.format(mask_i, repli, mae, rmse))
                f.close()

            else:
                print('mask_index:{}, repli:{}, mae:{}, rmse:{}'.format(mask_i, repli, mae, rmse))
                f = open(f"{experiment_dir}/val.txt",'a')
                f.write('mask_index:{}, repli:{}, mae:{}, rmse:{}\n'.format(mask_i, repli, mae, rmse))
                f.close()


        mask_result[mask_i] = (rmse, mae)

    model.train()

    return mask_result   


