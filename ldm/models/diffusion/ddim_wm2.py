import torch
import numpy as np
import einops
from tqdm import tqdm
import time
import cv2
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor


class Dilation2d(nn.Module):
    def __init__(self, m=1):
        super(Dilation2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=-1e9)
        channel = self.unfold(x_pad).view(batch_size, c, -1, h, w)
        result = torch.max(channel, dim=2)[0]
        return result


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                  eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def calculate_overlap(self, mask1, mask2):
        return cv2.bitwise_and(mask1, mask2)

    def evaluate_masks(self, mask1, mask2, translation=(0, 0), angle=0):
        M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        translated_mask2 = cv2.warpAffine(mask2, M, (mask2.shape[1], mask2.shape[0]))

        center = (translated_mask2.shape[1] // 2, translated_mask2.shape[0] // 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_mask2 = cv2.warpAffine(translated_mask2, M_rot,
                                       (translated_mask2.shape[1], translated_mask2.shape[0]))

        overlap = self.calculate_overlap(mask1, rotated_mask2)
        return np.sum(overlap > 0), rotated_mask2, M, M_rot

    def find_best_overlap(self, mask1, mask2):
        best_area = 0
        best_mask = None
        best_translation = (0, 0)
        best_angle = 0

        for dx in range(-8, 9, 4):
            for dy in range(-8, 9, 4):
                for angle in range(0, 360, 60):
                    area, transformed_mask, M, M_rot = self.evaluate_masks(mask1, mask2, translation=(dx, dy),
                                                                           angle=angle)
                    if area > best_area:
                        best_area = area
                        best_mask = transformed_mask
                        best_translation = (dx, dy)
                        best_angle = angle

        return best_area, best_mask, best_translation, best_angle

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               use_weight_map=True,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               dynamic_threshold=None,
               ucg_schedule=None,
               real_mask=None,
               real_image=None,
               adaptive_real_image=False,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        if mask is not None:
            print("mask is not None")

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    real_mask=real_mask,
                                                    real_image=real_image,
                                                    adaptive_real_image=adaptive_real_image,
                                                    use_weight_map=use_weight_map)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, use_weight_map=True, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, real_mask=None, real_image=None, adaptive_real_image=False):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        weight_map = None
        wm2_total_ms = 0.0
        wm2_steps = 0
        aligned_real_image_list = None
        aligned_real_mask_list = None

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                new_mask = (mask > 0.5).float()
                img_orig = self.model.q_sample(x0, ts)

                if len(timesteps) == 50:
                    dialation = Dilation2d(m=int((ts[0] + 100) * 16 // 1000))
                else:
                    dialation = Dilation2d(m=int((ts[0] + 100) * 8 // 1000))
                new_mask = dialation(mask)

                if use_weight_map:
                    dis_map = ((new_mask * (img - x0)) ** 2).mean(dim=1)
                    dis_map = torch.where(dis_map == 0, -1, dis_map)
                    dis_map = 1 / dis_map
                    dis_map = new_mask.squeeze(1) * dis_map
                    bs = dis_map.size(0)
                    weight_map = new_mask.view(bs, -1).sum(dim=1).view(bs, 1, 1) * dis_map / (
                        dis_map.view(bs, -1).sum(dim=1).view(bs, 1, 1))
                    weight_map = torch.where(weight_map == 0, 1, weight_map)
                    weight_map = torch.where(weight_map < 1, 1, weight_map)
                    weight_map = torch.where(weight_map > 1.5, 1.5, weight_map)
                    for tmpi in range(bs):
                        if dis_map[tmpi].view(-1).sum() == 0:
                            weight_map[tmpi, :, :] = 1

                img = img * new_mask + (1 - new_mask) * img_orig

            if real_mask is not None and real_image is not None and adaptive_real_image:
                assert mask is not None
                assert x0 is not None

                if aligned_real_image_list is None or aligned_real_mask_list is None:
                    wm2_t0 = time.perf_counter()

                    if len(timesteps) == 50:
                        dialation = Dilation2d(m=int((ts[0] + 100) * 16 // 1000))
                    else:
                        dialation = Dilation2d(m=int((ts[0] + 100) * 8 // 1000))
                    new_mask_for_align = dialation(mask)

                    bs = b
                    real_image_list = torch.zeros((bs,) + img.shape[1:], dtype=img.dtype, device=device)
                    real_mask_list = torch.zeros((bs,) + new_mask_for_align.shape[1:], dtype=new_mask_for_align.dtype, device=device)

                    real_image_cpu = real_image.detach().float().cpu()
                    real_mask_cpu = real_mask.detach().float().cpu()
                    to_tensor = transforms.ToTensor()

                    for bi in range(bs):
                        gen_image = img[bi:bi + 1]
                        gen_mask = new_mask_for_align[bi:bi + 1]

                        gen_image_cpu = gen_image.detach().float().cpu()
                        gen_mask_cpu = gen_mask.detach().float().cpu()

                        if real_image_cpu.dim() >= 4 and real_image_cpu.shape[0] > 1:
                            real_image_latent_np = real_image_cpu[bi:bi + 1].numpy()
                        elif real_image_cpu.dim() >= 4:
                            real_image_latent_np = real_image_cpu[0:1].numpy()
                        else:
                            real_image_latent_np = real_image_cpu.numpy()

                        if real_mask_cpu.dim() >= 4 and real_mask_cpu.shape[0] > 1:
                            real_mask_np = real_mask_cpu[bi:bi + 1].numpy()
                        elif real_mask_cpu.dim() >= 4:
                            real_mask_np = real_mask_cpu[0:1].numpy()
                        else:
                            real_mask_np = real_mask_cpu.numpy()

                        gen_image_latent_np = gen_image_cpu.numpy()
                        gen_mask_np = gen_mask_cpu.numpy()

                        real_image_latent_tensor = torch.from_numpy(real_image_latent_np).to(device)
                        if real_image_latent_tensor.dim() == 3:
                            real_image_latent_tensor = real_image_latent_tensor.unsqueeze(0)
                        real_image_decoded = self.model.decode_first_stage(real_image_latent_tensor)
                        real_image_img_np = (einops.rearrange(real_image_decoded, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]

                        gen_image_latent_tensor = torch.from_numpy(gen_image_latent_np).to(device)
                        if gen_image_latent_tensor.dim() == 3:
                            gen_image_latent_tensor = gen_image_latent_tensor.unsqueeze(0)
                        gen_image_decoded = self.model.decode_first_stage(gen_image_latent_tensor)
                        gen_image_img_np = (einops.rearrange(gen_image_decoded, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]

                        real_mask_img_np = cv2.resize(
                            real_mask_np.squeeze().astype(np.float32),
                            (512, 512),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        gen_mask_img_np = cv2.resize(
                            gen_mask_np.squeeze().astype(np.float32),
                            (512, 512),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        _, real_mask_img_np = cv2.threshold(real_mask_img_np, 0.5, 1, cv2.THRESH_BINARY)
                        _, gen_mask_img_np = cv2.threshold(gen_mask_img_np, 0.5, 1, cv2.THRESH_BINARY)

                        _, best_mask_img, best_translation, best_angle = self.find_best_overlap(
                            gen_mask_img_np.astype(np.uint8), real_mask_img_np.astype(np.uint8))
                        if best_mask_img is None:
                            best_mask_img = gen_mask_img_np.astype(np.uint8)

                        M = np.float32([[1, 0, best_translation[0]], [0, 1, best_translation[1]]])
                        translated_real_image_img = cv2.warpAffine(real_image_img_np, M, (real_image_img_np.shape[1], real_image_img_np.shape[0]))
                        center = (translated_real_image_img.shape[1] // 2, translated_real_image_img.shape[0] // 2)
                        M_rot = cv2.getRotationMatrix2D(center, best_angle, 1.0)
                        rotated_real_image_img = cv2.warpAffine(
                            translated_real_image_img, M_rot,
                            (translated_real_image_img.shape[1], translated_real_image_img.shape[0])
                        )

                        if rotated_real_image_img.shape[:2] != gen_image_img_np.shape[:2]:
                            rotated_real_image_img = cv2.resize(
                                rotated_real_image_img,
                                (gen_image_img_np.shape[1], gen_image_img_np.shape[0]),
                            )

                        aligned_real_image_tensor = to_tensor(rotated_real_image_img).unsqueeze(0).to(device)
                        encoder_posterior_aligned = self.model.encode_first_stage(aligned_real_image_tensor)
                        aligned_real_image_latent = self.model.get_first_stage_encoding(encoder_posterior_aligned).detach()

                        aligned_real_mask_latent_np = cv2.resize(
                            best_mask_img.astype(np.float32), (shape[2], shape[3]), interpolation=cv2.INTER_NEAREST
                        )
                        aligned_real_mask_latent_np = aligned_real_mask_latent_np[np.newaxis, np.newaxis, :, :]
                        aligned_real_mask_latent_tensor = torch.from_numpy(aligned_real_mask_latent_np).to(device)

                        real_image_list[bi] = aligned_real_image_latent[0]
                        real_mask_list[bi] = aligned_real_mask_latent_tensor[0]

                    aligned_real_image_list = real_image_list
                    aligned_real_mask_list = real_mask_list
                    wm2_total_ms += (time.perf_counter() - wm2_t0) * 1000.0
                    wm2_steps += 1

                new_mask = (mask > 0.5).float()
                dis_map2 = ((new_mask * (img - aligned_real_image_list)) ** 2).mean(dim=1)
                min_val = dis_map2.min()
                max_val = dis_map2.max()
                normalized_dis_map2 = torch.where(
                    max_val - min_val == 0,
                    torch.zeros_like(dis_map2),
                    (dis_map2 - min_val) / (max_val - min_val),
                )
                dis_map2 = normalized_dis_map2 + new_mask.squeeze(1)
                weight_map2 = new_mask.squeeze(1) * dis_map2
                weight_map2 = torch.where(weight_map2 == 0, 1, weight_map2)
                weight_map = weight_map2

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            if weight_map is None:
                _, h, w = shape
                weight_map = torch.ones((b, h, w), device=device)

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold,
                                      weight_map=weight_map)
            img, pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        wm2_enabled = (real_mask is not None and real_image is not None and adaptive_real_image and wm2_steps > 0)
        # if wm2_enabled:
        #     avg_ms = wm2_total_ms / max(wm2_steps, 1)
        #     print(
        #         f"[ddim_wm2] per-image summary | weight_map2_enabled=True | wm2_steps={wm2_steps}/{total_steps} | wm2_avg_ms={avg_ms:.2f} | wm2_total_ms={wm2_total_ms:.2f}")
        # else:
        #     print(f"[ddim_wm2] per-image summary | weight_map2_enabled=False | wm2_steps={wm2_steps}/{total_steps}")

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None,
                      weight_map=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c, weight_map=weight_map)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback:
                callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback:
                callback(i)
        return x_dec
