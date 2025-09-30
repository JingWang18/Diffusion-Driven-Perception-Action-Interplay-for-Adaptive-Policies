from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class VAEVisionEncoder(nn.Module):
    def __init__(self, base_encoder, input_dim, latent_dim):
        super().__init__()
        self.base_encoder = base_encoder
        self.fc_log_var = nn.Linear(input_dim, latent_dim)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, obs_dict):
        # obs_dict: dictionary of observations
        features = self.base_encoder(obs_dict)  # (B*T, input_dim)
        mu = features  # (B*T, latent_dim)
        log_var = self.fc_log_var(features)  # (B*T, latent_dim)
        
        return features, mu, log_var
    
class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_feature_dim = obs_encoder.output_shape()[0]
        self.vae_encoder = VAEVisionEncoder(
            base_encoder=obs_encoder,
            input_dim=obs_feature_dim,
            latent_dim=obs_feature_dim
        )

        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("VAE params: %e" % sum(p.numel() for p in self.vae_encoder.parameters()))
    
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict noise
            noise_pred = model(
                sample=trajectory,
                timestep=t,
                local_cond=local_cond,
                global_cond=global_cond
            )

            # 3. compute previous sample: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=trajectory,
                generator=generator,
                **kwargs
            ).prev_sample
        
        # Ensure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        # Get VAE features
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        obs_features, _, _ = self.vae_encoder(this_nobs)
        
        # # Reparameterization trick
        # sigma = torch.exp(0.5*log_var)
        # eps = torch.randn_like(mu)
        # obs_features = mu + sigma * eps  # z_t
        
        # Reshape for global conditioning
        obs_features = obs_features.reshape(B, To, -1)
        global_cond = obs_features.flatten(start_dim=1)  # (B, To * Do)

        # Initialize conditioning data
        local_cond = None
        if self.obs_as_global_cond:
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = obs_features
            cond_mask[:,:To,Da:] = True

        # Run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs
        )

        # Unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Select action steps
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        device = nactions.device

        # Handle observations
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            obs_features, mu, log_var = self.vae_encoder(this_nobs)
            obs_features = obs_features.reshape(batch_size, self.n_obs_steps, -1)
            mu = mu.reshape(batch_size, self.n_obs_steps, -1)
            log_var = log_var.reshape(batch_size, self.n_obs_steps, -1)
            global_cond = obs_features.flatten(start_dim=1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            obs_features, mu, log_var = self.vae_encoder(this_nobs)
            obs_features = obs_features.reshape(batch_size, horizon, -1)
            mu = mu.reshape(batch_size, horizon, -1)
            log_var = log_var.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, obs_features], dim=-1)
            trajectory = cond_data.detach()

        # Reparameterization trick
        log_var = torch.clamp(log_var, min=-10.0, max=2.0)
        sigma = torch.exp(0.5*log_var)

        # Generate conditioning mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise and timesteps
        noise = torch.randn(trajectory.shape, device=device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device).long()

        # Add noise to trajectory
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # Apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict noise residual
        noise_pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)
        
        # Compute losses
        # 1. Diffusion loss with masking
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss_mask = ~condition_mask
        diffusion_loss = F.mse_loss(noise_pred, target/0.4, reduction='none')
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
        diffusion_loss = reduce(diffusion_loss, 'b ... -> b (...)', 'mean')
        diffusion_loss = diffusion_loss.mean()
        
        # 2. Guidance loss
        v = noise_pred.reshape(batch_size, -1) 
        vjp = torch.autograd.grad(
            outputs=v,
            inputs=obs_features,
            grad_outputs=v,
            retain_graph=True,
            create_graph=True)[0].detach()
        
        vjp = vjp / (vjp.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        z_t_guided = mu + sigma * vjp
        
        noise_pred_guided = self.model(noisy_trajectory, timesteps, local_cond=local_cond,
                                       global_cond=z_t_guided.flatten(start_dim=1))
        
        # 2. InfoNCE-based contrastive loss
        # Normalize features for cosine similarity
        noise_pred_re = noise_pred.reshape(batch_size,-1)
        noise_pred_guided_re = noise_pred_guided.reshape(batch_size,-1).detach()
        noise_pred_norm = F.normalize(noise_pred_re, dim=-1)
        noise_pred_guided_norm = F.normalize(noise_pred_guided_re, dim=-1)

        # Compute similarity scores
        temperature = 0.8
        logits = torch.matmul(noise_pred_norm, noise_pred_guided_norm.T) / temperature  # (B, B)
        
        # Create labels (diagonal is positive)
        labels = torch.arange(batch_size, device=device)
        
        # Compute InfoNCE loss (cross-entropy over softmax of logits)
        loss_z = F.cross_entropy(logits, labels)
        loss_z = loss_z * loss_mask.type(loss_z.dtype)
        loss_z = reduce(loss_z, 'b ... -> b (...)', 'mean')
        loss_z = loss_z.mean()
        
        # 3. Regularize the mean and variance of variational inference
        # KLD
        loss_kl = torch.mean(-0.5 * (1 + log_var - mu.pow(2) - torch.exp(log_var)))

        # Combine losses
        lambda_z = 1.0  # Guidance weight
        loss = diffusion_loss + lambda_z * loss_z + 1.0*loss_kl

        return loss
    
    
# class VAEVisionEncoder(nn.Module):
#     def __init__(self, base_encoder, input_dim, latent_dim):
#         super().__init__()
#         self.base_encoder = base_encoder
#         self.fc_mu = nn.Linear(input_dim, latent_dim)
#         self.fc_log_var = nn.Linear(input_dim, latent_dim)
        
#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def forward(self, obs_dict):
#         # obs_dict: dictionary of observations
#         features = self.base_encoder(obs_dict)  # (B*T, input_dim)
#         mu = self.fc_mu(features)  # (B*T, latent_dim)
#         log_var = self.fc_log_var(features)  # (B*T, latent_dim)
        
#         return features, mu, log_var
    
# class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
#     def __init__(self, 
#             shape_meta: dict,
#             noise_scheduler: DDPMScheduler,
#             horizon, 
#             n_action_steps, 
#             n_obs_steps,
#             num_inference_steps=None,
#             obs_as_global_cond=True,
#             crop_shape=(76, 76),
#             diffusion_step_embed_dim=256,
#             down_dims=(256,512,1024),
#             kernel_size=5,
#             n_groups=8,
#             cond_predict_scale=True,
#             obs_encoder_group_norm=False,
#             eval_fixed_crop=False,
#             **kwargs):
#         super().__init__()

#         # parse shape_meta
#         action_shape = shape_meta['action']['shape']
#         assert len(action_shape) == 1
#         action_dim = action_shape[0]
#         obs_shape_meta = shape_meta['obs']
#         obs_config = {
#             'low_dim': [],
#             'rgb': [],
#             'depth': [],
#             'scan': []
#         }
#         obs_key_shapes = dict()
#         for key, attr in obs_shape_meta.items():
#             shape = attr['shape']
#             obs_key_shapes[key] = list(shape)
#             type = attr.get('type', 'low_dim')
#             if type == 'rgb':
#                 obs_config['rgb'].append(key)
#             elif type == 'low_dim':
#                 obs_config['low_dim'].append(key)
#             else:
#                 raise RuntimeError(f"Unsupported obs type: {type}")

#         config = get_robomimic_config(
#             algo_name='bc_rnn',
#             hdf5_type='image',
#             task_name='square',
#             dataset_type='ph')
        
#         with config.unlocked():
#             config.observation.modalities.obs = obs_config
#             if crop_shape is None:
#                 for key, modality in config.observation.encoder.items():
#                     if modality.obs_randomizer_class == 'CropRandomizer':
#                         modality['obs_randomizer_class'] = None
#             else:
#                 ch, cw = crop_shape
#                 for key, modality in config.observation.encoder.items():
#                     if modality.obs_randomizer_class == 'CropRandomizer':
#                         modality.obs_randomizer_kwargs.crop_height = ch
#                         modality.obs_randomizer_kwargs.crop_width = cw

#         ObsUtils.initialize_obs_utils_with_config(config)

#         policy: PolicyAlgo = algo_factory(
#                 algo_name=config.algo_name,
#                 config=config,
#                 obs_key_shapes=obs_key_shapes,
#                 ac_dim=action_dim,
#                 device='cpu',
#             )

#         obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
#         if obs_encoder_group_norm:
#             replace_submodules(
#                 root_module=obs_encoder,
#                 predicate=lambda x: isinstance(x, nn.BatchNorm2d),
#                 func=lambda x: nn.GroupNorm(
#                     num_groups=x.num_features//16, 
#                     num_channels=x.num_features)
#             )

#         if eval_fixed_crop:
#             replace_submodules(
#                 root_module=obs_encoder,
#                 predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
#                 func=lambda x: dmvc.CropRandomizer(
#                     input_shape=x.input_shape,
#                     crop_height=x.crop_height,
#                     crop_width=x.crop_width,
#                     num_crops=x.num_crops,
#                     pos_enc=x.pos_enc
#                 )
#             )

#         obs_feature_dim = obs_encoder.output_shape()[0]
#         self.vae_encoder = VAEVisionEncoder(
#             base_encoder=obs_encoder,
#             input_dim=obs_feature_dim,
#             latent_dim=obs_feature_dim
#         )

#         input_dim = action_dim + obs_feature_dim
#         global_cond_dim = None
#         if obs_as_global_cond:
#             input_dim = action_dim
#             global_cond_dim = obs_feature_dim * n_obs_steps

#         model = ConditionalUnet1D(
#             input_dim=input_dim,
#             local_cond_dim=None,
#             global_cond_dim=global_cond_dim,
#             diffusion_step_embed_dim=diffusion_step_embed_dim,
#             down_dims=down_dims,
#             kernel_size=kernel_size,
#             n_groups=n_groups,
#             cond_predict_scale=cond_predict_scale
#         )

#         self.model = model
#         self.noise_scheduler = noise_scheduler
#         self.mask_generator = LowdimMaskGenerator(
#             action_dim=action_dim,
#             obs_dim=0 if obs_as_global_cond else obs_feature_dim,
#             max_n_obs_steps=n_obs_steps,
#             fix_obs_steps=True,
#             action_visible=False
#         )
#         self.normalizer = LinearNormalizer()
#         self.horizon = horizon
#         self.obs_feature_dim = obs_feature_dim
#         self.action_dim = action_dim
#         self.n_action_steps = n_action_steps
#         self.n_obs_steps = n_obs_steps
#         self.obs_as_global_cond = obs_as_global_cond
#         self.kwargs = kwargs

#         if num_inference_steps is None:
#             num_inference_steps = noise_scheduler.config.num_train_timesteps
#         self.num_inference_steps = num_inference_steps

#         print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
#         print("VAE params: %e" % sum(p.numel() for p in self.vae_encoder.parameters()))
    
#     def conditional_sample(self, 
#             condition_data, condition_mask,
#             local_cond=None, global_cond=None,
#             generator=None,
#             **kwargs):
#         model = self.model
#         scheduler = self.noise_scheduler

#         trajectory = torch.randn(
#             size=condition_data.shape, 
#             dtype=condition_data.dtype,
#             device=condition_data.device,
#             generator=generator)
    
#         scheduler.set_timesteps(self.num_inference_steps)

#         for t in scheduler.timesteps:
#             # 1. apply conditioning
#             trajectory[condition_mask] = condition_data[condition_mask]

#             # 2. predict noise
#             noise_pred = model(
#                 sample=trajectory,
#                 timestep=t,
#                 local_cond=local_cond,
#                 global_cond=global_cond
#             )

#             # 3. compute previous sample: x_t -> x_t-1
#             trajectory = scheduler.step(
#                 model_output=noise_pred,
#                 timestep=t,
#                 sample=trajectory,
#                 generator=generator,
#                 **kwargs
#             ).prev_sample
        
#         # Ensure conditioning is enforced
#         trajectory[condition_mask] = condition_data[condition_mask]        
#         return trajectory

#     def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         assert 'past_action' not in obs_dict
#         nobs = self.normalizer.normalize(obs_dict)
#         value = next(iter(nobs.values()))
#         B, To = value.shape[:2]
#         T = self.horizon
#         Da = self.action_dim
#         Do = self.obs_feature_dim
#         To = self.n_obs_steps

#         device = self.device
#         dtype = self.dtype

#         # Get VAE features
#         this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
#         obs_features, _, _ = self.vae_encoder(this_nobs)
        
#         # # Reparameterization trick
#         # sigma = torch.exp(0.5*log_var)
#         # eps = torch.randn_like(mu)
#         # obs_features = mu + sigma * eps  # z_t
        
#         # Reshape for global conditioning
#         obs_features = obs_features.reshape(B, To, -1)
#         global_cond = obs_features.flatten(start_dim=1)  # (B, To * Do)

#         # Initialize conditioning data
#         local_cond = None
#         if self.obs_as_global_cond:
#             cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
#             cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
#         else:
#             cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
#             cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
#             cond_data[:,:To,Da:] = obs_features
#             cond_mask[:,:To,Da:] = True

#         # Run sampling
#         nsample = self.conditional_sample(
#             condition_data=cond_data,
#             condition_mask=cond_mask,
#             local_cond=local_cond,
#             global_cond=global_cond,
#             **self.kwargs
#         )

#         # Unnormalize prediction
#         naction_pred = nsample[...,:Da]
#         action_pred = self.normalizer['action'].unnormalize(naction_pred)

#         # Select action steps
#         start = To - 1
#         end = start + self.n_action_steps
#         action = action_pred[:, start:end]

#         result = {
#             'action': action,
#             'action_pred': action_pred
#         }
#         return result
    
#     def set_normalizer(self, normalizer: LinearNormalizer):
#         self.normalizer.load_state_dict(normalizer.state_dict())
    
#     def compute_loss(self, batch):
#         assert 'valid_mask' not in batch
#         nobs = self.normalizer.normalize(batch['obs'])
#         nactions = self.normalizer['action'].normalize(batch['action'])
#         batch_size = nactions.shape[0]
#         horizon = nactions.shape[1]

#         device = nactions.device

#         # Handle observations
#         local_cond = None
#         global_cond = None
#         trajectory = nactions
#         cond_data = trajectory

#         if self.obs_as_global_cond:
#             this_nobs = dict_apply(nobs, 
#                 lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
#             obs_features, mu, log_var = self.vae_encoder(this_nobs)
#             obs_features = obs_features.reshape(batch_size, self.n_obs_steps, -1)
#             mu = mu.reshape(batch_size, self.n_obs_steps, -1)
#             log_var = log_var.reshape(batch_size, self.n_obs_steps, -1)
#             global_cond = obs_features.flatten(start_dim=1)
#         else:
#             this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
#             obs_features, mu, log_var = self.vae_encoder(this_nobs)
#             obs_features = obs_features.reshape(batch_size, horizon, -1)
#             mu = mu.reshape(batch_size, horizon, -1)
#             log_var = log_var.reshape(batch_size, horizon, -1)
#             cond_data = torch.cat([nactions, obs_features], dim=-1)
#             trajectory = cond_data.detach()

#         # Reparameterization trick
#         log_var = torch.clamp(log_var, min=-10.0, max=2.0)
#         sigma = torch.exp(0.5*log_var)

#         # Generate conditioning mask
#         condition_mask = self.mask_generator(trajectory.shape)

#         # Sample noise and timesteps
#         noise = torch.randn(trajectory.shape, device=device)
#         timesteps = torch.randint(
#             0, self.noise_scheduler.config.num_train_timesteps,
#             (batch_size,), device=device).long()

#         # Add noise to trajectory
#         noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

#         # Apply conditioning
#         noisy_trajectory[condition_mask] = cond_data[condition_mask]

#         # Predict noise residual
#         noise_pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)
        
#         # Compute losses
#         # 1. Diffusion loss with masking
#         pred_type = self.noise_scheduler.config.prediction_type
#         if pred_type == 'epsilon':
#             target = noise
#         elif pred_type == 'sample':
#             target = trajectory
#         else:
#             raise ValueError(f"Unsupported prediction type {pred_type}")

#         loss_mask = ~condition_mask
#         diffusion_loss = F.mse_loss(noise_pred, target/0.4, reduction='none')
#         diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
#         diffusion_loss = reduce(diffusion_loss, 'b ... -> b (...)', 'mean')
#         diffusion_loss = diffusion_loss.mean()

#         # 2. Guidance loss
#         v = noise_pred.reshape(batch_size, -1)
#         epsilon_z_guided = torch.autograd.grad(
#             (v ** 2).sum(),
#             obs_features,
#             create_graph=True)[0]
#         z_t_guided = mu + 0.1 * sigma * epsilon_z_guided
        
#         noise_pred_guided = self.model(noisy_trajectory, timesteps, local_cond=local_cond,
#                                        global_cond=z_t_guided.flatten(start_dim=1))
        
#         # 2. InfoNCE-based contrastive loss
#         # Normalize features for cosine similarity
#         noise_pred_re = noise_pred.reshape(batch_size,-1)
#         noise_pred_guided_re = noise_pred_guided.reshape(batch_size,-1)
#         noise_pred_norm = F.normalize(noise_pred_re, dim=-1)
#         noise_pred_guided_norm = F.normalize(noise_pred_guided_re, dim=-1).detach()

#         # Compute similarity scores
#         temperature = 0.8
#         logits = torch.matmul(noise_pred_norm, noise_pred_guided_norm.T) / temperature  # (B, B)
        
#         # Create labels (diagonal is positive)
#         labels = torch.arange(batch_size, device=device)
        
#         # Compute InfoNCE loss (cross-entropy over softmax of logits)
#         loss_z = F.cross_entropy(logits, labels)
#         loss_z = loss_z * loss_mask.type(loss_z.dtype)
#         loss_z = reduce(loss_z, 'b ... -> b (...)', 'mean')
#         loss_z = loss_z.mean()
        
#         # 3. Regularize the mean and variance of variational inference
#         # KLD
#         # loss_kl = torch.mean(-0.5 * (1 + log_var - mu.pow(2) - torch.exp(log_var)))

#         # Combine losses
#         lambda_z = 1.0  # Guidance weight
#         loss = diffusion_loss + lambda_z * loss_z

#         return loss

# class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
#     def __init__(self, 
#             shape_meta: dict,
#             noise_scheduler: DDPMScheduler,
#             horizon, 
#             n_action_steps, 
#             n_obs_steps,
#             num_inference_steps=None,
#             obs_as_global_cond=True,
#             crop_shape=(76, 76),
#             diffusion_step_embed_dim=256,
#             down_dims=(256,512,1024),
#             kernel_size=5,
#             n_groups=8,
#             cond_predict_scale=True,
#             obs_encoder_group_norm=False,
#             eval_fixed_crop=False,
#             # parameters passed to step
#             **kwargs):
#         super().__init__()

#         # parse shape_meta
#         action_shape = shape_meta['action']['shape']
#         assert len(action_shape) == 1
#         action_dim = action_shape[0]
#         obs_shape_meta = shape_meta['obs']
#         obs_config = {
#             'low_dim': [],
#             'rgb': [],
#             'depth': [],
#             'scan': []
#         }
#         obs_key_shapes = dict()
#         for key, attr in obs_shape_meta.items():
#             shape = attr['shape']
#             obs_key_shapes[key] = list(shape)

#             type = attr.get('type', 'low_dim')
#             if type == 'rgb':
#                 obs_config['rgb'].append(key)
#             elif type == 'low_dim':
#                 obs_config['low_dim'].append(key)
#             else:
#                 raise RuntimeError(f"Unsupported obs type: {type}")

#         # get raw robomimic config
#         config = get_robomimic_config(
#             algo_name='bc_rnn',
#             hdf5_type='image',
#             task_name='square',
#             dataset_type='ph')
        
#         with config.unlocked():
#             # set config with shape_meta
#             config.observation.modalities.obs = obs_config

#             if crop_shape is None:
#                 for key, modality in config.observation.encoder.items():
#                     if modality.obs_randomizer_class == 'CropRandomizer':
#                         modality['obs_randomizer_class'] = None
#             else:
#                 # set random crop parameter
#                 ch, cw = crop_shape
#                 for key, modality in config.observation.encoder.items():
#                     if modality.obs_randomizer_class == 'CropRandomizer':
#                         modality.obs_randomizer_kwargs.crop_height = ch
#                         modality.obs_randomizer_kwargs.crop_width = cw

#         # init global state
#         ObsUtils.initialize_obs_utils_with_config(config)

#         # load model
#         policy: PolicyAlgo = algo_factory(
#                 algo_name=config.algo_name,
#                 config=config,
#                 obs_key_shapes=obs_key_shapes,
#                 ac_dim=action_dim,
#                 device='cpu',
#             )

#         obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
#         if obs_encoder_group_norm:
#             # replace batch norm with group norm
#             replace_submodules(
#                 root_module=obs_encoder,
#                 predicate=lambda x: isinstance(x, nn.BatchNorm2d),
#                 func=lambda x: nn.GroupNorm(
#                     num_groups=x.num_features//16, 
#                     num_channels=x.num_features)
#             )
#             # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
#         # obs_encoder.obs_randomizers['agentview_image']
#         if eval_fixed_crop:
#             replace_submodules(
#                 root_module=obs_encoder,
#                 predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
#                 func=lambda x: dmvc.CropRandomizer(
#                     input_shape=x.input_shape,
#                     crop_height=x.crop_height,
#                     crop_width=x.crop_width,
#                     num_crops=x.num_crops,
#                     pos_enc=x.pos_enc
#                 )
#             )

#         # create diffusion model
#         obs_feature_dim = obs_encoder.output_shape()[0]
#         input_dim = action_dim + obs_feature_dim
#         global_cond_dim = None
#         if obs_as_global_cond:
#             input_dim = action_dim
#             global_cond_dim = obs_feature_dim * n_obs_steps

#         model = ConditionalUnet1D(
#             input_dim=input_dim,
#             local_cond_dim=None,
#             global_cond_dim=global_cond_dim,
#             diffusion_step_embed_dim=diffusion_step_embed_dim,
#             down_dims=down_dims,
#             kernel_size=kernel_size,
#             n_groups=n_groups,
#             cond_predict_scale=cond_predict_scale
#         )

#         self.obs_encoder = obs_encoder
#         self.model = model
#         self.noise_scheduler = noise_scheduler
#         self.mask_generator = LowdimMaskGenerator(
#             action_dim=action_dim,
#             obs_dim=0 if obs_as_global_cond else obs_feature_dim,
#             max_n_obs_steps=n_obs_steps,
#             fix_obs_steps=True,
#             action_visible=False
#         )
#         self.normalizer = LinearNormalizer()
#         self.horizon = horizon
#         self.obs_feature_dim = obs_feature_dim
#         self.action_dim = action_dim
#         self.n_action_steps = n_action_steps
#         self.n_obs_steps = n_obs_steps
#         self.obs_as_global_cond = obs_as_global_cond
#         self.kwargs = kwargs

#         if num_inference_steps is None:
#             num_inference_steps = noise_scheduler.config.num_train_timesteps
#         self.num_inference_steps = num_inference_steps

#         print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
#         print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
#     # ========= inference  ============
#     def conditional_sample(self, 
#             condition_data, condition_mask,
#             local_cond=None, global_cond=None,
#             generator=None,
#             # keyword arguments to scheduler.step
#             **kwargs
#             ):
#         model = self.model
#         scheduler = self.noise_scheduler

#         trajectory = torch.randn(
#             size=condition_data.shape, 
#             dtype=condition_data.dtype,
#             device=condition_data.device,
#             generator=generator)
    
#         # set step values
#         scheduler.set_timesteps(self.num_inference_steps)

#         for t in scheduler.timesteps:
#             # 1. apply conditioning
#             trajectory[condition_mask] = condition_data[condition_mask]

#             # 2. predict model output
#             model_output = model(trajectory, t, 
#                 local_cond=local_cond, global_cond=global_cond)

#             # 3. compute previous image: x_t -> x_t-1
#             trajectory = scheduler.step(
#                 model_output, t, trajectory, 
#                 generator=generator,
#                 **kwargs
#                 ).prev_sample
        
#         # finally make sure conditioning is enforced
#         trajectory[condition_mask] = condition_data[condition_mask]        

#         return trajectory


#     def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """
#         obs_dict: must include "obs" key
#         result: must include "action" key
#         """
#         assert 'past_action' not in obs_dict # not implemented yet
#         # normalize input
#         nobs = self.normalizer.normalize(obs_dict)
#         value = next(iter(nobs.values()))
#         B, To = value.shape[:2]
#         T = self.horizon
#         Da = self.action_dim
#         Do = self.obs_feature_dim
#         To = self.n_obs_steps

#         # build input
#         device = self.device
#         dtype = self.dtype

#         # handle different ways of passing observation
#         local_cond = None
#         global_cond = None
#         if self.obs_as_global_cond:
#             # condition through global feature
#             this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
#             nobs_features = self.obs_encoder(this_nobs)
#             # reshape back to B, Do
#             global_cond = nobs_features.reshape(B, -1)
#             # empty data for action
#             cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
#             cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
#         else:
#             # condition through impainting
#             this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
#             nobs_features = self.obs_encoder(this_nobs)
#             # reshape back to B, To, Do
#             nobs_features = nobs_features.reshape(B, To, -1)
#             cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
#             cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
#             cond_data[:,:To,Da:] = nobs_features
#             cond_mask[:,:To,Da:] = True

#         # run sampling
#         nsample = self.conditional_sample(
#             cond_data, 
#             cond_mask,
#             local_cond=local_cond,
#             global_cond=global_cond,
#             **self.kwargs)
        
#         # unnormalize prediction
#         naction_pred = nsample[...,:Da]
#         action_pred = self.normalizer['action'].unnormalize(naction_pred)

#         # get action
#         start = To - 1
#         end = start + self.n_action_steps
#         action = action_pred[:,start:end]
        
#         result = {
#             'action': action,
#             'action_pred': action_pred
#         }
#         return result

#     # ========= training  ============
#     def set_normalizer(self, normalizer: LinearNormalizer):
#         self.normalizer.load_state_dict(normalizer.state_dict())

#     def compute_loss(self, batch):
#         # normalize input
#         assert 'valid_mask' not in batch
#         nobs = self.normalizer.normalize(batch['obs'])
#         nactions = self.normalizer['action'].normalize(batch['action'])
#         batch_size = nactions.shape[0]
#         horizon = nactions.shape[1]

#         # handle different ways of passing observation
#         local_cond = None
#         global_cond = None
#         trajectory = nactions
#         cond_data = trajectory
#         if self.obs_as_global_cond:
#             # reshape B, T, ... to B*T
#             this_nobs = dict_apply(nobs, 
#                 lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
#             nobs_features = self.obs_encoder(this_nobs)
#             # reshape back to B, Do
#             global_cond = nobs_features.reshape(batch_size, -1)
#         else:
#             # reshape B, T, ... to B*T
#             this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
#             nobs_features = self.obs_encoder(this_nobs)
#             # reshape back to B, T, Do
#             nobs_features = nobs_features.reshape(batch_size, horizon, -1)
#             cond_data = torch.cat([nactions, nobs_features], dim=-1)
#             trajectory = cond_data.detach()

#         # generate impainting mask
#         condition_mask = self.mask_generator(trajectory.shape)

#         # Sample noise that we'll add to the images
#         noise = torch.randn(trajectory.shape, device=trajectory.device)
#         bsz = trajectory.shape[0]
#         # Sample a random timestep for each image
#         timesteps = torch.randint(
#             0, self.noise_scheduler.config.num_train_timesteps, 
#             (bsz,), device=trajectory.device
#         ).long()
#         # Add noise to the clean images according to the noise magnitude at each timestep
#         # (this is the forward diffusion process)
#         noisy_trajectory = self.noise_scheduler.add_noise(
#             trajectory, noise, timesteps)
        
#         # compute loss mask
#         loss_mask = ~condition_mask

#         # apply conditioning
#         noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
#         # Predict the noise residual
#         pred = self.model(noisy_trajectory, timesteps, 
#             local_cond=local_cond, global_cond=global_cond)

#         pred_type = self.noise_scheduler.config.prediction_type 
#         if pred_type == 'epsilon':
#             target = noise
#         elif pred_type == 'sample':
#             target = trajectory
#         else:
#             raise ValueError(f"Unsupported prediction type {pred_type}")

#         loss = F.mse_loss(pred, target, reduction='none')
#         loss = loss * loss_mask.type(loss.dtype)
#         loss = reduce(loss, 'b ... -> b (...)', 'mean')
#         loss = loss.mean()
#         return loss
