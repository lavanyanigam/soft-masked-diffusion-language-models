#
# Copyright 2026- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import hydra.utils
import torch
import torch.nn.functional as F

# Local imports
import trainer_base
import utils

# General imports
from tqdm import tqdm
from trainer_base import sample_categorical
from transparency_head import TransparencyHead


class MDLM(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

    def _process_model_output(self, model_output, xt, sigma):
        del sigma
        model_output[:, :, self.mask_index] += self.neg_infinity

        # Normalize the model_output such that x.exp() is
        # a probability distribution over vocab_size.
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True
        )
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_index
        model_output[unmasked_indices] = self.neg_infinity
        model_output[unmasked_indices, xt[unmasked_indices]] = 0
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        del xt
        log_p_theta = torch.gather(
            input=log_x_theta, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)
        return log_p_theta * dalpha_t / (1 - alpha_t)

    def _get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)

        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))

        log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1

        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_index] = 0

        unmasked_score = self.neg_infinity * torch.ones_like(model_output)
        unmasked_score = torch.scatter(
            unmasked_score, -1, x[..., None], torch.zeros_like(unmasked_score[..., :1])
        )
        unmasked_score[:, :, self.mask_index] = -(log_k[:, None] * torch.ones_like(x))

        masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
        model_output = masked_score * masked_indices + unmasked_score * (
            1 - masked_indices
        )
        return model_output.exp()


class MDLM_SM(MDLM):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        # Initialize transparency head for soft feedback
        self.tran_head = TransparencyHead(
            mask_token_id=self.mask_index, trans_args=config.algo.tran_head
        )
        if not self.tran_head.learnable:
            for param in self.tran_head.parameters():
                param.requires_grad = False
        # Realized sm_prob from the most recent training step (for logging
        # only; plain attr so it never enters state_dict / EMA).
        self._last_sm_prob = None
        self._last_slerp_norm = None
        self._last_standard_norm = None

        self.register_buffer("initial_nll", torch.tensor(-1.0))
        self.register_buffer("current_nll_ema", torch.tensor(-1.0))
        self.register_buffer("R_ema", torch.tensor(-1.0))

    def _eval_mode(self):
        self.tran_head.eval()
        return super()._eval_mode()

    def _train_mode(self):
        self.tran_head.train()
        return super()._train_mode()

    def configure_optimizers(self):
        """
        Configures the optimizer with separate parameter groups for the main model
        and the TransparencyHead module.
        """
        # Separate the parameters into two groups
        special_lr_params = []
        main_params = []

        for name, param in self.named_parameters():
            # Check if the parameter belongs to the TransparencyHead module
            if name.startswith("tran_head."):
                special_lr_params.append(param)
            else:
                main_params.append(param)

        # Create the parameter groups with different learning rates
        param_groups = []
        if main_params:
            param_groups.append({"params": main_params, "lr": self.config.optim.lr})
        if special_lr_params:
            param_groups.append(
                {"params": special_lr_params, "lr": self.config.optim.tran_head_lr}
            )

        # Instantiate the optimizer with the parameter groups
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )

        # Instantiate the learning rate scheduler
        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "val/loss",
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler_dict]

    def training_step(self, batch, batch_idx):
        # Run the forward/loss first so the realized-interpolation stats below
        # reflect the current step.
        loss = super().training_step(batch, batch_idx)

        # Log the learnable transparency parameters.
        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        self.log(
            "transparency/scale",
            self.tran_head.scale.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        if (
            hasattr(self.tran_head, "last_effective_centre")
            and self.tran_head.last_effective_centre is not None
        ):
            self.log(
                "transparency/centre",
                self.tran_head.last_effective_centre.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
        else:
            self.log(
                "transparency/centre",
                self.tran_head.centre.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
        self.log(
            "transparency/steepness",
            self.tran_head.steepness.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        self.log(
            "transparency/temperature",
            self.tran_head.temperature.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

        # Log the realized interpolation behavior (set during the head's forward).
        if self.tran_head.last_lambda_mean is not None:
            self.log(
                "transparency/lambda_mean",
                self.tran_head.last_lambda_mean.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        if self.tran_head.last_lambda_std is not None:
            self.log(
                "transparency/lambda_std",
                self.tran_head.last_lambda_std.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        if self.tran_head.last_slerp_angle_mean is not None:
            self.log(
                "transparency/slerp_angle_mean",
                self.tran_head.last_slerp_angle_mean.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        if getattr(self.tran_head, "last_mean_disagreement_angle", None) is not None:
            self.log(
                "transparency/mean_disagreement_angle",
                self.tran_head.last_mean_disagreement_angle.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        if self.tran_head.last_feedback_norm_mean is not None:
            self.log(
                "transparency/feedback_embedding_norm_mean",
                self.tran_head.last_feedback_norm_mean.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        if self.tran_head.last_feedback_norm_std is not None:
            self.log(
                "transparency/feedback_embedding_norm_std",
                self.tran_head.last_feedback_norm_std.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        if self.tran_head.last_raw_lerp_norm_mean is not None:
            self.log(
                "transparency/raw_lerp_norm_mean",
                self.tran_head.last_raw_lerp_norm_mean.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        if self._last_slerp_norm is not None:
            self.log(
                "transparency/slerp_embedding_norm",
                self._last_slerp_norm.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
        if self._last_standard_norm is not None:
            self.log(
                "transparency/standard_embedding_norm",
                self._last_standard_norm.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

        # Log feedback norm histogram only at step 5000 (and every 5000 steps)
        if (
            self.global_step > 0
            and self.global_step % 5000 == 0
            and getattr(self.tran_head, "last_feedback_norms", None) is not None
        ):
            try:
                if (
                    hasattr(self, "logger")
                    and hasattr(self.logger, "experiment")
                    and hasattr(self.logger.experiment, "log")
                ):
                    import wandb

                    if isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
                        self.logger.experiment.log(
                            {
                                "transparency/feedback_norm_histogram": wandb.Histogram(
                                    self.tran_head.last_feedback_norms.cpu()
                                    .float()
                                    .numpy()
                                )
                            },
                            step=self.global_step,
                        )
            except Exception:
                pass

        # Log the current (possibly ramping) sm_prob gate probability. Set in
        # `nll()`; deterministic given global_step, so identical on every rank
        # (no sync needed).
        if self._last_sm_prob is not None:
            self.log(
                "transparency/sm_prob",
                self._last_sm_prob,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

        if getattr(self.config.algo.tran_head, "reliability_conditioned", False):
            self.log(
                "transparency/reliability_ema",
                self.R_ema.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
            self.log(
                "transparency/reliability_multiplier",
                self._get_reliability_multiplier(),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

        return loss

    def _get_reliability_multiplier(self):
        if not getattr(self.config.algo.tran_head, "reliability_conditioned", False):
            return 1.0

        r_ema = self.R_ema.item()
        if r_ema < 0:
            return 0.0

        r_start = float(getattr(self.config.algo.tran_head, "reliability_start", 0.05))
        r_full = float(getattr(self.config.algo.tran_head, "reliability_full", 0.25))

        if r_full <= r_start:
            return 1.0

        u = max(0.0, min(1.0, (r_ema - r_start) / (r_full - r_start)))
        r = u * u * (3.0 - 2.0 * u)
        return r

    def _current_sm_prob(self):
        """Effective soft-mask gate probability for the CURRENT training step.

        Linearly ramps from 0 at `global_step=0` up to `optim.sm_prob` at
        `global_step=optim.sm_prob_warmup_steps`, then holds constant at
        `optim.sm_prob`. `sm_prob_warmup_steps<=0` disables the ramp (returns
        `optim.sm_prob` unconditionally), which is the default and preserves
        behavior for any config that doesn't set it.

        This is independent of `algo.tran_head.fixed_lambda` (fixed vs. learnt
        lambda) — it only decides whether the two-pass soft-mask forward runs
        at all this step; the transparency head's lambda mode governs how much
        of the fed-back prediction is mixed in once that gate is open.
        """
        target = self.config.optim.sm_prob
        if getattr(self.config.algo.tran_head, "reliability_conditioned", False):
            r = self._get_reliability_multiplier()
            target = target * r

        warmup_steps = getattr(self.config.optim, "sm_prob_warmup_steps", 0)
        if warmup_steps and warmup_steps > 0:
            return target * min(1.0, self.global_step / warmup_steps)
        return target

    def forward(self, xt, sigma, log_p_x0=None):
        """
        Performs a forward pass with the option of using soft-masking.

        Args:
            xt: The input tensor of token ids.
            sigma: The noise level for the current timestep.
            log_p_x0: The model output from the previous step, used for feedback.

        Returns:
            The output logits from the model.
        """
        sigma_processed = self._process_sigma(sigma)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            if log_p_x0 is not None:
                # If previous predictions are available, create a soft-masked input
                current_nll_val = (
                    self.current_nll_ema.item()
                    if hasattr(self, "current_nll_ema")
                    and self.current_nll_ema.item() >= 0
                    else None
                )
                initial_nll_val = (
                    self.initial_nll.item()
                    if hasattr(self, "initial_nll") and self.initial_nll.item() >= 0
                    else None
                )

                r_val = self._get_reliability_multiplier() if self.training else 1.0

                if self.tran_head.transparency_alg in (
                    "slerp_sm",
                    "slerp_euclid_mean",
                    "lerp_renorm",
                    "mixinputs_with_topk",
                ):
                    # Soft feedback works in embedding space / requires embedding table for norm tracking
                    embedding_matrix = self.backbone.vocab_embed.embedding
                    p_x0_sm = self.tran_head(
                        xt,
                        log_p_x0,
                        embedding_matrix=embedding_matrix,
                        current_nll=current_nll_val,
                        initial_nll=initial_nll_val,
                        r_multiplier=r_val,
                    )
                    if isinstance(p_x0_sm, torch.Tensor):
                        self._last_slerp_norm = p_x0_sm.norm(dim=-1).mean().detach()
                else:
                    p_x0_sm = self.tran_head(
                        xt,
                        log_p_x0,
                        current_nll=current_nll_val,
                        initial_nll=initial_nll_val,
                        r_multiplier=r_val,
                    )
                model_output = self.backbone(p_x0_sm, sigma=sigma_processed)
            else:
                # Standard forward pass if no previous prediction is available
                self._last_standard_norm = (
                    self.backbone.vocab_embed.embedding[xt].norm(dim=-1).mean().detach()
                )
                model_output = self.backbone(xt, sigma=sigma_processed)

        return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)

    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=False):
        """
        Calculates the negative log-likelihood with find_unused_parameters=True.
        """
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            t += 1 / self.T
        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        xt = self.q_xt(x0, alpha_t)

        # --- DDP-safe soft-mask gate (training only) -----------------------------
        # Under data-parallel TRAINING every rank must take the same branch here: if
        # some ranks exercise the tran_head sub-graph and others do not, the tran_head
        # params receive gradients on a subset of ranks only and the DDP gradient
        # all-reduce desyncs (wrong averaging; hangs in stricter setups). The naive
        # `torch.rand(1)` draw and the per-rank `t.mean()` band test both diverge
        # across ranks, so during training we make both decisions rank-identical:
        #   (1) Bernoulli gate seeded by the global step  -> same on every rank, no comms.
        #   (2) time-band test on the GLOBAL mean of t (all-reduced) -> all ranks agree.
        #
        # Validation runs NO backward, so there is no gradient-sync constraint at eval.
        # We therefore keep the original per-batch local decision during eval. This
        # matters: the global mean of t is tightly concentrated at ~0.5, so an
        # all-reduced band test is in-band on essentially every val batch, whereas the
        # per-batch mean legitimately falls out of band on some. Forcing soft-masking
        # onto every val batch inflates perplexity while the tran_head is untrained
        # (e.g. at init), so we must NOT do it here. It also avoids an unnecessary
        # collective at eval that can deadlock on an uneven validation shard.
        if train_mode:
            gate_gen = torch.Generator()
            gate_gen.manual_seed(
                int(self.config.seed) * 1_000_003 + int(self.global_step)
            )
            sm_prob_now = self._current_sm_prob()
            self._last_sm_prob = sm_prob_now
            sm_gate = torch.rand(1, generator=gate_gen).item() < sm_prob_now
            t_mean = t.mean()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                t_mean = t_mean.clone()
                torch.distributed.all_reduce(t_mean, op=torch.distributed.ReduceOp.SUM)
                t_mean = t_mean / torch.distributed.get_world_size()
            t_mean = t_mean.item()
        else:
            # Eval: original behavior — always consider soft-mask, local per-batch band.
            sm_gate = True
            t_mean = t.mean().item()

        in_band = self.config.optim.sm_t_min <= t_mean <= self.config.optim.sm_t_max
        use_soft_mask = sm_gate and in_band

        if use_soft_mask:
            # Pass 1: Get predictions for feedback. No grad needed — avoids building
            # the full backward graph for a forward whose output is only used as data.
            with torch.no_grad():
                log_x_theta_pass1 = self.forward(xt, sigma=sigma)
            log_x_theta_for_reliability = log_x_theta_pass1
            # Pass 2: Main pass that computes the gradients and sets _last_slerp_norm.
            log_x_theta = self.forward(xt, sigma=sigma, log_p_x0=log_x_theta_pass1)
        else:
            # --- Standard Path (gate off, or batch is out of band) ---
            log_x_theta = self.forward(xt, sigma=sigma)
            log_x_theta_for_reliability = log_x_theta

        utils.print_nans(log_x_theta, "model_output")

        loss = self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt,
            x0=x0,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            low_var=train_mode and self.loss_type == "low_var",
        )

        if train_mode:
            loss_val = loss.detach().mean()
            if self.initial_nll.item() < 0:
                self.initial_nll.copy_(loss_val)
                self.current_nll_ema.copy_(loss_val)
            else:
                self.current_nll_ema.copy_(
                    0.99 * self.current_nll_ema + 0.01 * loss_val
                )

            # Update Reliability Schedule if enabled
            if getattr(self.config.algo.tran_head, "reliability_conditioned", False):
                reliability_beta = getattr(
                    self.config.algo.tran_head, "reliability_beta", 0.99
                )
                mask_pos = xt == self.mask_index
                num_masked = mask_pos.sum()

                # Compute correct log-probabilities and probabilities
                log_p_theta = torch.gather(
                    input=log_x_theta_for_reliability.detach(),
                    dim=-1,
                    index=x0[:, :, None],
                ).squeeze(-1)
                p_theta = torch.exp(log_p_theta)
                sum_p_theta = (
                    p_theta[mask_pos].sum()
                    if num_masked > 0
                    else torch.tensor(0.0, device=p_theta.device)
                )

                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    stats = torch.stack(
                        [sum_p_theta, num_masked.float().to(sum_p_theta.device)]
                    )
                    torch.distributed.all_reduce(
                        stats, op=torch.distributed.ReduceOp.SUM
                    )
                    sum_p_theta_all = stats[0]
                    num_masked_all = stats[1]
                    if num_masked_all > 0:
                        R_batch = sum_p_theta_all / num_masked_all
                    else:
                        R_batch = torch.tensor(0.0, device=p_theta.device)
                else:
                    if num_masked > 0:
                        R_batch = sum_p_theta / num_masked
                    else:
                        R_batch = torch.tensor(0.0, device=p_theta.device)

                # Update R_ema
                if self.R_ema.item() < 0:
                    self.R_ema.copy_(R_batch)
                else:
                    self.R_ema.copy_(
                        reliability_beta * self.R_ema
                        + (1.0 - reliability_beta) * R_batch
                    )

        return loss

    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        """Generate samples from the model."""

        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self.prior_sample(num_samples, self.num_tokens)

        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        p_x0_cache = None  # Standard cache
        log_p_x0_cache_sm = None  # Use log-probabilities for the SM cache

        confident_score = (
            -torch.ones_like(x, device=self.device).to(torch.bfloat16) * torch.inf
        )
        for i in tqdm(range(num_steps)):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "ddpm_cache":
                log_p_x0_cache_sm, p_x0_cache, x_next, confident_score = (
                    self._ddpm_caching_update(
                        x,
                        t,
                        dt,
                        p_x0=p_x0_cache,
                        log_p_x0_sm=log_p_x0_cache_sm,
                        conf=confident_score,
                    )
                )
                x = x_next
            else:
                x = self._analytic_update(x, t, dt)

        if self.config.sampling.noise_removal:
            min_t = timesteps[-1].item()
            t = min_t * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "analytic":
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self._sigma_from_alphat(self.noise(t)[1])
                # Use the final feedback pass for noise removal, but only feed back the
                # cached prediction when this final timestep is inside the soft-mask
                # band. Noise removal runs at min_t (≈eps), which is normally below
                # sm_t_min, so by default this is a standard forward.
                nr_time = t.reshape(-1)[0].item()
                nr_in_band = (
                    self.config.optim.sm_t_min <= nr_time <= self.config.optim.sm_t_max
                )
                final_feedback = log_p_x0_cache_sm if nr_in_band else None
                final_log_p_x0 = self.forward(
                    x, unet_conditioning, log_p_x0=final_feedback
                )
                x = final_log_p_x0.argmax(dim=-1)
        return x

    def _ddpm_caching_update(self, x, t, dt, p_x0=None, log_p_x0_sm=None, conf=None):
        """
        DDPM caching update borrowed and adapted from ReMDM.
        """
        assert self.config.noise.type == "log-linear"
        sigma_t = self._sigma_from_alphat(self.noise(t)[1])
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            # Time-band gate: only feed the cached prediction back (soft-mask path)
            # when the current timestep is inside [sm_t_min, sm_t_max]; otherwise run
            # a standard forward, matching the original soft-masking paper.
            time_val = t.reshape(-1)[0].item()
            in_band = (
                self.config.optim.sm_t_min <= time_val <= self.config.optim.sm_t_max
            )
            feedback = log_p_x0_sm if in_band else None
            log_p_x0_sm = self.forward(x, sigma_t, feedback)
            p_x0 = log_p_x0_sm.exp()
            if self.config.sampling.p_nucleus < 1:
                sorted_probs, sorted_indices = torch.sort(
                    p_x0, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                top_p_mask = cumulative_probs <= self.config.sampling.p_nucleus
                top_p_mask[..., 0] = True
                nucleus_probs = sorted_probs * top_p_mask
                nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
                p_x0 = torch.zeros_like(p_x0).scatter_(
                    -1, sorted_indices, nucleus_probs
                )

        assert move_chance_t.ndim == p_x0.ndim

        if self.config.sampling.sampler == "mdlm":
            q_xs = p_x0 * (move_chance_t - move_chance_s)
            q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
            _x = sample_categorical(q_xs)
            copy_flag = (x != self.mask_index).to(x.dtype)
            xs = copy_flag * x + (1 - copy_flag) * _x
        elif self.config.sampling.sampler == "remdm-cap":
            alpha_t = (1 - move_chance_t)[0].item()
            alpha_s = (1 - move_chance_s)[0].item()
            if alpha_t > 0:
                sigma = min(self.config.sampling.eta, (1 - alpha_s) / alpha_t)
            else:
                sigma = self.config.sampling.eta
            q_xs = p_x0 * (1 - sigma)
            q_xs[..., self.mask_index] = sigma
            q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
            q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (
                1 - alpha_t
            )
            copy_flag = (x != self.mask_index).to(torch.bool)
            q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
            xs = sample_categorical(q_xs)
        elif self.config.sampling.sampler == "remdm-loop":
            time = t[0].item()
            # compute alpha_t and alpha_s
            if time > self.config.sampling.t_on:
                move_chance_t = (
                    1
                    - (1 - t)
                    * self.config.sampling.alpha_on
                    / (1 - self.config.sampling.t_on)
                )[:, None, None]
                move_chance_s = (
                    1
                    - (1 - t + dt)
                    * self.config.sampling.alpha_on
                    / (1 - self.config.sampling.t_on)
                )[:, None, None]
            elif time <= self.config.sampling.t_off:
                move_chance_t = (
                    t * (1 - self.config.sampling.alpha_on) / self.config.sampling.t_off
                )[:, None, None]
                move_chance_s = (
                    (t - dt)
                    * (1 - self.config.sampling.alpha_on)
                    / self.config.sampling.t_off
                )[:, None, None]
            else:
                move_chance_t, move_chance_s = None, None
            # use MDLM
            if time > self.config.sampling.t_on or time <= self.config.sampling.t_off:
                q_xs = p_x0 * (move_chance_t - move_chance_s)
                q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
                _x = sample_categorical(q_xs)
                copy_flag = (x != self.mask_index).to(x.dtype)
                xs = copy_flag * x + (1 - copy_flag) * _x
            else:  # use ReMDM
                sigma = self.config.sampling.eta
                q_xs = p_x0 * (1 - sigma)
                q_xs[..., self.mask_index] = sigma
                q_xs_2 = p_x0 * (
                    (
                        self.config.sampling.alpha_on
                        - (1 - sigma) * self.config.sampling.alpha_on
                    )
                    / (1 - self.config.sampling.alpha_on)
                )
                q_xs_2[..., self.mask_index] = (
                    1
                    - self.config.sampling.alpha_on
                    - self.config.sampling.alpha_on * sigma
                ) / (1 - self.config.sampling.alpha_on)
                copy_flag = (x != self.mask_index).to(torch.bool)
                q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
                xs = sample_categorical(q_xs)

        # changed
        if torch.equal(xs, x) and not self.time_conditioning:
            p_x0_cache = p_x0
        else:
            p_x0_cache = None

        return log_p_x0_sm, p_x0_cache, xs, conf
