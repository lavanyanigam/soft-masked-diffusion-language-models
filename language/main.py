#
# Copyright 2026- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# General imports
import json
import os

# Local imports
import algo
import dataloader
import fsspec
import hydra
import lightning as L
import mauve
import omegaconf
import rich.syntax
import rich.tree
import torch
import utils

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
    if "hf" in config.algo.backbone:
        return diffusion_model(config, tokenizer=tokenizer).to("cuda")

    return diffusion_model.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config
    )


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve
            )

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            "{}/config_tree.txt".format(config.checkpointing.save_dir), "w"
        ) as fp:
            rich.print(tree, file=fp)


def _generate_samples(diffusion_model, config, logger, tokenizer):
    """
    Unconstrained generation with gen. perplexity and entropy
    computation
    """
    logger.info("Loading model")
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    samples = []
    entropies = []

    logger.info("Starting evaluation")
    for _ in range(config.sampling.num_sample_batches):
        print("Starting Batch ", _)
        sample = model.restore_model_and_sample(num_steps=config.sampling.steps)

        for i in range(config.loader.batch_size):
            row = sample[i]
            # Compute Entropy
            counts = torch.unique(row, return_counts=True, sorted=True)[1]
            entropies.append(
                torch.special.entr(counts.float() / counts.sum()).sum().item()
            )

            # Store text
            sample_text = tokenizer.batch_decode(row.unsqueeze(0))
            samples.append(sample_text[0])

    logger.info("Compute Generative Perplexity.")
    model.metrics.record_generative_perplexity(
        samples, config.model.length, model.device
    )
    gen_ppl = model.metrics.gen_ppl.compute().item()

    return samples, gen_ppl, entropies


def _eval_ppl(diffusion_model, config, logger, tokenizer):
    """
    Perplexity on evaluation dataset.
    """
    logger.info("Starting Perplexity Eval.")
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            if callback is not None:
                instantiated = hydra.utils.instantiate(callback)
                if instantiated is not None:
                    callbacks.append(instantiated)
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed
    )

    # Run actual validation
    trainer.validate(model, valid_ds)


def _train(diffusion_model, config, logger, tokenizer):
    """
    Training function
    """
    logger.info("Starting Training.")
    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            if callback is not None:
                instantiated = hydra.utils.instantiate(callback)
                if instantiated is not None:
                    callbacks.append(instantiated)

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)

    # Load pretrained model when finetuning/pretraining continuation
    if config.training.finetune_path != "":
        assert utils.fsspec_exists(config.training.finetune_path)

        # Create an instance of your new model architecture.
        model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

        # Load the state dictionary from the old checkpoint file.
        # weights_only=False is required: Lightning checkpoints are pickled Python
        # dicts, not pure-tensor files. Only load checkpoints from trusted sources.
        old_checkpoint = torch.load(
            config.training.finetune_path, map_location="cpu", weights_only=False
        )

        old_state_dict = old_checkpoint["state_dict"]

        # Load the old state_dict into the new model with strict=False.
        missing_keys, unexpected_keys = model.load_state_dict(
            old_state_dict, strict=False
        )

        logger.warning(
            f"Weights loaded with {len(missing_keys)} missing keys (new parameters): {missing_keys}"
        )
        logger.warning(f"And {len(unexpected_keys)} unexpected keys: {unexpected_keys}")
    else:
        model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )

    # Run actual training
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    """Main entry point for training."""

    # Seeding
    L.seed_everything(config.seed)

    # Printing configs
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
    if config.algo.name == "mdlm":
        diffusion_model = algo.MDLM
    elif config.algo.name == "mdlm_sm":
        diffusion_model = algo.MDLM_SM
    else:
        raise ValueError(f"Invalid algorithm name: {config.algo.name}")
    kwargs = {
        "diffusion_model": diffusion_model,
        "config": config,
        "tokenizer": tokenizer,
        "logger": logger,
    }
    if config.mode == "sample_eval":
        # Unsconstrained generation experiment
        samples, gen_ppl, entropies = _generate_samples(**kwargs)
        human_references = dataloader.load_human_references(
            config.model.length,
            len(samples),
            config.seed,
            config.data.cache_dir,
            logger,
            tokenizer,
            config,
        )

        logger.info("Compute MAUVE score.")
        results = mauve.compute_mauve(
            p_text=human_references,
            q_text=samples,
            device_id=0,
            max_text_length=1024,
            verbose=False,
            batch_size=config.eval.perplexity_batch_size,
        )
        mauve_score = results.mauve

        result_dict = {
            "gen_ppl": gen_ppl,
            "entropy": sum(entropies) / len(entropies),
            "MAUVE": mauve_score,
            # Keep the generations. Without these two keys every sample_eval run
            # discards all 5000 decoded sequences and retains three floats, which
            # is why the human-evaluation set had to be rebuilt from older output
            # files. run_modal_mauve.py's docstring has always claimed these are
            # written; now they are.
            "entropies": entropies,
            "text_samples": samples,
        }
        with open(config.sampling.generated_seqs_path, "w") as file:
            json.dump(result_dict, file, indent=4)
    elif config.mode == "ppl_eval":
        _eval_ppl(**kwargs)
    else:
        _train(**kwargs)


if __name__ == "__main__":
    main()
