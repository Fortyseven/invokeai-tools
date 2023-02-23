#!/usr/bin/env python3

# This is a reworking of the training notebook from the stable diffusion repo

import shutil
from torch.utils.data import Dataset
import argparse
import itertools
import json
import math
import os
import random
import sys

from PIL import Image
from rich import print
from rich.console import Console
from rich.prompt import IntPrompt, Prompt

console = Console()

DEFAULT_INPUT_IMAGE_PATH = "images"
DEFAULT_OUTPUT_IMAGE_PATH = "embeddings"
DEFAULT_TRAINING_STEPS = 3000
PROJECT_CONFIG_FILE = "project.json"

# pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"
# pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-base"
# pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
# pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

config = {
    'image_path': None,             # not in config
    'output_path': None,             # not in config
    'what_to_teach': None,
    'placeholder_token': None,
    'max_train_steps': None,
    'initializer_token': None,
    'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
}

IMAGENET_TEMPLATES_SMALL = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

IMAGENET_STYLE_TEMPLATES_SMALL = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class ModeTrain():
    class TextualInversionDataset(Dataset):

        def __init__(
            self,
            config,
            data_root,
            tokenizer,
            learnable_property="object",  # [object, style]
            size=512,
            repeats=100,
            interpolation="bicubic",
            flip_p=0.5,
            set="train",
            center_crop=False,
        ):
            global np, console
            from PIL import Image
            from torchvision import transforms
            import numpy as np

            self.config = config
            self.console = console
            self.data_root = data_root
            self.tokenizer = tokenizer
            self.learnable_property = learnable_property
            self.size = size
            self.placeholder_token = self.config['placeholder_token']
            self.center_crop = center_crop
            self.flip_p = flip_p

            self.image_paths = [os.path.join(
                self.data_root, file_path) for file_path in os.listdir(self.data_root)]

            self.num_images = len(self.image_paths)
            self._length = self.num_images

            if set == "train":
                self._length = self.num_images * repeats

            self.interpolation = {
                "linear": Image.Resampling.BILINEAR,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
                "lanczos": Image.Resampling.LANCZOS,
            }[interpolation]

            self.templates = IMAGENET_STYLE_TEMPLATES_SMALL if learnable_property == "style" else IMAGENET_TEMPLATES_SMALL

            self.flip_transform = transforms.RandomHorizontalFlip(
                p=self.flip_p
            )

        def __len__(self):
            return self._length

        def __getitem__(self, i):
            example = {}
            image = Image.open(self.image_paths[i % self.num_images])

            if not image.mode == "RGB":
                image = image.convert("RGB")

            placeholder_string = self.config['placeholder_token']
            text = random.choice(self.templates).format(placeholder_string)

            example["input_ids"] = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8)

            if self.center_crop:
                crop = min(img.shape[0], img.shape[1])
                h, w, = (
                    img.shape[0],
                    img.shape[1],
                )
                img = img[(h - crop) // 2: (h + crop) // 2,
                          (w - crop) // 2: (w + crop) // 2]

            image = Image.fromarray(img)
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)

            image = self.flip_transform(image)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)

            example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
            return example

    def __init__(self, config, args):
        global console, torch, StableDiffusionPipeline, F, tqdm, Accelerator, accelerate

        import accelerate
        import torch
        import torch.nn.functional as F
        import torch.utils.checkpoint

        import warnings

        warnings.filterwarnings('ignore')

        from accelerate import Accelerator
        # from accelerate.logging import get_logger
        from diffusers import (AutoencoderKL, DDPMScheduler, PNDMScheduler,
                               StableDiffusionPipeline, UNet2DConditionModel)
        from tqdm.auto import tqdm
        from transformers import CLIPTextModel, CLIPTokenizer

        self.config = config
        self.args = args
        self.console = console

        self.console.log("Initializing textual inversion dataset...")

        # self.logger = get_logger(__name__)

        # Load the tokenizer and add the placeholder token as a additional special token.
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config["pretrained_model_name_or_path"],
            subfolder="tokenizer",
        )

        # Add the placeholder token in tokenizer
        num_added_tokens = self.tokenizer.add_tokens(
            self.config['placeholder_token'])
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {self.config['placeholder_token']}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        # Convert the initializer_token, placeholder token to ids
        token_ids = self.tokenizer.encode(
            self.config['initializer_token'],
            add_special_tokens=False
        )

        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]

        self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
            self.config['placeholder_token']
        )

        # Load models and create wrapper for stable diffusion
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config["pretrained_model_name_or_path"],
            subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.config["pretrained_model_name_or_path"],
            subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config["pretrained_model_name_or_path"],
            subfolder="unet"
        )

        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        token_embeds[self.placeholder_token_id] = token_embeds[initializer_token_id]

        # Freeze self.vae and self.unet
        self.freeze_params(self.vae.parameters())
        self.freeze_params(self.unet.parameters())

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            self.text_encoder.text_model.encoder.parameters(),
            self.text_encoder.text_model.final_layer_norm.parameters(),
            self.text_encoder.text_model.embeddings.position_embedding.parameters(),
        )

        self.freeze_params(params_to_freeze)

        self.train_dataset = self.TextualInversionDataset(
            self.config,
            data_root=self.config['image_path'],
            tokenizer=self.tokenizer,
            size=self.vae.sample_size,
            repeats=100,
            learnable_property=self.config['what_to_teach'],
            center_crop=False,
            set="train",
        )

        self.noise_scheduler = DDPMScheduler.from_config(
            self.config["pretrained_model_name_or_path"],
            subfolder="scheduler"
        )

        self.hyperparameters = {
            "learning_rate": 5e-04,
            "scale_lr": True,
            "max_train_steps": self.config['max_train_steps'],
            "save_steps": 250,
            "train_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
            "mixed_precision": "fp16",
            "seed": 42,
            "output_dir": self.config['output_path']
        }

    def freeze_params(self, params):
        for param in params:
            param.requires_grad = False

    def create_dataloader(self, train_batch_size=1):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True
        )

    def save_progress(self, text_encoder, placeholder_token_id, accelerator, save_path):
        self.console.log("Saving embeddings")

        learned_embeds = accelerator.unwrap_model(
            self.text_encoder
        ).get_input_embeddings().weight[placeholder_token_id]

        learned_embeds_dict = {
            self.config['placeholder_token']: learned_embeds.detach().cpu()
        }

        torch.save(learned_embeds_dict, save_path)

    def training_function(self):
        train_batch_size = self.hyperparameters["train_batch_size"]
        gradient_accumulation_steps = self.hyperparameters["gradient_accumulation_steps"]
        learning_rate = self.hyperparameters["learning_rate"]
        max_train_steps = self.hyperparameters["max_train_steps"]
        output_dir = self.hyperparameters["output_dir"]
        gradient_checkpointing = self.hyperparameters["gradient_checkpointing"]

        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=self.hyperparameters["mixed_precision"]
        )

        if gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable()
            self.unet.enable_gradient_checkpointing()

        self.train_dataloader = self.create_dataloader(train_batch_size)

        if self.hyperparameters["scale_lr"]:
            learning_rate = (
                learning_rate * gradient_accumulation_steps *
                train_batch_size * accelerator.num_processes
            )

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            # only optimize the embeddings
            self.text_encoder.get_input_embeddings().parameters(),
            lr=learning_rate,
        )

        self.text_encoder, optimizer, self.train_dataloader = accelerator.prepare(
            self.text_encoder, optimizer, self.train_dataloader
        )

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move self.vae and self.unet to device
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.unet.to(accelerator.device, dtype=weight_dtype)

        # Keep self.vae in eval mode as we don't train it
        self.vae.eval()
        # Keep self.unet in train mode to enable gradient checkpointing
        self.unet.train()

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / gradient_accumulation_steps)
        num_train_epochs = math.ceil(
            max_train_steps / num_update_steps_per_epoch)

        # Train!
        total_batch_size = train_batch_size * \
            accelerator.num_processes * gradient_accumulation_steps

        self.console.log("***** Running training *****")
        self.console.log(f"Num examples = {len(self.train_dataset)}")
        self.console.log(
            f"Instantaneous batch size per device = {train_batch_size}")
        self.console.log(
            f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.console.log(
            f"Gradient Accumulation steps = {gradient_accumulation_steps}")
        self.console.log(f"  Total optimization steps = {max_train_steps}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(max_train_steps),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(num_train_epochs):
            self.text_encoder.train()
            for step, batch in enumerate(self.train_dataloader):
                with accelerator.accumulate(self.text_encoder):
                    # Convert images to latent space
                    latents = self.vae.encode(
                        batch["pixel_values"]
                        .to(dtype=weight_dtype)
                    ).latent_dist.sample().detach()

                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)

                    bsz = latents.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(
                        batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = self.unet(noisy_latents, timesteps,
                                           encoder_hidden_states.to(weight_dtype)).sample

                    # self.noise_scheduler.config = "v_prediction"
                    # print("FUCK", self.noise_scheduler.config)

                    # HACK - NOTE - for some reason the noise scheduler does not contain a 'config' property,
                    # so I'm forcing the 'epsilon' behavior since the opposite doesn't work. This is a recent
                    # change since I've moved to conda, but even the old pip venv isn't working. So something
                    # is going on here that I can't account for.

                    # Get the target for loss depending on the prediction type
                    # if self.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                    # elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    #     target = self.noise_scheduler.get_velocity(
                    #         latents, noise, timesteps)
                    # else:
                    #     raise ValueError(
                    #         f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(noise_pred, target, reduction="none").mean(
                        [1, 2, 3]).mean()

                    accelerator.backward(loss)

                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if accelerator.num_processes > 1:
                        grads = self.text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = self.text_encoder.get_input_embeddings().weight.grad

                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(
                        len(self.tokenizer)) != self.placeholder_token_id

                    grads.data[index_grads_to_zero,
                               :] = grads.data[index_grads_to_zero, :].fill_(0)

                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % self.hyperparameters["save_steps"] == 0:
                        save_path = os.path.join(
                            output_dir, f"learned_embeds-step-{global_step}.bin")
                        self.save_progress(self.text_encoder, self.placeholder_token_id,
                                           accelerator, save_path)

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_step >= max_train_steps:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.config["pretrained_model_name_or_path"],
                text_encoder=accelerator.unwrap_model(self.text_encoder),
                tokenizer=self.tokenizer,
                vae=self.vae,
                unet=self.unet,
            )
            pipeline.save_pretrained(output_dir)

            # Also save the newly trained embeddings
            save_path = os.path.join(output_dir, f"learned_embeds.bin")

            self.save_progress(
                self.text_encoder,
                self.placeholder_token_id,
                accelerator,
                save_path
            )

    def run(self):
        accelerate.notebook_launcher(
            self.training_function,
            num_processes=1
        )

        for param in itertools.chain(self.unet.parameters(), self.text_encoder.parameters()):
            if param.grad is not None:
                del param.grad  # free some memory
            torch.cuda.empty_cache()

        self.console.log("Done!")
        self.console.bell()


class ModeInit():
    def __init__(self, config, args):
        global console
        self.config = config
        self.args = args
        self.console = console

    def run(self):
        self.config['image_path'] = os.path.join(
            self.args.project_path,
            DEFAULT_INPUT_IMAGE_PATH
        )

        self.config['image_path'] = Prompt.ask(
            "Path to training images? (512x512 *.png,jpg)",
            default=self.config['image_path']
        )

        self.config['output_path'] = os.path.join(
            self.args.project_path,
            DEFAULT_OUTPUT_IMAGE_PATH
        )

        self.config['output_path'] = Prompt.ask(
            "Path to save embeddings?",
            default=self.config['output_path']
        )

        self.config['what_to_teach'] = Prompt.ask(
            "What kind of project is this?",
            choices=['object', 'concept']
        )

        while not self.config['placeholder_token']:
            self.config['placeholder_token'] = Prompt.ask(
                f"What should the keyword be for this {self.config['what_to_teach']}?",
                default=f'<{self.args.project_path}>'
            )

        while not self.config['max_train_steps']:
            self.config['max_train_steps'] = IntPrompt.ask(
                "How many steps should be trained for?",
                default=DEFAULT_TRAINING_STEPS
            )

        while not self.config['initializer_token']:
            self.config['initializer_token'] = Prompt.ask(
                f"Choose a term or word describing this {self.config['what_to_teach']}. (e.g. 'person', 'painting')",
                default='person'
            )

        while not self.config['pretrained_model_name_or_path']:
            self.config["pretrained_model_name_or_path"] = Prompt.ask(
                "Pretrained model name or path?",
                default=self.config["pretrained_model_name_or_path"]
            )

        os.makedirs(self.args.project_path, exist_ok=True)

        # save config to json file
        config_file = os.path.join(self.args.project_path, PROJECT_CONFIG_FILE)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)

        self.console.log(f"Wrote config to '{config_file}'...")
        self.console.log(self.config)

        # create input/output directories
        os.makedirs(self.config['image_path'], exist_ok=True)
        os.makedirs(self.config['output_path'], exist_ok=True)

        self.console.log(
            f"Place training images in '[bold]{self.config['image_path']}[/bold]'...")
        self.console.log(
            f"Embed checkpoints will be in '[bold]{self.config['output_path']}[/bold]'...")


class ModeInstall():
    def __init__(self, config, args):
        global console
        self.config = config
        self.args = args
        self.console = console

    def run(self):
        # get INVOKEAI_ROOT envrionment variable
        INVOKEAI_ROOT = os.environ.get('INVOKEAI_ROOT')

        # check if INVOKEAI_ROOT exists
        if not INVOKEAI_ROOT or not os.path.exists(INVOKEAI_ROOT):
            self.console.log(
                f'[red]INVOKEAI_ROOT environment variable not set or invalid:[/red] {INVOKEAI_ROOT}')
            os._exit(1)

        embed_path = os.path.join(INVOKEAI_ROOT, 'embeddings')

        # if path exists
        if not os.path.exists(embed_path):
            self.console.log(
                f'[red]Invoke root exists, but embeddings path not found:[/red] {embed_path}')
            os._exit(1)

        final_embed_path = os.path.join(
            self.config['output_path'],
            'learned_embeds.bin'
        )

        if not os.path.exists(final_embed_path):
            self.console.log(
                f'[red]Final embedding file not found -- did training finish?[/red] {final_embed_path}')
            os._exit(1)

        destination_embed_name = os.path.join(
            embed_path,
            f"{self.args.project_path}-{self.config['max_train_steps']}.bin"
        )

        if (os.path.exists(destination_embed_name)):
            self.console.log(
                f'[red]Embedding file already exists:[/red] {destination_embed_name}'
            )
            if not Prompt.ask("Overwrite?", choices=['yes', 'no'], default='no') == 'yes':
                self.console.log('Aborted')
                os._exit(1)

        self.console.log(
            f"[green]Installing[/green] {destination_embed_name}...")

        # do it
        try:
            shutil.copyfile(final_embed_path, destination_embed_name)
        except Exception as e:
            self.console.log(f'[red]Error copying file:[/red] {e}')
            os._exit(1)

        self.console.log(
            f"[green]Done.[/green]")


def loadConfig(project_path):
    global config

    config_path = os.path.join(project_path, PROJECT_CONFIG_FILE)

    if not os.path.exists(config_path):
        raise Exception(f'{PROJECT_CONFIG_FILE} not found in project path')
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['image_path'] = os.path.join(project_path, DEFAULT_INPUT_IMAGE_PATH)

    os.makedirs(config['image_path'], exist_ok=True)
    os.makedirs(config['output_path'], exist_ok=True)

    return config


MODES = {
    'init': ModeInit,
    'train': ModeTrain,
    'install': ModeInstall


    # 'publish': ModePublish
}


def main():
    global console

    parser = argparse.ArgumentParser(
        description='Swiss army knife for InvokeAI-generated PNG files')

    # accept a single-word command, required
    parser.add_argument('command', choices=[
                        'init', 'train', 'publish', 'install'], help='todo')

    # one or many files
    parser.add_argument('project_path',
                        help='specifies the training project path')

    parser.add_argument('--dry-run', '-d',
                        action='store_true', help='Do not actually rename files, just show what might have happened.')

    parser.add_argument('--verbose', '-v',
                        action='store_true', help='Generate more verbose output')

    # parser.epilog = "Format string ids are pulled directly from the sub-keys of the `image` property of invoke metadata. Also available: {prompt-50}, a truncated prompt."

    if len(sys.argv) == 1:
        parser.print_help()
        os._exit(1)

    args = parser.parse_args()

    if args.command in MODES:
        # for these operations we'll require the path to exist, so we can load the config
        if args.command in ['train', 'publish', 'install']:
            if not os.path.exists(args.project_path):
                console.log('Cannot open project path: ', args.project_path)
                os._exit(1)
            loadConfig(args.project_path)
            console.log('Project path: ', args.project_path)
            console.log(config)
        else:

            if os.path.exists(os.path.join(args.project_path, PROJECT_CONFIG_FILE)):
                console.log('Project already exists: ', args.project_path)
                os._exit(1)

        mode = MODES[args.command](config, args)
        mode.run()
    else:
        console.log('Unimplemented command: ', args.command)
        os._exit(1)


if __name__ == '__main__':
    main()
