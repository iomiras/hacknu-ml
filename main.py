import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# %%
import torch
from pathlib import Path
from PIL import Image, ImageEnhance
import json
from datetime import datetime
import gc
import numpy as np
import os
import sys

# %%
os.system("pip uninstall -y bitsandbytes peft diffusers transformers accelerate huggingface_hub -q")

# # ============================================================================
# # CELL 2: Clean install with correct versions
# # ============================================================================
print("📦 Installing dependencies...")
os.system("pip install -q huggingface_hub==0.20.3")
os.system("pip install -q transformers==4.36.2")
os.system("pip install -q diffusers==0.25.1")
os.system("pip install -q accelerate==0.26.1")
os.system("pip install -q peft==0.8.2")
os.system("pip install -q safetensors==0.4.1")
os.system("pip install openai")
os.system("pip install -q opencv-python pillow -q")

# Block bitsandbytes
sys.modules['bitsandbytes'] = None

# %%
class LoRAConfig:
    """Optimized configuration for cartoon/3D character training"""

    # Paths
    INSTANCE_DIR = "datasetAldar"
    OUTPUT_DIR = "aldar_kose_lora_test"

    # Base model - USE STYLIZED MODEL!
    MODEL_NAME = "prompthero/openjourney-v4"  # Better for 3D cartoon/illustrated style
    # Alternative: "dreamlike-art/dreamlike-diffusion-1.0" for stylized art
    # Alternative: "nitrosocke/Arcane-Diffusion" for 3D animated style
    # Alternative: "stablediffusionapi/anything-v5" for anime/cartoon style
    # NOT recommended: "dreamlike-art/dreamlike-photoreal-2.0" (conflicts with cartoon goal)

    # Training prompts - Enhanced for better character definition
    INSTANCE_PROMPT = "aldar_kose, a clever trickster character wearing traditional Central Asian clothing with colorful chapan robe and doppa hat, mischievous expression, 3d cartoon style, vibrant colors, animated movie quality"
    CLASS_PROMPT = "a person wearing traditional clothing, 3d cartoon style, vibrant colors, animated movie quality"

    # Training settings - OPTIMIZED FOR STYLE TRANSFER
    RESOLUTION = 512
    TRAIN_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4
    LEARNING_RATE = 1e-4  # Increased for faster style learning
    LR_SCHEDULER = "cosine"  # Better convergence than constant
    LR_WARMUP_STEPS = 100  # More warmup for stability
    MAX_TRAIN_STEPS = 2000  # More steps for better style capture

    # LoRA configuration - OPTIMIZED FOR CHARACTER FIDELITY
    LORA_RANK = 64  # Higher rank captures more details (try 32/64/128)
    LORA_ALPHA = 64  # Match rank for balanced training
    LORA_DROPOUT = 0.0  # No dropout for character consistency
    LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"]  # More modules for better coverage

    # Memory optimization
    GRADIENT_CHECKPOINTING = True
    MIXED_PRECISION = "no"
    USE_8BIT_ADAM = False

    # Advanced options for better style transfer
    TRAIN_TEXT_ENCODER = False  # Set to True for better prompt understanding (uses more memory)
    TEXT_ENCODER_LR = 5e-6  # Much lower LR for text encoder if training it

    # Validation
    VALIDATION_PROMPT = "aldar_kose character in a bustling Central Asian marketplace, wearing his traditional chapan robe and doppa hat, mischievous smile, interacting with merchants, 3d cartoon style, vibrant colors, dynamic pose, animated movie quality"
    VALIDATION_STEPS = 200
    NUM_VALIDATION_IMAGES = 4

    # Prior preservation
    WITH_PRIOR_PRESERVATION = True
    PRIOR_LOSS_WEIGHT = 1.0
    NUM_CLASS_IMAGES = 50

    SEED = 42

config = LoRAConfig()

# Create directories
os.makedirs(config.INSTANCE_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


print(f"   Model: {config.MODEL_NAME}")
print(f"   Training steps: {config.MAX_TRAIN_STEPS}")
print(f"   Resolution: {config.RESOLUTION}x{config.RESOLUTION}\n")

# %%
class ImageUploader:
    """Handle image uploads and preprocessing"""

    def __init__(self, target_dir=None):
        self.target_dir = target_dir or config.INSTANCE_DIR
        os.makedirs(self.target_dir, exist_ok=True)
        self.images = []

    def upload_from_colab(self):
        """Load images from directory - hard-coded for Colab"""

        print("\n📤 LOADING ALDAR KoSE IMAGES")

        # Auto-create target directory if it doesn't exist
        os.makedirs(self.target_dir, exist_ok=True)

        # Load all images from directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []

        for ext in image_extensions:
            image_files.extend(Path(self.target_dir).glob(ext))

        # If no images found, provide helpful instructions
        if len(image_files) == 0:
            print(f"\n⚠️  No images found in '{self.target_dir}'")
            print("\n💡 To add images, you have 3 options:")
            print("\n   Option 1: Upload via Colab files panel")
            print(f"   - Click the folder icon on the left sidebar")
            print(f"   - Navigate to '{self.target_dir}' folder (or create it)")
            print(f"   - Drag and drop your images there")
            print("\n   Option 2: Mount Google Drive")
            print("   - Run: from google.colab import drive")
            print("   - Run: drive.mount('/content/drive')")
            print("   - Place images in: /content/drive/MyDrive/datasetAldar")
            print("   - Update INSTANCE_DIR in config to point there")
            print("\n   Option 3: Download from URL")
            print("   - Use !wget or !curl to download images")
            print(f"   - Example: !wget -P {self.target_dir} <your-image-url>")
            print("\n   Supported formats: .jpg, .jpeg, .png")
            return self.images

        # Process each image found
        print(f"   Found {len(image_files)} images in '{self.target_dir}'")
        print()

        for filepath in sorted(image_files):
            try:
                img = Image.open(filepath).convert('RGB')
                self.images.append({
                    'filename': filepath.name,
                    'filepath': str(filepath),
                    'size': img.size
                })
                print(f"   ✅ {filepath.name} ({img.size[0]}x{img.size[1]})")
            except Exception as e:
                print(f"   ❌ {filepath.name}: {str(e)}")

        print(f"\n✅ Successfully loaded {len(self.images)} images")

        if len(self.images) < 5:
            print(f"\n⚠️  WARNING: Only {len(self.images)} images found.")
            print("   For best results, use 15-30 high-quality images of your character.")

        return self.images

    def preprocess_images(self, target_size=512):
        """Enhanced preprocessing for cartoon-style training"""

        print(f"\n🔄 Preprocessing to {target_size}x{target_size}...")

        for img_data in self.images:
            try:
                img = Image.open(img_data['filepath']).convert('RGB')

                # Smart resize maintaining aspect ratio
                aspect = img.width / img.height
                if aspect > 1:
                    new_size = (int(target_size * aspect), target_size)
                else:
                    new_size = (target_size, int(target_size / aspect))

                img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Center crop
                left = (img.width - target_size) // 2
                top = (img.height - target_size) // 2
                img = img.crop((left, top, left + target_size, top + target_size))

                # Slight sharpening
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.2)

                img.save(img_data['filepath'], quality=95)
                img_data['size'] = img.size

            except Exception as e:
                print(f"   ❌ {img_data['filename']}: {e}")

        print("✅ Preprocessing complete")

    def create_metadata(self):
        """Create metadata.jsonl for training"""

        metadata_path = os.path.join(self.target_dir, "metadata.jsonl")

        with open(metadata_path, 'w') as f:
            for img_data in self.images:
                entry = {
                    "file_name": img_data['filename'],
                    "text": config.INSTANCE_PROMPT
                }
                f.write(json.dumps(entry) + "\n")

        print(f"✅ Created metadata: {metadata_path}")
        return metadata_path

    def display_dataset(self):
        """Show uploaded images"""
        import matplotlib.pyplot as plt

        n = len(self.images)
        if n == 0:
            print("❌ No images!")
            return

        cols = min(5, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

        if rows == 1 and cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

        for i, img_data in enumerate(self.images):
            img = Image.open(img_data['filepath'])
            axes[i].imshow(img)
            axes[i].set_title(img_data['filename'], fontsize=8)
            axes[i].axis('off')

        for i in range(n, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


# %%
def train_lora():
    """Train LoRA adapter using diffusers"""

    print("\n" + "="*70)
    print("🎓 TRAINING LORA ADAPTER")
    print("="*70 + "\n")

    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from torch.utils.data import Dataset, DataLoader
    from accelerate import Accelerator
    from peft import LoraConfig as PeftLoraConfig, get_peft_model
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    from torchvision import transforms

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=config.MIXED_PRECISION
    )

    print(" Loading base model...")

    # Load components with explicit dtype handling
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float32,  # Changed from float16 for stability
        safety_checker=None
    )

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.MODEL_NAME,
        subfolder="scheduler"
    )

    # Freeze everything except UNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Add LoRA to UNet
    print("🔧 Adding LoRA layers...")

    lora_config = PeftLoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        init_lora_weights="gaussian",
        target_modules=config.LORA_TARGET_MODULES,  # Use config for flexibility
        lora_dropout=config.LORA_DROPOUT,
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Enable gradient checkpointing
    if config.GRADIENT_CHECKPOINTING:
        unet.enable_gradient_checkpointing()

    # Dataset with augmentation
    class LoRADataset(Dataset):
        def __init__(self, image_dir, tokenizer, size=512):
            self.image_paths = list(Path(image_dir).glob('*.[jp][pn]g'))
            self.tokenizer = tokenizer
            self.size = size

            # Enhanced augmentation for better style generalization
            self.transforms = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  # Better quality
                transforms.CenterCrop(size),
                # More aggressive augmentation for robustness
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                # Random rotation for pose variety (small angles only)
                transforms.RandomRotation(degrees=5, interpolation=transforms.InterpolationMode.BICUBIC),
                # Random perspective for depth variety
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert('RGB')
            pixel_values = self.transforms(img)

            text_inputs = self.tokenizer(
                config.INSTANCE_PROMPT,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )

            return {
                "pixel_values": pixel_values,
                "input_ids": text_inputs.input_ids[0]
            }

    dataset = LoRADataset(config.INSTANCE_DIR, tokenizer, config.RESOLUTION)
    dataloader = DataLoader(
        dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    print(f"📊 Dataset: {len(dataset)} images")

    # Optimizer with better settings for style transfer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )

    # Learning rate scheduler for better convergence
    from torch.optim.lr_scheduler import CosineAnnealingLR
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.MAX_TRAIN_STEPS,
        eta_min=config.LEARNING_RATE * 0.1  # Decay to 10% of initial LR
    )

    # Prepare for training
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, dataloader, lr_scheduler)

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # Training loop
    print("\n🚀 Starting training...\n")

    global_step = 0
    progress_bar = tqdm(range(config.MAX_TRAIN_STEPS), desc="Training")

    unet.train()

    for epoch in range(1000):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Convert to latents - FIXED: No dtype conversion
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample timestep
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device
                ).long()

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Calculate loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Check for NaN loss and skip if found
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️  WARNING: NaN/Inf loss detected at step {global_step}! Skipping batch...")
                    optimizer.zero_grad()
                    continue

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()  # Step the scheduler
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                progress_bar.set_postfix({"loss": loss.detach().item(), "step": global_step})

                # Print milestone updates
                if global_step % 100 == 0:
                    print(f"\n📊 Step {global_step}/{config.MAX_TRAIN_STEPS} - Loss: {loss.detach().item():.4f}")

                if global_step >= config.MAX_TRAIN_STEPS:
                    print(f"\n✅ Training complete! Reached {global_step} steps.")
                    break

        if global_step >= config.MAX_TRAIN_STEPS:
            break


    print("\n Saving LoRA weights...")

    unwrapped_unet = accelerator.unwrap_model(unet)

    # Save in diffusers format
    unwrapped_unet.save_pretrained(
        config.OUTPUT_DIR,
        safe_serialization=True
    )

    # Also save as pytorch_lora_weights.bin for compatibility
    lora_state_dict = {}
    for name, param in unwrapped_unet.named_parameters():
        if 'lora' in name:
            lora_state_dict[name] = param.cpu()

    torch.save(
        lora_state_dict,
        os.path.join(config.OUTPUT_DIR, "pytorch_lora_weights.bin")
    )

    print(f"✅ LoRA saved to: {config.OUTPUT_DIR}")
    print(f"   Files: {os.listdir(config.OUTPUT_DIR)}")

    # Cleanup
    del unet, vae, text_encoder, pipeline
    torch.cuda.empty_cache()
    gc.collect()

    return config.OUTPUT_DIR

# %%
# ============================================================================
# INFERENCE CLASSES
# ============================================================================

class AldarKoseGenerator:
    """Generate images of Aldar Kose character using trained LoRA"""

    def __init__(self, lora_path=None, base_model=None):
        """Initialize the generator"""
        self.lora_path = lora_path or config.OUTPUT_DIR
        self.base_model = base_model or config.MODEL_NAME
        self.pipe = None

    def load_model(self):
        """Load the model with LoRA weights"""

        print("\n" + "="*70)
        print("🔧 LOADING ALDAR KOSE MODEL")
        print("="*70)

        print(f"\n📦 Loading base model: {self.base_model}")

        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

        # Load base pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            safety_checker=None
        )

        # Use DPMSolver for faster generation
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe = self.pipe.to("cuda")

        # Load LoRA weights
        print(f"🎨 Loading LoRA weights from: {self.lora_path}")

        try:
            self.pipe.unet.load_attn_procs(self.lora_path)
            print("   ✅ LoRA loaded successfully (load_attn_procs)")
        except Exception as e:
            print(f"   ⚠️  load_attn_procs failed: {e}")
            print("   📝 Trying direct weight loading...")

            lora_weights_path = os.path.join(self.lora_path, "pytorch_lora_weights.bin")
            if os.path.exists(lora_weights_path):
                lora_state_dict = torch.load(lora_weights_path, map_location="cuda")
                self.pipe.unet.load_state_dict(lora_state_dict, strict=False)
                print("   ✅ LoRA loaded successfully (direct load)")
            else:
                raise FileNotFoundError(f"LoRA weights not found in {self.lora_path}")

        # Memory optimizations
        print("\n⚡ Enabling optimizations...")
        self.pipe.enable_attention_slicing()

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("   ✅ xFormers enabled")
        except:
            print("   ⚠️  xFormers not available")

        print("\n✅ Model ready for generation!")
        print("="*70 + "\n")

    def generate(
        self,
        prompt,
        negative_prompt=None,
        num_steps=50,
        guidance_scale=7.5,
        seed=None,
        width=512,
        height=512,
        save_path=None
    ):
        """Generate a single image"""

        if self.pipe is None:
            self.load_model()

        # Enhanced prompt with style keywords
        style_prefix = (
            "mdjrny-v4 style, 3d rendered character, vibrant colors, colorful, "
            "pixar style, dreamworks animation quality, stylized character design, "
            "cinematic lighting, high detail, "
        )

        # Add character trigger if not present
        if "aldar_kose" not in prompt.lower():
            prompt = f"aldar_kose character, {prompt}"

        full_prompt = style_prefix + prompt

        # Default negative prompt - OPTIMIZED FOR CHARACTER CONSISTENCY
        if negative_prompt is None:
            negative_prompt = (
                # Character consistency
                "multiple people, different face, inconsistent style, different character, "
                "multiple characters, crowd, duplicated features, face change, different person, "
                # Style consistency
                "realistic, photorealistic, photograph, photo, dull colors, desaturated, monochrome, "
                "2d, flat, painting, sketch, drawing, anime, manga, "
                # Quality control
                "low quality, blurry, distorted face, deformed, ugly, bad anatomy, mutated, "
                "bad hands, extra limbs, extra fingers, missing limbs, worst quality, low res, "
                "jpeg artifacts, compression, grainy, noise, "
                # Unwanted elements
                "watermark, text, signature, logo, dark, gloomy, horror"
            )

        # Set random seed
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        print(f"\n🎨 Generating image...")
        print(f"   Prompt: {prompt[:80]}...")
        print(f"   Steps: {num_steps} | Guidance: {guidance_scale} | Seed: {seed}")

        # Generate
        image = self.pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width
        ).images[0]

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            image.save(save_path, quality=95)
            print(f"   ✅ Saved to: {save_path}")

        return image

    def generate_storyboard(
        self,
        script,
        num_shots=6,
        openai_api_key=None,
        output_dir="storyboard_output"
    ):
        """Generate a storyboard from a script"""

        if self.pipe is None:
            self.load_model()

        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        print("\n" + "="*70)
        print("🎬 GENERATING STORYBOARD")
        print("="*70)

        # Get shots from script
        if openai_api_key:
            shots = self._generate_shots_with_openai(script, num_shots, openai_api_key)
        else:
            shots = self._generate_shots_simple(script, num_shots)

        print(f"\n📝 Generated {len(shots)} shots\n")

        # Generate images for each shot
        frames = []

        for i, shot in enumerate(shots):
            shot_num = i + 1
            scene = shot.get('scene_description', shot.get('description', f'Scene {shot_num}'))
            shot_type = shot.get('shot_type', 'medium shot')

            print(f"\n[{shot_num}/{len(shots)}] {shot_type}")
            print(f"   {scene[:80]}...")

            # Build prompt
            prompt = f"{shot_type}, {scene}, dramatic composition, depth of field"

            # Generate image
            filename = f"frame_{shot_num:02d}.jpg"
            filepath = os.path.join(frames_dir, filename)

            image = self.generate(
                prompt=prompt,
                seed=config.SEED + i*10,
                save_path=filepath
            )

            frames.append({
                'shot_number': shot_num,
                'shot_type': shot_type,
                'scene': scene,
                'filename': filename,
                'filepath': filepath
            })

            torch.cuda.empty_cache()

        # Save metadata
        metadata = {
            'script': script,
            'num_shots': len(shots),
            'shots': shots,
            'generated_at': datetime.now().isoformat()
        }

        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "="*70)
        print(f"✅ COMPLETE: {len(frames)} frames saved to {output_dir}/")
        print("="*70 + "\n")

        return frames

    def _generate_shots_with_openai(self, script, num_shots, api_key):
        """Generate shots using OpenAI - Compatible with old API (0.28.x)"""

        print(f"🤖 Using OpenAI to generate {num_shots} shots...")

        try:
            import openai

            # Set API key for old API
            openai.api_key = api_key

            # Use old-style API (openai < 1.0)
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a storyboard artist for a 3D animated film. "
                            f"Break the story into {num_shots} cinematic shots. "
                            f"Focus on: character expressions, dynamic poses, clear staging. "
                            f"Return ONLY valid JSON array with: shot_number, shot_type "
                            f"(e.g. 'close-up on face', 'medium shot', 'wide establishing shot'), "
                            f"scene_description (describe scene, character action, expression, environment)."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Create {num_shots} shots for this story:\n\n{script}\n\n"
                            f"Character: Aldar Kose (clever trickster in traditional Central Asian clothing)"
                        )
                    }
                ],
                temperature=0.7
            )

            content = response['choices'][0]['message']['content']

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            shots = json.loads(content)

            if isinstance(shots, dict):
                shots = shots.get('shots', [])

            return shots

        except Exception as e:
            print(f"   ⚠️  OpenAI failed: {e}")
            print("   Falling back to simple mode...")
            return self._generate_shots_simple(script, num_shots)

    def _generate_shots_simple(self, script, num_shots):
        """Generate shots without OpenAI"""

        print(f"📋 Creating {num_shots} shots (simple mode)...")

        sentences = [s.strip() for s in script.replace('\n', ' ').split('.') if s.strip()]

        shots = []
        shot_types = [
            'wide establishing shot',
            'medium shot of character',
            'close-up on face',
            'over the shoulder shot',
            'medium shot',
            'wide shot'
        ]

        for i in range(num_shots):
            shot = {
                'shot_number': i + 1,
                'shot_type': shot_types[i % len(shot_types)],
                'scene_description': sentences[i % len(sentences)]
            }
            shots.append(shot)

        return shots


# %%
# ============================================================================
# MAIN EXECUTION - TRAINING & INFERENCE
# ============================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("🎓 ALDAR KOSE LORA - TRAINING & INFERENCE PIPELINE")
    print("="*70 + "\n")

    # ========== STEP 1: TRAINING ==========
    print("📸 Step 1: Loading training images...")
    uploader = ImageUploader()
    uploader.upload_from_colab()
    uploader.preprocess_images(config.RESOLUTION)
    uploader.create_metadata()
    uploader.display_dataset()

    print("\n🚀 Step 2: Starting LoRA training...")
    lora_path = train_lora()

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📦 Trained LoRA saved to: {lora_path}\n")

    # ========== STEP 2: INFERENCE ==========
    print("\n" + "="*70)
    print("🎨 STARTING INFERENCE EXAMPLES")
    print("="*70 + "\n")

    # Get OpenAI API key from environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

    if OPENAI_API_KEY:
        print(f"✅ OpenAI API key found in environment")
    else:
        print(f"⚠️  No OpenAI API key found - using simple mode")
        print(f"   To use OpenAI: export OPENAI_API_KEY='your-key'")

    # Initialize generator
    generator = AldarKoseGenerator(lora_path=lora_path)

    # Example 1: Single image
    print("\n📷 Example 1: Generating single image...")
    generator.generate(
        prompt="aldar_kose in marketplace, wearing traditional chapan robe",
        seed=42,
        save_path="generated_images/aldar_marketplace.jpg"
    )

    # Example 2: Storyboard
    print("\n🎬 Example 2: Generating storyboard...")

    SCRIPT = """
    Aldar Kose approached a checkpoint with a sack and told the lazy guard it was full of goat hair. Curious, the guard peeked in and found only rocks—while he was distracted, Aldar slipped past with real goods hidden in his coat.
    """

    frames = generator.generate_storyboard(
        script=SCRIPT,
        num_shots=6,
        openai_api_key=OPENAI_API_KEY
    )

    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  - Trained LoRA: {lora_path}/")
    print(f"  - Single images: generated_images/")
    print(f"  - Storyboard: storyboard_output/frames/")
    print("\n" + "="*70 + "\n")

