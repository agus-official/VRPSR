# VRPSR: minimal two-stage training release

This is an **additive training package** for `agus-official/VRPSR`, reviewed against
repository commit `fbc13b5a979d0486b55d016cec14d2cde20c6215` and the six author-supplied
training files. Copy its contents into your existing checkout. Existing inference
files, model implementations and `configs/sr.yaml` are not overwritten.

**Validation status:** syntax, CLI, dependency-name and source-level regression checks
passed. GPU training, checkpoint compatibility, codec binaries and DDP have NOT been
run in the preparation environment. Complete the smoke checks below before publishing
this as a tested training release. Exact metric reproduction is not claimed.

## What remains

- `src/train_simulator.py`: diffusion codec simulator, with MSE + LPIPS + GAN.
- `src/train_sr.py`: train SR through a frozen simulator, with simulated lightly
  compressed targets at `clamp(QP - 10, 0, 51)`.
- `src/vrpsr_training/data.py`: codec dataset, SR dataset and seeded validation.
- `src/vrpsr_training/codec.py`: shared real codecs, prompts, rate matching and embeddings.
- `src/vrpsr_training/options.py`: only arguments used by the final recipe.
- `src/vrpsr_training/evaluation.py`: PSNR/SSIM (Y), LPIPS, DISTS and FID.
- `configs/vrpsr_degradation.yaml`: original numerical degradation settings.
- Two launch scripts, one environment/check script and supplemental requirements.

The simulator and SR use the existing `S3DiffMetaEmbed` implementation. Historical
S3Diff-without-meta, five-dimensional embeddings, STE, alternating simulator/SR
optimization, rate-loss models, watermark fonts, prompt ablations and debug hooks are
not dependencies of this recipe. Real codecs and the repository's modified BasicSR
remain necessary.

## Environment and weights

Prefer the author's **existing working training environment**. This package does not
invent a version lock; the repository's broad dependency lower bounds are not a tested
training environment. In particular, the custom UNet and LoRA forward depend on
Diffusers and PEFT internals. Record a working environment with `pip freeze`.

For a new environment, install PyTorch/torchvision appropriate for your CUDA first,
then review and install `requirements-training.txt`. It includes the repository's
requirements plus explicit `peft` and `vision-aided-loss` dependencies. Do not blindly
upgrade an environment already used for the paper. The default precision is explicitly
`no`; if the original Accelerate configuration enabled mixed precision, match that
setting explicitly after testing it.

Build `cpp/` and its submodules as described in the repository README, and place
`codecsimulator*.so` in `src/`. All three codec classes (H.264/H.265/H.266) are required.
Do not replace the repository's modified BasicSR with an arbitrary pip version.

Supply these paths; no weights or training data are bundled:

| Path | Meaning |
| --- | --- |
| `DE_NET` | degradation estimator, normally `checkpoints/de_net.pth` |
| `SD_PATH` | local SD-Turbo snapshot; if omitted, the original automatic download is used |
| `SR_INIT` | original S3Diff initialization `s3diff.pkl`, **not** the final VRPSR SR weight |
| `SIM_CHECKPOINT` | trained simulator; either an existing compatible weight or stage-one output |
| `TRAIN_DATA` | image directory or text file, one image path per line |
| `VAL_DATA` | validation image directory or text file |

Use absolute image paths in text lists. Relative entries resolve from the repository
root, as in the original code. Directory discovery supports PNG/JPEG recursively.
Validation image basenames (and stems for simulator validation) must be unique. Inputs
are resized by short edge and center cropped to 512x512. This recipe is fixed at 512;
the original command's `resolution` flag did not control those dataset transforms.
`sf: 1` is preserved deliberately: do not change it to four as a cleanup operation.

LPIPS, IQA/FID and the DINO-based discriminator may fetch their own pretrained weights.
Prepare their caches as well when training offline. An omitted simulator
`--pretrained_path` means base SD-Turbo plus freshly initialized adaptation parameters.

## Launch

Run from the repository root after copying the package contents:

```bash
export TRAIN_DATA=/path/to/imagenet_train.txt
export VAL_DATA=/path/to/kodak
export DE_NET=/path/to/de_net.pth
export SD_PATH=/path/to/sd-turbo
export SR_INIT=/path/to/s3diff.pkl
export GPU_IDS=0,1,2,3,4,5,6,7

python scripts/check_training.py
python scripts/check_training.py --runtime

# Stage 1: default stopping follows the original 50 epochs.
bash scripts/train_simulator.sh

# Or explicitly stop at the selected simulator checkpoint's update count:
# bash scripts/train_simulator.sh --stop_steps 60001
# 60001 comes from the supplied checkpoint filename, not a claim about the best duration.

# Stage 2 automatically uses outputs/simulator/checkpoints/simulator_final.pkl.
bash scripts/train_sr.sh
```

To use your existing simulator instead of retraining it:

```bash
export SIM_CHECKPOINT=/path/to/lora_60001.pkl
bash scripts/train_sr.sh
```

Set `GPU_IDS=0` for one GPU. Changing GPU count or batch size changes the effective
batch; one-GPU smoke runs are functional checks, not paper-equivalent training.
`PORT`, `SIM_OUTPUT`, and `SR_OUTPUT` override the ports and output directories.
Additional CLI flags can be appended to either shell command. Run either Python
entry point with `--help` to see all exposed options without importing GPU libraries.

## Preserved recipe and step semantics

| Setting | Simulator | SR |
| --- | --- | --- |
| Training codecs | uniform choice of libx264/libx265/libx266 | same |
| Training QP | inclusive 30–50 | inclusive 25–48 |
| Validation | libx264 / QP 38 | same |
| Per-device batch / accumulation | 4 / 2 | 2 / 4 |
| Learning rate / scheduler | 2e-5 / constant | same |
| MSE / LPIPS / GAN multipliers | 2 / 5 / 0.5 | same, on post-simulator output |
| Codec conditioning | codec one-hot (3) + legacy rate scalar (1) | same |
| Direct SR reconstruction loss | n/a | removed because multiplier was zero |
| Simulator during SR | n/a | frozen parameters, **input gradients retained** |

The variable historically called `bpp` is **encoded packet bytes / 1000**, not standard
bits per pixel. Prompts and numerical embeddings keep this convention for checkpoint
compatibility. Codec IDs and their one-hot order remain unchanged.

**Do not equate step counts across the scripts.** Simulator `global_step` counts
synchronized optimizer updates. Its original `max_train_steps=50000` configures the
scheduler/progress display but did not terminate training; by default this release
also stops after 50 epochs. `--stop_steps N` adds an explicit update limit. The original
save/validation offset is preserved: 1, 5001, 10001, ... for interval 5000. The final
model is always exported too.

SR `max_train_steps=50000` counts microbatches per rank. It retains the original
1000-microbatch outer chunks and scheduler horizons, including the discriminator's
original multiplier. Save/validation decisions occur at chunk boundaries. With
constant LR this leaves the supplied schedule unchanged. Partial final chunks now
stop exactly at the requested limit. Intervals below 1000 are only checked at chunk
boundaries; use the defaults for the original recipe. `--eval_freq 0` disables evaluation.

Optimizer zeroing, GAN update order and Accelerate accumulation calls are retained.
The extra SR `zero_grad` has not been declared a bug or removed on speculation:
its behavior depends on the installed Accelerate wrapper. Gradient equivalence across
versions must be checked in the actual environment.

## Outputs and deliberate scope

- Simulator: `checkpoints/lora_<update>.pkl`, `checkpoints/simulator_final.pkl`.
- SR: `checkpoints/step_<microbatch>/sr_<microbatch>.pkl`, `checkpoints/sr_final.pkl`.
- `training_args.json` and `environment.txt` record new runs.
- `--wandb` enables offline rank-zero logging; default training needs no WandB login.
- Weight exports retain the existing model's `save_model` format for inference.

This minimal release exports model weights, **not exact-resume training states**.
The original simulator debug `ckpt_*` dump was removed: the supplied trainer had no
corresponding full restore path, and that dump lacked complete data/RNG recovery.
Do not present loading `--pretrained_path` as resuming an optimizer trajectory.

## GPU smoke checks before release

Use separate output directories and your original working environment:

```bash
GPU_IDS=0 SIM_OUTPUT=outputs/smoke_sim bash scripts/train_simulator.sh \
  --stop_steps 2 --train_batch_size 1 --gradient_accumulation_steps 1 \
  --dataloader_num_workers 0 --eval_freq 0

GPU_IDS=0 SR_OUTPUT=outputs/smoke_sr bash scripts/train_sr.sh \
  --max_train_steps 2 --train_batch_size 1 --gradient_accumulation_steps 1 \
  --dataloader_num_workers 0 --eval_freq 0
```

The SR smoke test needs a compatible pretrained simulator via `SIM_CHECKPOINT`;
use the original trained simulator for a meaningful functional test. Verify finite
losses and successful exports, then load those exports with the existing inference
scripts. Repeat a short run on at least two GPUs, including one validation and
checkpoint event. Validate that SR receives a finite, nonzero gradient through the
frozen simulator and that simulator parameters remain unchanged. Finally compare
fixed-seed losses/updates against the originals in the same environment.

No paper-level quality or full training result has been verified here. The public
model still contains intermediate-tensor debug collection; this package intentionally
does not modify its forward computation while preparing the training release.
