# MiniMax-H3 on PPU-ZW810E (FlagOS baseline)

This page records a single-card BF16 bring-up of the H3 integration on a
T-Head PPU-ZW810E (96 GiB) with FlagGems `thead` and the vendor `flash-attn`
build. It is the plain operator path: FlashAttention for dense attention,
FlagGems RMSNorm, and the host Triton RoPE. No Sage, no fusion opt-in, and no
multi-card parallelism were enabled.

The H3 launcher, registry keys and config schema come from the H3 integration
PR (#8). The examples added here are meant to land with or after that PR; they
do not run against a tree that lacks `examples/minimax_h3.py`.

## Validated environment

| Component | Version |
|---|---|
| Device | PPU-ZW810E, 96 GiB, single card |
| PyTorch | 2.13.0 |
| FlagGems | 5.5.0.dev0+ga4ca511d137d.ppu285b (`vendor_name=thead`) |
| FlagTree | 0.7.0+ppu3.6 |
| flash-attn | 2.7.4.post1 (vendor build, exposes `flash_attn_func`) |
| transformers / tokenizers | 5.14.1 / 0.22.2 |
| diffusers / huggingface_hub | 0.43.0 / 1.33.0 |
| LightX2V host | c58c16d607c49bf6021fecc6238ed177752d7e87 |
| lightx2v-plugin-FL | 2cf49116b12383122db0d1247b484d24ba4409c3 (PR #8 head) |
| FlagOmni | 431a7e6fcb9251fd7b44cdcb665839f4771b425f (PR #1 head) |

`diffusers` 0.43 needs `huggingface_hub>=1.32`; transformers 5.x requires
`huggingface_hub<2.0`, so 1.33.x is the compatible range. `flag_gems`,
`flagtree`, PyTorch and `flash-attn` must stay on the vendor distributions.

## Configs

- `examples/minimax_h3_ppu_small.json` — 64x64, 22 frames, 1 step. Functional
  smoke test only; output is not a quality reference.
- `examples/minimax_h3_ppu.json` — 544x960, 124 frames, 29 steps. Target
  configuration used for the semantic check.

Both select `attn_type=flagos_flash_attn`, `rms_type=flagos_rms_norm` and keep
`rope_type=minimax_h3_triton_rope`. The text encoder runs on the real prompt
(`text_encoder_load_on_init=true`); no fixed-conditioning fixture is required.

`text_encoder_offload_granularity="block"` is important on a 96 GiB card: the
FL2VA transformer is ~66 GB and the video VAE ~11 GB, so moving the whole
Qwen3-VL encoder (≈52 GB) to the device at once does not fit. Block granularity
keeps two encoder blocks resident (1.82 GiB) and keeps the text-encoding peak at
3.27 GiB. The default `model` granularity OOMs on this device.

## Run

```bash
export PLATFORM=flagos
export LIGHTX2V_FL_GLOBAL_GEMS=0
export LIGHTX2V_FL_DISABLE_FLAGCX=1
export CUDA_VISIBLE_DEVICES=0

# Functional smoke test (~27 s cold; ~5 s warm)
python examples/minimax_h3.py \
  --model-path /models/MiniMax-H3 --config examples/minimax_h3_ppu_small.json \
  --prompt 'A small boat moves slowly across a calm lake at sunrise.' \
  --seed 42 --output /tmp/h3-small.mp4

# Target configuration (~6 min on one card)
python examples/minimax_h3.py \
  --model-path /models/MiniMax-H3 --config examples/minimax_h3_ppu.json \
  --prompt 'A small boat moves slowly across a calm lake at sunrise.' \
  --seed 42 --output /tmp/h3.mp4
```

The weights are the Diffusers-layout release (`transformer/`, `text_encoder/`,
`vae/`, `audio_vae/`, `tokenizer/`, `processor/`). Only the root layout is
needed for `model_variant=fl2av`; the `FL2VA/`, `Ref2VA/` and `transformer_ref/`
trees are not used by this path.

The upstream FastAPI service uses the same runner and environment:

```bash
python -m lightx2v.server --model_cls minimax_h3 --model-variant fl2av \
  --model_path /models/MiniMax-H3 --config_json examples/minimax_h3_ppu.json \
  --host 0.0.0.0 --port 18300

curl -s -X POST http://127.0.0.1:18300/v1/tasks/video/ \
  -H 'Content-Type: application/json' \
  -d '{"task":"t2av","prompt":"...","seed":42,"num_frames":124,
       "size":[544,960],"save_result_path":"/tmp/h3-api.mp4"}'
curl -s http://127.0.0.1:18300/v1/tasks/<task_id>/status
```

## Measured results

Smoke test (64x64 / 22 frames / 1 step):

| Metric | Value |
|---|---|
| model load | 41.2 s |
| text encoder | 2.5 s |
| DiT step 1 (includes first-call compile) | 16.1 s |
| video + audio VAE decode | 4.1 s + 3.9 s |
| pipeline | 26.7 s cold, 4.96 s warm |

Target configuration (544x960 / 124 frames / 29 steps):

| Metric | Value |
|---|---|
| packed sequence | text 12 + audio 414 + video 18,870 = **19,296** tokens |
| steady-state DiT | **11.23 s/step** (first step 17.96 s) |
| video + audio VAE decode | 13.9 s + 0.1 s |
| pipeline | **352 s** |
| resident device memory (denoise) | **75.8 GiB** |
| output | h264 960x544 24 fps + AAC 32 kHz stereo, 5.175 s |

Operator hit counts on the smoke test, wrapping `flash_attn.flash_attn_func`
and `flag_gems.rms_norm`:

```
OPERATOR_CALLS {'flash': 52, 'rms': 210}
```

52 is consistent with 50 DiT layers plus 2 refiner layers; the RMS count covers
per-layer norms and qk-norms. The target run produced a stable, semantically
correct sunrise-lake boat clip; audio was non-silent (RMS 0.24).

## Limitations

- One card, BF16 baseline only. Ulysses sequence parallelism
  (`parallel.seq_p_size`) and the FlagOmni SageAttention path were not
  exercised on this device.
- Only 544x960 / 124 frames / 29 steps was checked for semantics. Higher
  resolution and 2K presets still need their own memory and quality runs.
- The service path was validated end-to-end with the 1-step config; the
  29-step config uses the same runner but should be re-run before production.
- No plugin source changes were required. The adaptation consists of the
  example configs, the memory-granularity choice above, and the Python
  dependency set documented here.
