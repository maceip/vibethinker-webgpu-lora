# Training & LoRA

**Canonical production pipeline: [emberglass-tune](https://github.com/maceip/emberglass-tune) (MLX/CUDA).**

This repo (**emberglass**) is inference-first, but now also ships an **experimental
in-browser LoRA trainer** that runs a full backward pass + AdamW directly on the
WebGPU runtime, keeping the int4 base frozen and training only the LoRA A/B matrices.

```bash
cd ~/emberglass-tune        # production tuning (MLX or CUDA)
uv sync
uv run emberglass-tune --help
```

Quick answers:

| Question | Answer |
|---|---|
| How were custom weights made? | LoRA on `WeiboAI/VibeThinker-3B` via **emberglass-tune** (Anthropic traces → SFT) |
| Can I train in the browser? | **Yes (experimental)** — `QwenLoraTrainer` (`src/qwgpu/trainer.js`) + `TrainingController` (`src/services/training_controller.js`); production tuning still uses emberglass-tune |
| How do I run weights here? | Load `adapter_model.safetensors` + base into WebGPU (`src/lora_gpu.js`) |

### In-browser trainer (experimental)

- **Kernels:** `src/qwgpu/backward_kernels.js` — frozen-int4 dX, LoRA dA/dB/dX, RMSNorm/SwiGLU/RoPE backward, recompute flash-attention backward, streamed int8 LM-head + cross-entropy, AdamW, grad-norm.
- **Trainer:** gradient-checkpointed f32 forward + reverse per-layer backward sweep, grad accumulation, AdamW with warmup/cosine + global-norm clip. Updates `mod.A/mod.B` in place and calls `rt.invalidateLora()` so hot-swap inference stays valid.
- **Data/objective:** completion-only loss masking with shifted CE labels (`TrainingController.prepareExample`).
- **Export:** `src/lora_export.js` reads back A/B, un-transposes to PEFT layout, emits `.safetensors` + `adapter_config.json`.
- **Validation:** `npm run test:backward-cpu` (finite-difference gradient algebra) and `npm run test:backward` (kernels on a real WebGPU device).

See also [`REPO_ARCHITECTURE.md`](REPO_ARCHITECTURE.md).
