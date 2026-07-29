# Model Compression & Edge Deployment

## Technique Selection

### Compression by Constraint

| Primary Constraint | Technique | Typical Compression | Accuracy Drop | Effort |
|-------------------|-----------|-------------------|---------------|--------|
| Memory (model size) | Quantization (PTQ INT8) | 2-4x | 0.1-1% | Low |
| Memory (extreme) | Quantization (INT4/GPTQ) | 4-8x | 1-5% | Medium |
| Latency (structured) | Structured pruning | 1.5-3x speedup | 1-5% | Medium |
| Latency + memory | Distillation | 2-10x smaller | 1-10% | High |
| Latency (HW-specific) | QAT + target runtime | 2-4x speedup | 0.5-2% | High |
| All constraints (extreme) | Pruning + distillation + quantization | 10-50x | 3-15% | Very High |

### Pruning Approach

| Scenario | Type | Granularity | Speedup Without Sparse HW |
|----------|------|-------------|--------------------------|
| General size reduction | Unstructured | Weight-level | None (need sparse kernels) |
| Actual inference speedup | Structured | Channel/head/layer | Yes |
| Transformer attention heads | Structured | Head-level | Yes |
| Conv-heavy vision models | Structured | Filter-level | Yes |
| NLP with hardware support | Semi-structured (2:4) | Block pattern | Yes (Ampere+ GPUs) |

### Quantization Tradeoffs

| Method | Accuracy Drop | Size Reduction | Speed Gain | Effort |
|--------|--------------|----------------|------------|--------|
| FP16 | < 0.1% | 2x | 1.5-2x (GPU) | Trivial |
| Dynamic INT8 | 0.5-1% | 2-4x | 1.5-3x (CPU) | Low |
| Static INT8 (PTQ) | 1-2% | 3-4x | 2-4x | Medium |
| QAT INT8 | < 0.5% | 3-4x | 2-4x | High |
| INT4 (GPTQ/AWQ) | 1-3% | 4-8x | 2-4x | Medium |

## Export Format Selection

| Format | Target | Strengths |
|--------|--------|-----------|
| **ONNX** | Cross-platform, server, edge | Universal interchange, wide runtime support |
| **CoreML** | iOS, macOS, Apple Silicon | Neural Engine acceleration, on-device privacy |
| **TensorRT** | NVIDIA GPUs | Fastest GPU inference, kernel fusion |
| **TFLite** | Android, microcontrollers | Small runtime, NNAPI/GPU delegate |
| **ONNX Runtime Web** | Browser (WASM/WebGPU) | Client-side inference, no server |
| **ExecuTorch** | iOS, Android | PyTorch-native mobile, replaces TorchScript |

**Decision rule**: ONNX for cross-platform. CoreML for Apple. TensorRT for max NVIDIA throughput. TFLite for Android/MCUs. ExecuTorch to stay in PyTorch ecosystem for mobile.

## Structured Pruning

```python
import torch.nn as nn
import torch.nn.utils.prune as prune

def prune_conv_channels(model, amount=0.3):
    """Prune conv2d filters by L1 norm (structured)."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
            prune.remove(module, "weight")
    return model
```

See `ai-ml/model-compression/pruning-and-distillation.md` for transformer head pruning and knowledge distillation.

## Quantization

### Post-Training Quantization (PTQ)

```python
from torch.ao.quantization import get_default_qconfig, prepare, convert

model.eval()
model.qconfig = get_default_qconfig("x86")  # or "qnnpack" for ARM
prepared = prepare(model)

with torch.no_grad():
    for batch in calibration_loader:
        prepared(batch)

quantized_model = convert(prepared)
```

### ONNX Runtime PTQ

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic("model.onnx", "model_int8.onnx", weight_type=QuantType.QInt8)
```

See `ai-ml/model-compression/edge-deployment.md` for static ONNX quantization with calibration reader.

### Quantization-Aware Training (QAT)

```python
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

model.train()
model.qconfig = get_default_qat_qconfig("x86")
prepared = prepare_qat(model)

# Fine-tune with fake quantization nodes
for epoch in range(3):
    for batch, targets in train_loader:
        output = prepared(batch)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

prepared.eval()
quantized = convert(prepared)
```

## ONNX Export

```python
import torch

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)

# Validate
import onnx
onnx.checker.check_model(onnx.load("model.onnx"))
```

## Model Size Analysis

```python
def model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total = (param_size + buffer_size) / (1024 ** 2)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Size: {total:.1f} MB")
    return total
```

## Gotchas

### Compression
- **Pruning recovery**: Fine-tune for 20-30% of original training budget. Pruning >50% without iterative prune+retrain cycles causes permanent accuracy loss.
- **Distillation capacity gap**: If student is too small relative to teacher, distillation underperforms training from scratch. Use progressive distillation (12->6->3) or intermediate layer matching.
- **Calibration data**: PTQ quality depends on representative calibration data. 100 samples is usually sufficient, but must match inference distribution.
- **Structured vs unstructured**: Unstructured pruning shows great sparsity numbers but zero speedup on standard hardware. Only structured pruning gives wall-clock speedup.
- **Layer sensitivity**: First/last layers in vision models, embedding layers in transformers -- more sensitive to compression. Profile per-layer sensitivity before uniform compression.

### Quantization
- **Hardware-specific**: INT8 on ARM (XNNPACK) differs from x86 (FBGEMM). Symmetric vs asymmetric, per-tensor vs per-channel -- hardware-dependent. Always profile on target device.
- **Sensitive layers**: Attention and first/last conv layers quantize poorly. Use mixed-precision: keep sensitive layers FP16, quantize rest to INT8.

### Export & Deployment
- **ONNX dynamic axes**: Always specify `dynamic_axes` for batch dimension, otherwise model has fixed batch size.
- **ONNX opset**: Use 17+ for modern ops. Lower opsets lack newer attention patterns and grouped convolutions.
- **CoreML vs simulator**: Neural Engine not available in simulator. Always test on real hardware.
- **TensorRT portability**: Engines are GPU-specific (A100 engine won't run on T4). Ship ONNX + build script, not the engine file.
- **Mobile memory**: iOS hard-kills apps exceeding ~1.5 GB RAM. Profile peak memory during inference, not just model size.
- **Browser models**: Compress aggressively (INT8 + gzip). Consider chunked progressive loading. Cache with IndexedDB.

## References

- `ai-ml/model-compression/pruning-and-distillation.md` -- Transformer head pruning, knowledge distillation training loop
- `ai-ml/model-compression/edge-deployment.md` -- CoreML, TensorRT, browser ML, benchmarking, ONNX static quantization
