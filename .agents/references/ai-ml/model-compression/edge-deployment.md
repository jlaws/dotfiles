# Edge Deployment Reference

## CoreML Conversion

```python
import coremltools as ct
import torch

model = MyModel()
model.eval()
traced = torch.jit.trace(model, torch.randn(1, 3, 224, 224))

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 224, 224), scale=1/255.0, bias=[0, 0, 0])],
    outputs=[ct.TensorType(name="logits")],
    compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + Neural Engine
    minimum_deployment_target=ct.target.iOS16,
)
mlmodel.save("model.mlpackage")
```

### CoreML Tips
- `compute_units=ct.ComputeUnit.ALL` to leverage Neural Engine
- Float16 is default on Neural Engine and sufficient for most tasks
- Use `ct.ImageType` for image inputs to avoid manual preprocessing on device
- Test on actual hardware; simulator performance differs significantly

## TensorRT Optimization

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
config.set_flag(trt.BuilderFlag.FP16)

engine_bytes = builder.build_serialized_network(network, config)
with open("model.engine", "wb") as f:
    f.write(engine_bytes)
```

### Dynamic Shapes

```python
profile = builder.create_optimization_profile()
profile.set_shape("image",
    min=(1, 3, 224, 224),
    opt=(8, 3, 224, 224),
    max=(32, 3, 224, 224),
)
config.add_optimization_profile(profile)
```

### torch-tensorrt (Simpler Path)

```python
import torch_tensorrt

model = MyModel().eval().cuda()
inputs = [torch_tensorrt.Input(
    min_shape=[1, 3, 224, 224],
    opt_shape=[8, 3, 224, 224],
    max_shape=[32, 3, 224, 224],
    dtype=torch.float16,
)]

trt_model = torch_tensorrt.compile(
    model, inputs=inputs, enabled_precisions={torch.float16},
)
output = trt_model(input_tensor.half().cuda())
```

## ONNX Static Quantization with Calibration

```python
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat

class MyCalibrationReader(CalibrationDataReader):
    def __init__(self, data_loader):
        self.data_iter = iter(data_loader)

    def get_next(self):
        try:
            batch = next(self.data_iter)
            return {"image": batch.numpy()}
        except StopIteration:
            return None

quantize_static(
    "model.onnx",
    "model_int8_static.onnx",
    calibration_data_reader=MyCalibrationReader(cal_loader),
    quant_format=QuantFormat.QDQ,  # Preferred for TensorRT compatibility
)
```

## Browser ML

### ONNX Runtime Web

```javascript
import * as ort from "onnxruntime-web";

ort.env.wasm.numThreads = 4;

const session = await ort.InferenceSession.create("model.onnx", {
  executionProviders: ["webgpu", "wasm"],
});

const inputTensor = new ort.Tensor("float32", floatArray, [1, 3, 224, 224]);
const results = await session.run({ image: inputTensor });
const logits = results.logits.data; // Float32Array
```

### Transformers.js

```javascript
import { pipeline } from "@xenova/transformers";

const classifier = await pipeline("sentiment-analysis");
const result = await classifier("I love this product!");
// [{ label: "POSITIVE", score: 0.9998 }]

const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
const embedding = await embedder("Hello world", { pooling: "mean", normalize: true });
```

## Model Profiling and Benchmarking

```python
import time
import torch
import numpy as np

def benchmark_pytorch(model, input_shape, device="cuda", n_warmup=10, n_runs=100):
    model.eval().to(device)
    x = torch.randn(*input_shape, device=device)

    for _ in range(n_warmup):
        with torch.no_grad():
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times = np.array(times) * 1000
    print(f"Latency: {np.mean(times):.2f} +/- {np.std(times):.2f} ms")
    print(f"P50: {np.percentile(times, 50):.2f} ms, P99: {np.percentile(times, 99):.2f} ms")
    print(f"Throughput: {1000 / np.mean(times):.1f} inferences/sec")
    return times

def benchmark_onnx(model_path, input_dict, n_warmup=10, n_runs=100):
    import onnxruntime as ort
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    for _ in range(n_warmup):
        session.run(None, input_dict)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, input_dict)
        times.append(time.perf_counter() - start)

    times = np.array(times) * 1000
    print(f"ONNX Latency: {np.mean(times):.2f} +/- {np.std(times):.2f} ms")
    return times
```
