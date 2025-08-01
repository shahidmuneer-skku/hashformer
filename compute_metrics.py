import torch
import time
from ptflops import get_model_complexity_info
from src.models.video_cav_mae import VideoCAVMAEFT
import torch.nn as nn
# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VideoCAVMAEFT().to(device)
model.eval()

class FlopsWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
    def forward(self, batch):
        audio, video = batch
        return self.base(audio=audio, video=video)

# 2) Instantiate & move to device
wrapped = FlopsWrapper(model).to(device)
wrapped.eval()



# 2. Total parameters (in millions)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# 3. GFLOPs via ptflops
#    You must supply the correct input shapes for your audio and video streams.
#    Example shapes (batch=1): audio=(1, audio_channels, audio_length),
#                              video=(1, 3, num_frames, height, width)
audio_shape = (1024, 128)          # ← adjust to your a_input shape
video_shape = (3, 16, 224, 224)      # ← adjust to your v_input shape
# B, 1024, 128
def model_fn(inputs):
    a, v = inputs
    return model(a, v)
def make_inputs(res):
    # res is (audio_shape, video_shape), but you can ignore it
    return (
        torch.zeros((1, *audio_shape), device=device),
        torch.zeros((1, *video_shape), device=device),
    )

macs, params = get_model_complexity_info(
    wrapped,
    input_res=(audio_shape, video_shape),
    input_constructor=make_inputs,
    as_strings=True,
    print_per_layer_stat=False,
)

print(f"Computational complexity: {macs} FLOPs")
# (ptflops will also print a params count, but you’ll trust your own sum above)

# 4. Inference time measurement
#    Create dummy inputs matching your real ones:
a_dummy = torch.randn((1, *audio_shape), device=device)
v_dummy = torch.randn((1, *video_shape), device=device)

# Warm‐up (GPU auto‐tuning)
for _ in range(10):
    _ = model(a_dummy, v_dummy)

# Time over several runs
reps = 100
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
starter.record()
for _ in range(reps):
    _ = model(a_dummy, v_dummy)
ender.record()
torch.cuda.synchronize()
avg_time_ms = starter.elapsed_time(ender) / reps
print(f"Average inference time: {avg_time_ms:.2f} ms per sample")