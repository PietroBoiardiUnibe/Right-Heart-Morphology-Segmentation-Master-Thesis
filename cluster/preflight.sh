# Sourced at the start of every GPU job, after the environment is active.
# Imports exactly what TotalSegmentator needs at run time (the chain that failed on UBELIX:
# nnunetv2 -> dynamic_network_architectures -> timm -> torchvision) and checks the GPU, so a
# broken environment stops the task in seconds instead of failing every case one by one.
python - <<'EOF'
import torch, torchvision, timm, nnunetv2
from dynamic_network_architectures.architectures.primus import Primus
assert torch.cuda.is_available(), "no GPU visible to torch"
print("preflight ok: torch", torch.__version__, "| torchvision", torchvision.__version__,
      "|", torch.cuda.get_device_name(0), flush=True)
EOF
