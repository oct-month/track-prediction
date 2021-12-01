from mxnet import gpu, cpu
from mxnet.util import get_gpu_count

gpu_counts = get_gpu_count()
device = gpu() if gpu_counts > 0 else cpu()
devices = [gpu(i) for i in range(gpu_counts)] if gpu_counts > 0 else [cpu()]
