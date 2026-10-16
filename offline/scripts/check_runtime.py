"""Validate the licensed CUDA runtime without commanding any hardware."""
import json
from pathlib import Path
import numpy as np
import torch
import MinkowskiEngine as ME
from pointnet2 import pointnet2_utils
from graspnetAPI import GraspGroup
from gsnet import check_license

root = Path(__file__).resolve().parents[1]
assert check_license('license') is True, 'Official license check failed'
assert torch.cuda.is_available(), 'CUDA unavailable'
# Exercise real CUDA kernels, rather than merely importing extension modules.
xyz = torch.rand(1, 64, 3, device='cuda')
indices = pointnet2_utils.furthest_point_sample(xyz, 8)
assert indices.shape == (1, 8)
coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]], dtype=torch.int32)
x = ME.SparseTensor(features=torch.ones(3, 2, device='cuda'), coordinates=coords, device='cuda')
conv = ME.MinkowskiConvolution(2, 3, kernel_size=3, dimension=3).cuda()
y = conv(x)
assert y.F.shape == (3, 3) and torch.isfinite(y.F).all()
a = np.zeros((2, 17), dtype=np.float64)
a[:, 0] = [0.9, 0.8]
a[:, 1:4] = [0.05, 0.03, 0.02]
a[:, 4:13] = np.eye(3).reshape(9)
a[:, 16] = -1
assert len(GraspGroup(a).nms()) == 1
torch.cuda.synchronize()
report = {'license_passed': True, 'torch': torch.__version__, 'cuda': torch.version.cuda,
          'gpu': torch.cuda.get_device_name(0), 'minkowski_engine': ME.__version__,
          'pointnet2_cuda_test': True, 'sparse_convolution_cuda_test': True,
          'grasp_nms_test': True, 'model_inference_ran': False}
print(json.dumps(report, indent=2))
(root / 'outputs/deployment_20261011/runtime_check.json').write_text(json.dumps(report, indent=2))
