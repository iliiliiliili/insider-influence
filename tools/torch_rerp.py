import torch


def custom_repr(self):

    has_mean = self.dtype in [torch.float16, torch.float32, torch.float64]
    if len(self.shape) > 0:
        return (
            f"T{tuple(self.shape)}".replace("(", "[").replace(")", "]")
            + (f"<m={self.mean().item():.2f}>" if has_mean else "")
            + (f"<s={self.sum().item()}>" if not has_mean else "")
            + f".{self.device.type}"
            )
    
    else:
        return original_repr(self)



original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr
torch.Tensor.__str__ = original_repr