import torch
from einops import rearrange, einsum
import einx
# example 1
D = torch.randn(64, 100, 128)
A = torch.randn(128, 128)
Y1 = D @ A.T  # Matrix multiplication

Y2 = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

Y3 = einsum(D, A, "... d_in, d_out d_in -> ... d_out")

assert torch.allclose(Y1, Y2)
assert torch.allclose(Y1, Y3)
print("All assertions passed!")

# example 2

images = torch.randn(64, 128, 128, 3)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
print(dim_by)  # torch.Size([10])
# Reshape and multiply
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value
print(dimmed_images.shape)  # torch.Size([64, 10, 128, 128, 3])
# in one go
dimmed_images = einsum(
    images, dim_by, 
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
print(dimmed_images.shape)  # torch.Size([64, 10, 128, 128, 3])

# example 3
channels_last = torch.randn(64, 32, 32, 3)
B = torch.randn(32*32, 32*32)
print(channels_last.shape)  # torch.Size([64, 32, 32, 3])
print(B.shape)  # torch.Size([1024, 1024])

channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)

channels_first_flat = channels_last_flat.transpose(1, 2)  # (64, 3, 1024)

channels_first_flat_transformed = channels_first_flat @ B.T  # (64, 3, 1024)

channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)  # (64, 1024, 3)

channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)

height = width = 32

channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)

channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)

channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height,
    width=width
)

height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)