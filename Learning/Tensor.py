import torch
import numpy as np

# Initializing Tensor directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Initializing Tensor from a NumPy array
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)

# Initializing Tensor from another tensor
# x_ones = torch.ones_like(x_data) # retains the properties of x_data
# print(f"Ones Tensor: \n {x_ones} \n")

# x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")
# The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
# Output: 
    # Ones Tensor: 
    #  tensor([[1, 1],
    #         [1, 1]]) 

    # Random Tensor:
    #  tensor([[0.9776, 0.5287],
    #         [0.3318, 0.5432]])

# With random og constant values
# shape = (2, 3,)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")
# Output:
    # Random Tensor: 
    #  tensor([[0.3365, 0.9461, 0.5236],
    #         [0.2885, 0.7171, 0.6463]])

    # Ones Tensor:
    #  tensor([[1., 1., 1.],
    #         [1., 1., 1.]])

    # Zeros Tensor:
    #  tensor([[0., 0., 0.],
    #         [0., 0., 0.]])

# Tensor attributes
# Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor = torch.rand(3, 4)
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")
# Output:
    # Shape of tensor: torch.Size([3, 4])
    # Datatype of tensor: torch.float32
    # Device tensor is stored on: cpu

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on: {tensor.device}")
else:
    print("Torch is not compiled with CUDA enabled")