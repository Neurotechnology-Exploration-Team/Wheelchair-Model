import torch
print(torch.version.cuda)
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should print 'GeForce RTX 3070' or similar
print(torch.version.cuda)