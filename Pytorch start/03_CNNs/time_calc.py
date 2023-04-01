from torch import device as troch_device
from torch.cuda import get_device_name
from timeit import default_timer as timer

def print_train_time(
    start:float,
    end:float,
    device:troch_device = None
):
    total_time = end - start
    if device == "cpu":
        print(f"Model took {total_time}s training on CPU")
    else:
        print(f"model Took {total_time}s training on {get_device_name()}")
        
    return total_time