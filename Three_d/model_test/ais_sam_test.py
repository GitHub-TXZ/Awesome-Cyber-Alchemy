from Three_d.ais_sam.ais_sam import AIS_SAM
import torch
import os
import torch
from Three_d.ais_sam.auto_prompter_3D import SPG

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    spg = SPG().to('cuda')
    ais_sam = AIS_SAM(r=4, adapter_dim=64, spg=spg, d_size=20).to('cuda')
    for name, param in ais_sam.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Parameters: {param.numel() / 1e6}M")
    total_params = sum(p.numel() for p in ais_sam.parameters())
    print(f"Total parameters: {total_params / 1e6}M")
    trainable_params = sum(p.numel() for p in ais_sam.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6}M")
    #
    # for name, param in ais_sam.named_parameters():
    #     print(f"Layer: {name}, Parameters: {param.numel()/1e6}M")
    input = torch.randn(1, 2, 20, 256, 256).to('cuda')
    output = ais_sam(input)
    total_allocated_memory_mib = torch.cuda.memory_allocated() / (1024 ** 2)
    max_memory_allocated_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Total Allocated GPU Memory: {total_allocated_memory_mib:.2f} MiB")
    print(f"Max GPU Memory Usage: {max_memory_allocated_mib:.2f} MiB")
    print(output.shape)




