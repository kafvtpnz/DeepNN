# -*- coding: utf-8 -*-

import torch

def checkpoint_from_distributed(state_dict):
    """
    Функция проверяет была ли сеть обучена с использованием  DistributedDataParallel (используется на кластерах GPU)
    Такая сеть не может быть запущена на CPU
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret

def unwrap_distributed(state_dict):
    """
    Функция преобразует сеть, обученную с использованием  DistributedDataParallel (используется на кластерах GPU)
    к виду, используемому для работы на CPU
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.1.', '')
        new_key = new_key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
if device.type == 'cpu':
    ssdetector = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', pretrained=False)
    checkpoint = torch.hub.load_state_dict_from_url('https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/19.09.0/files/nvidia_ssdpyt_fp16_190826.pt', map_location="cpu")
    checkpoint = checkpoint['model']
    if checkpoint_from_distributed(checkpoint):
        checkpoint = unwrap_distributed(checkpoint)
    ssdetector.load_state_dict(checkpoint)
elif 'cuda':
    ssdetector = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', pretrained=True)

ssdetector.eval()

if(ssdetector):
    print('')
    print('OK!')

