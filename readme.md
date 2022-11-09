# OCAP: Personalize Edge DNN models Via On-Device Class-Aware Pruning
This repository contains the code for OCAP.

OCAP proposes a new class-aware pruning method based on the 
intermediate activation of input images to identify and remove 
the class-irrelevant channels. 

For other details, please refer to the paper.

All the model definitions are available under `./models`, 
and the auxiliary functions are available under `./utils`.

## OCAP on CIFAR
`MobileNetV2_Cifar_Pruning.py`
`ResNet_Cifar_Pruning.py`
`Vgg_Cifar_Pruning.py`


## OCAP on ImageNet
'Vgg_ImageNet_Pruning.py'