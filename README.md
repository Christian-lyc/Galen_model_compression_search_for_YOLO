# Galen: Hardware-specific Automatic Compression of Neural Networks
(This code is an adapted version for YOLOv8 model compression. It incorporates Ultralitics code and the updated version of torch pruning.)

Galen is a framework for automatic compression of neural networks by applying layer-specific pruning and quantization.
The layer-wise compression is determined by a RL algorithm which uses the sensitivity but also hardware inference latency as features.

[**Towards Hardware-Specific Automatic Compression of Neural Networks**
](https://arxiv.org/abs/2212.07818) ([AAAI-23: 2nd International Workshop on Practical
Deep Learning in the Wild](https://practical-dl.github.io/))

original repository is here: https://github.com/UniHD-CEG/galen/tree/main
