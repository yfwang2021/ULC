# Implementation of "Unbiased Delayed Feedback Label Correction for Conversion Rate Prediction" (KDD 2023)

This is the code to reproduce the results on the public Criteo dataset. We also provide partial output logs for easy verification of reproducibility.

To reproduce the results, first download the [Criteo dataset](https://drive.google.com/file/d/1x4KktfZtls9QjNdFYKCjTpfjM4tG2PcK/view?usp=sharing) and place data.txt in the data directory. Then, run the code according to run.sh. 

This code draws on the code of https://github.com/ThyrixYang/es_dfm, including the implementation of DFM (Chapelle 2014) and FSIW (Yasui et al. 2020). Thanks for their code.

A preprint version of our paper is available at http://arxiv.org/abs/2307.12756. If you have any questions, feel free to submit an issue or contact me (yf-wang21@mails.tsinghua.edu.cn).
