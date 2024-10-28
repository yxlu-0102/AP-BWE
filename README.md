# Towards High-Quality and Efficient Speech Bandwidth Extension with Parallel Amplitude and Phase Prediction
### Ye-Xin Lu, Yang Ai, Hui-Peng Du, Zhen-Hua Ling

**Abstract:** 
Speech bandwidth extension (BWE) refers to widening the frequency bandwidth range of speech signals, enhancing the speech quality towards brighter and fuller.
This paper proposes a generative adversarial network (GAN) based BWE model with parallel prediction of Amplitude and Phase spectra, named AP-BWE, which achieves both high-quality and efficient wideband speech waveform generation.
The proposed AP-BWE generator is entirely based on convolutional neural networks (CNNs).
It features a dual-stream architecture with mutual interaction, where the amplitude stream and the phase stream communicate with each other and respectively extend the high-frequency components from the input narrowband amplitude and phase spectra.
To improve the naturalness of the extended speech signals, we employ a multi-period discriminator at the waveform level and design a pair of multi-resolution amplitude and phase discriminators at the spectral level, respectively.
Experimental results demonstrate that our proposed AP-BWE achieves state-of-the-art performance in terms of speech quality for BWE tasks targeting sampling rates of both 16 kHz and 48 kHz. 
In terms of generation efficiency, due to the all-convolutional architecture and all-frame-level operations, the proposed AP-BWE can generate 48 kHz waveform samples 292.3 times faster than real-time on a single RTX 4090 GPU and 18.1 times faster than real-time on a single CPU.
Notably, to our knowledge, AP-BWE is the first to achieve the direct extension of the high-frequency phase spectrum, which is beneficial for improving the effectiveness of existing BWE methods.

**We will provide our implementation as open source in this repository after paper acceptance.**
Audio samples can be found [here](http://yxlu-0102.github.io/AP-BWE).

## Pre-requisites
1. Python >= 3.6.
2. Clone this repository.
3. Install python requirements. Please refer `requirements.txt`.
4. Download and extract the [VCTK-0.92 dataset](https://datashare.ed.ac.uk/handle/10283/3443).

## Training
```
cd train
CUDA_VISIBLE_DEVICES=0 python train_16k.py --config [config file path]
CUDA_VISIBLE_DEVICES=0 python train_48k.py --config [config file path]
```
Checkpoints and copies of the configuration file are saved in the `cp_model` directory by default.<br>
You can change the path by using the `--checkpoint_path` option.
Here is an example:
```
CUDA_VISIBLE_DEVICES=0 python train_16k.py --config ../configs/config_2kto16k.json --checkpoint_path ../checkpoints/AP-BWE_2kto16k
```

## Inference
```
cd inference
python inference_16k.py --checkpoint_file [generator checkpoint file path]
```
You can also use the pretrained checkpoint files we provide in the `checkpoints` directory.
<br>
Generated wav files are saved in `generated_files` by default.
You can change the path by adding `--output_dir` option.<br>
Here is an example:
```
python inference_16k.py --checkpoint_file ../checkpoints/2kto16k/g_2kto16k --output_dir ../generated_files/2kto16k
```

## Model Structure
![model](Figures/model.png)

## Comparison with other speech BWE methods
![comparison](Figures/table.png)

## Acknowledgements
We referred to [HiFi-GAN](https://github.com/jik876/hifi-gan) and [NSPP](https://github.com/YangAi520/NSPP) to implement this.

## Citation
```
@article{lu2024towards,
  title={Towards high-quality and efficient speech bandwidth extension with parallel amplitude and phase prediction},
  author={Lu, Ye-Xin and Ai, Yang and Du, Hui-Peng and Ling, Zhen-Hua},
  journal={arXiv preprint arXiv:2401.06387},
  year={2024}
}

@inproceedings{lu2024multi,
  title={Multi-Stage Speech Bandwidth Extension with Flexible Sampling Rate Control},
  author={Lu, Ye-Xin and Ai, Yang and Sheng, Zheng-Yan and Ling, Zhen-Hua},
  booktitle={Proc. Interspeech},
  pages={2270--2274},
  year={2024}
}
```
