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
