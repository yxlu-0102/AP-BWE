import os
import argparse
import torch
import torchaudio
import numpy as np
from rich.progress import track

def stft(audio, n_fft=2048, hop_length=512):
    hann_window = torch.hann_window(n_fft).to(audio.device)
    stft_spec = torch.stft(audio, n_fft, hop_length, window=hann_window, return_complex=True)
    stft_mag = torch.abs(stft_spec)
    stft_pha = torch.angle(stft_spec)

    return stft_mag, stft_pha


def cal_snr(pred, target):
    snr = (20 * torch.log10(torch.norm(target, dim=-1) / torch.norm(pred - target, dim=-1).clamp(min=1e-8))).mean()
    return snr


def cal_lsd(pred, target):
    sp = torch.log10(stft(pred)[0].square().clamp(1e-8))
    st = torch.log10(stft(target)[0].square().clamp(1e-8))
    return (sp - st).square().mean(dim=1).sqrt().mean()


def anti_wrapping_function(x):
    return x - torch.round(x / (2 * np.pi)) * 2 * np.pi


def cal_apd(pred, target):
    pha_pred = stft(pred)[1]
    pha_target = stft(target)[1]
    dim_freq = 1025
    dim_time = pha_pred.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(device)
    gd_r = torch.matmul(pha_target.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(pha_pred.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(device)
    iaf_r = torch.matmul(pha_target, iaf_matrix)
    iaf_g = torch.matmul(pha_pred, iaf_matrix)

    apd_ip = anti_wrapping_function(pha_pred - pha_target).square().mean(dim=1).sqrt().mean()
    apd_gd = anti_wrapping_function(gd_r - gd_g).square().mean(dim=1).sqrt().mean()
    apd_iaf = anti_wrapping_function(iaf_r - iaf_g).square().mean(dim=1).sqrt().mean()

    return apd_ip, apd_gd, apd_iaf


def main(h):

    wav_indexes = os.listdir(h.reference_wav_dir)
    
    metrics = {'lsd':[], 'apd_ip': [], 'apd_gd': [], 'apd_iaf': [], 'snr':[]}

    for wav_index in track(wav_indexes):

        ref_wav, ref_sr = torchaudio.load(os.path.join(h.reference_wav_dir, wav_index))
        syn_wav, syn_sr = torchaudio.load(os.path.join(h.synthesis_wav_dir, wav_index))
        
        length = min(ref_wav.size(1), syn_wav.size(1))
        ref_wav = ref_wav[:, : length].to(device)
        syn_wav = syn_wav[:, : length].to(device)
        ref_wav = ref_wav.to(device)
        syn_wav = syn_wav[:, : ref_wav.size(1)].to(device)

        lsd_score = cal_lsd(syn_wav, ref_wav)
        apd_score = cal_apd(syn_wav, ref_wav)
        snr_score = cal_snr(syn_wav, ref_wav)


        metrics['lsd'].append(lsd_score)
        metrics['apd_ip'].append(apd_score[0])
        metrics['apd_gd'].append(apd_score[1])
        metrics['apd_iaf'].append(apd_score[2])
        metrics['snr'].append(snr_score)


    lsd_mean = torch.stack(metrics['lsd'], dim=0).mean()
    apd_ip_mean = torch.stack(metrics['apd_ip'], dim=0).mean()
    apd_gd_mean = torch.stack(metrics['apd_gd'], dim=0).mean()
    apd_iaf_mean = torch.stack(metrics['apd_iaf'], dim=0).mean()
    snr_mean = torch.stack(metrics['snr'], dim=0).mean()

    print('LSD: {:.3f}'.format(lsd_mean))
    print('SNR: {:.3f}'.format(snr_mean))
    print('APD_IP: {:.3f}'.format(apd_ip_mean))
    print('APD_GD: {:.3f}'.format(apd_gd_mean))
    print('APD_IAF: {:.3f}'.format(apd_iaf_mean))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--reference_wav_dir', default='VCTK-Corpus-0.92/wav16/test')
    parser.add_argument('--synthesis_wav_dir', default='generated_files/AP-BWE')

    h = parser.parse_args()

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    main(h)