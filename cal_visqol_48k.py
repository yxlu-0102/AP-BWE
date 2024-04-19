import os
import numpy as np
import argparse
import librosa
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2
from rich.progress import track
from joblib import Parallel, delayed

config = visqol_config_pb2.VisqolConfig()

def cal_vq(reference, degraded, mode='audio'):
    if mode == "audio":
        config.audio.sample_rate = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        config.audio.sample_rate = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")

    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)

    api = visqol_lib_py.VisqolApi()

    api.Create(config)

    similarity_result = api.Measure(reference, degraded)

    return similarity_result.moslqo


def main(h):
    # with open(h.test_file, 'r', encoding='utf-8') as fi:
    #     wav_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]
    wav_indexes = os.listdir(h.ref_wav_dir)

    metrics = {'vq':[]}

    for wav_index in track(wav_indexes):

        ref_wav, ref_sr = librosa.load(os.path.join(h.ref_wav_dir, wav_index), sr=float(h.sampling_rate), dtype=np.float64)
        syn_wav, syn_sr = librosa.load(os.path.join(h.syn_wav_dir, wav_index), sr=float(h.sampling_rate), dtype=np.float64)

        if float(h.sampling_rate) != 48000:
            ref_wav = librosa.resample(ref_wav, orig_sr=float(h.sampling_rate), target_sr=48000)
            syn_wav = librosa.resample(syn_wav, orig_sr=float(h.sampling_rate), target_sr=48000)
        
        length = min(len(ref_wav), len(syn_wav))
        ref_wav = ref_wav[: length]
        syn_wav = syn_wav[: length]
        try:
            vq_score = cal_vq(ref_wav, syn_wav)
            metrics['vq'].append(vq_score)
        except:
            vq_score = 0
    
    vq_mean = np.mean(metrics['vq'])

    print('VISQOL: {:.3f}'.format(vq_mean))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_rate', required=True)
    parser.add_argument('--ref_wav_dir', required=True)
    parser.add_argument('--syn_wav_dir', required=True)

    h = parser.parse_args()

    main(h)