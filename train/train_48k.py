import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append("..")
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from datasets.dataset import Dataset, amp_pha_stft, amp_pha_istft, get_dataset_filelist
from models.model import APNet_BWE_Model,  MultiPeriodDiscriminator, MultiResolutionAmplitudeDiscriminator, MultiResolutionPhaseDiscriminator, \
     feature_loss, generator_loss, discriminator_loss, phase_losses, cal_snr, cal_lsd
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = APNet_BWE_Model(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrad = MultiResolutionAmplitudeDiscriminator().to(device)
    mrpd = MultiResolutionPhaseDiscriminator().to(device)

    if rank == 0:
        print(generator)
        num_params = 0
        for p in generator.parameters():
            num_params += p.numel()
        print(num_params)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(a.checkpoint_path, 'logs'), exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        mrad.load_state_dict(state_dict_do['mrad'])
        mrpd.load_state_dict(state_dict_do['mrpd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        mrad = DistributedDataParallel(mrad, device_ids=[rank]).to(device)
        mrpd = DistributedDataParallel(mrpd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(mrad.parameters(), mrpd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_indexes, validation_indexes = get_dataset_filelist(a)

    trainset = Dataset(training_indexes, a.input_training_wavs_dir, h.segment_size, h.hr_sampling_rate, h.lr_sampling_rate,
                       split=True, n_cache_reuse=0, shuffle=False if h.num_gpus > 1 else True, device=device)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    if rank == 0:
        validset = Dataset(validation_indexes, a.input_validation_wavs_dir, h.segment_size, h.hr_sampling_rate, h.lr_sampling_rate,
                           split=False, shuffle=False, n_cache_reuse=0, device=device)
        
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    mrad.train()
    mrpd.train()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):

            if rank == 0:
                start_b = time.time()
            audio_wb, audio_nb = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
            audio_wb = torch.autograd.Variable(audio_wb.to(device, non_blocking=True))
            audio_nb = torch.autograd.Variable(audio_nb.to(device, non_blocking=True))
            
            mag_wb, pha_wb, com_wb = amp_pha_stft(audio_wb, h.n_fft, h.hop_size, h.win_size)
            mag_nb, pha_nb, com_nb = amp_pha_stft(audio_nb, h.n_fft, h.hop_size, h.win_size)
            
            mag_wb_g, pha_wb_g, com_wb_g = generator(mag_nb, pha_nb)

            audio_wb_g = amp_pha_istft(mag_wb_g, pha_wb_g, h.n_fft, h.hop_size, h.win_size)
            mag_wb_g_hat, pha_wb_g_hat, com_wb_g_hat = amp_pha_stft(audio_wb_g, h.n_fft, h.hop_size, h.win_size)
            audio_wb, audio_wb_g = audio_wb.unsqueeze(1), audio_wb_g.unsqueeze(1)

            optim_d.zero_grad()

            # MPD
            audio_df_r, audio_df_g, _, _ = mpd(audio_wb, audio_wb_g.detach())
            loss_disc_f, losses_disc_p_r, losses_disc_p_g = discriminator_loss(audio_df_r, audio_df_g)

            # MRAD
            spec_da_r, spec_da_g, _, _ = mrad(audio_wb, audio_wb_g.detach())
            loss_disc_a, losses_disc_a_r, losses_disc_a_g = discriminator_loss(spec_da_r, spec_da_g)

            # MRPD
            spec_dp_r, spec_dp_g, _, _ = mrpd(audio_wb, audio_wb_g.detach())
            loss_disc_p, losses_disc_p_r, losses_disc_p_g = discriminator_loss(spec_dp_r, spec_dp_g)

            loss_disc_all = (loss_disc_a + loss_disc_p) * 0.1 + loss_disc_f

            loss_disc_all.backward()
            torch.nn.utils.clip_grad_norm_(parameters=mpd.parameters(), max_norm=10, norm_type=2)
            torch.nn.utils.clip_grad_norm_(parameters=mrad.parameters(), max_norm=10, norm_type=2)
            torch.nn.utils.clip_grad_norm_(parameters=mrpd.parameters(), max_norm=10, norm_type=2)
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L2 Magnitude Loss
            loss_mag = F.mse_loss(mag_wb, mag_wb_g) * 45
            # Anti-wrapping Phase Loss
            loss_ip, loss_gd, loss_iaf = phase_losses(pha_wb, pha_wb_g)
            loss_pha = (loss_ip + loss_gd + loss_iaf) * 100
            # L2 Complex Loss
            loss_com = F.mse_loss(com_wb, com_wb_g) * 90
            # L2 Consistency Loss
            loss_stft = F.mse_loss(com_wb_g, com_wb_g_hat) * 90

            audio_df_r, audio_df_g, fmap_f_r, fmap_f_g = mpd(audio_wb, audio_wb_g)
            spec_da_r, spec_da_g, fmap_a_r, fmap_a_g = mrad(audio_wb, audio_wb_g)
            spec_dp_r, spec_dp_g, fmap_p_r, fmap_p_g = mrpd(audio_wb, audio_wb_g)

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_a = feature_loss(fmap_a_r, fmap_a_g)
            loss_fm_p = feature_loss(fmap_p_r, fmap_p_g)

            loss_gen_f, losses_gen_f = generator_loss(audio_df_g)
            loss_gen_a, losses_gen_a = generator_loss(spec_da_g)
            loss_gen_p, losses_gen_p = generator_loss(spec_dp_g)

            loss_gen = (loss_gen_a + loss_gen_p) * 0.1 + loss_gen_f
            loss_fm = (loss_fm_a + loss_fm_p) * 0.1 + loss_fm_f

            loss_gen_all = loss_mag  + loss_pha + loss_com + loss_stft + loss_gen + loss_fm

            loss_gen_all.backward()
            torch.nn.utils.clip_grad_norm_(parameters=generator.parameters(), max_norm=10, norm_type=2)
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mag_error = F.mse_loss(mag_wb, mag_wb_g).item()
                        ip_error, gd_error, iaf_error = phase_losses(pha_wb, pha_wb_g)
                        pha_error = (ip_error + gd_error + iaf_error).item()
                        com_error = F.mse_loss(com_wb, com_wb_g).item()
                        stft_error = F.mse_loss(com_wb_g, com_wb_g_hat).item()
                    print('Steps : {:d}, Gen Loss: {:4.3f}, Magnitude Loss : {:4.3f}, Phase Loss : {:4.3f}, Complex Loss : {:4.3f}, STFT Loss : {:4.3f}, s/b : {:4.3f}'.
                           format(steps, loss_gen_all, mag_error, pha_error, com_error, stft_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'mrad': (mrad.module if h.num_gpus > 1
                                                         else mrad).state_dict(),
                                     'mrpd': (mrpd.module if h.num_gpus > 1
                                                         else mrpd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Phase Loss", pha_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Consistency Loss", stft_error, steps)

                # Validation
                if steps % a.validation_interval == 0:
                    start_v = time.time()
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    val_stft_err_tot = 0
                    val_snr_score_tot = 0
                    val_lsd_score_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            audio_wb, audio_nb = batch
                            audio_wb = torch.autograd.Variable(audio_wb.to(device, non_blocking=True))
                            audio_nb = torch.autograd.Variable(audio_nb.to(device, non_blocking=True))
                            
                            mag_wb, pha_wb, com_wb = amp_pha_stft(audio_wb, h.n_fft, h.hop_size, h.win_size)
                            mag_nb, pha_nb, com_nb = amp_pha_stft(audio_nb, h.n_fft, h.hop_size, h.win_size)

                            mag_wb_g, pha_wb_g, com_wb_g = generator(mag_nb.to(device), pha_nb.to(device))

                            audio_wb = amp_pha_istft(mag_wb, pha_wb, h.n_fft, h.hop_size, h.win_size)
                            audio_wb_g = amp_pha_istft(mag_wb_g, pha_wb_g, h.n_fft, h.hop_size, h.win_size)
                            mag_wb_g_hat, pha_wb_g_hat, com_wb_g_hat = amp_pha_stft(audio_wb_g, h.n_fft, h.hop_size, h.win_size)

                            val_mag_err_tot += F.mse_loss(mag_wb, mag_wb_g_hat).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(pha_wb, pha_wb_g_hat)
                            val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_com_err_tot += F.mse_loss(com_wb, com_wb_g_hat).item()
                            val_stft_err_tot += F.mse_loss(com_wb_g, com_wb_g_hat).item()
                            val_snr_score_tot += cal_snr(audio_wb_g, audio_wb).item()
                            val_lsd_score_tot += cal_lsd(audio_wb_g, audio_wb).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/audio_nb_{}'.format(j), audio_nb[0], steps, h.hr_sampling_rate)
                                    sw.add_audio('gt/audio_wb_{}'.format(j), audio_wb[0], steps, h.hr_sampling_rate)
                                    sw.add_figure('gt/spec_nb_{}'.format(j), plot_spectrogram(mag_nb.squeeze().cpu().numpy()), steps)
                                    sw.add_figure('gt/spec_wb_{}'.format(j), plot_spectrogram(mag_wb.squeeze().cpu().numpy()), steps)

                                sw.add_audio('generated/audio_g_{}'.format(j), audio_wb_g[0], steps, h.hr_sampling_rate)
                                sw.add_figure('generated/spec_g_{}'.format(j), plot_spectrogram(mag_wb_g.squeeze().cpu().numpy()), steps)

                        val_mag_err = val_mag_err_tot / (j+1)
                        val_pha_err = val_pha_err_tot / (j+1)
                        val_com_err = val_com_err_tot / (j+1)
                        val_stft_err = val_stft_err_tot / (j+1)
                        val_snr_score = val_snr_score_tot / (j+1)
                        val_lsd_score = val_lsd_score_tot / (j+1)

                        print('Steps : {:d}, SNR Score: {:4.3f}, LSD Score: {:4.3f}, s/b : {:4.3f}'.
                                format(steps, val_snr_score, val_lsd_score, time.time() - start_v))
                        sw.add_scalar("Validation/LSD Score", val_lsd_score, steps)
                        sw.add_scalar("Validation/SNR Score", val_snr_score, steps)
                        sw.add_scalar("Validation/Magnitude Loss", val_mag_err, steps)
                        sw.add_scalar("Validation/Phase Loss", val_pha_err, steps)
                        sw.add_scalar("Validation/Complex Loss", val_com_err, steps)
                        sw.add_scalar("Validation/Consistency Loss", val_stft_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_training_wavs_dir', default='VCTK-Corpus-0.92/wav48/train')
    parser.add_argument('--input_validation_wavs_dir', default='VCTK-Corpus-0.92/wav48/test')
    parser.add_argument('--input_training_file', default='VCTK-Corpus-0.92/training.txt')
    parser.add_argument('--input_validation_file', default='VCTK-Corpus-0.92/test.txt')
    parser.add_argument('--checkpoint_path', default='cp_model')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()