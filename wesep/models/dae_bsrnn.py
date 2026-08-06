import numpy as np
import torch
import torch.nn as nn

from wesep.modules.common.speaker import SpeakerFuseLayer

from wesep.models.bsrnn import BSNet
from wesep.modules.common.text import get_asr_encoder

class TextAwareFuseSeparation(nn.Module):

    def __init__(
        self,
        nband=7,
        num_repeat=6,
        feature_dim=128,
        cue_emb_dim=256,
        cue_fuse={},
    ):
        """

        :param nband : len(self.band_width)
        """
        super(TextAwareFuseSeparation, self).__init__()

        assert cue_fuse != {}

        self.nband = nband
        self.feature_dim = feature_dim

        self.multi_fuse = cue_fuse.get('multi_fuse', True)
        self.fuse_type = cue_fuse.get('fuse_type', 'FiLM')

        self.separation = nn.ModuleList([])
        self.fuse_layers = nn.ModuleList([])

        for _ in range(num_repeat):
            self.separation.append(BSNet(nband * feature_dim, nband))

        if not self.multi_fuse:
            num_repeat = 1

        for _ in range(num_repeat):
            self.fuse_layers.append(SpeakerFuseLayer(
                embed_dim=cue_emb_dim,
                feat_dim=feature_dim,
                fuse_type=self.fuse_type,
            ))

    def forward_fuse(self, fuse_layer: nn.Module, x: torch.Tensor, transposed_cue_embedding: torch.Tensor):
        '''
        x: [B, nband, feature_dim, T]
        transposed_cue_embedding: [B, nband, T, feature_dim] (transposed in main)
        out: [B, nband, feature_dim, T]

        id(layer) checked
        '''

        x = x.transpose(2, 3)                # B, nband, T, feature_dim
        x = fuse_layer(x, transposed_cue_embedding)    # B, nband, T, feature_dim
        x = x.transpose(2, 3).contiguous()   # B, nband, feature_dim, T

        return x

    def forward(self, x: torch.Tensor, cue_embedding: torch.Tensor, nch: torch.Tensor = torch.tensor(1)):
        """
        cue_embedding is the return value of the upstream embedding fusion.

        FiLM requires feature_dim appearing at the last dim
        FiLM operates cue_embedding.squeeze(-1) before forward

        x: [B, nband, feature_dim, T]
        cue_embedding: [B, nband, feature_dim, T]
        out: [B, nband, feature_dim, T]
        """
        batch_size = x.shape[0]

        assert len(cue_embedding.shape) == 4
        assert cue_embedding.shape[-1] == 1 or x.shape[-1] == cue_embedding.shape[-1]

        transposed_cue_embedding = cue_embedding.transpose(2, 3)   # B, nband, T, feature_dim (for fuse)

        if self.multi_fuse:
            for fuse_layer, sep_func in zip(self.fuse_layers, self.separation):
                x = self.forward_fuse(fuse_layer, x, transposed_cue_embedding)
                x = x.view(batch_size * nch, self.nband * self.feature_dim, -1)
                x = sep_func(x)
                x = x.view(batch_size * nch, self.nband, self.feature_dim, -1)   
        else:
            x = self.forward_fuse(self.fuse_layers[0], x, transposed_cue_embedding)
            x = x.view(batch_size * nch, self.nband * self.feature_dim, -1)
            for sep_func in self.separation:
                x = sep_func(x)
            x = x.view(batch_size * nch, self.nband, self.feature_dim, -1)

        return x

class DAEBSRNN(nn.Module):
    # self, sr=16000, win=512, stride=128, feature_dim=128, num_repeat=6,
    # use_bidirectional=True
    def __init__(
        self,
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        cue_emb_dim=128,
        num_repeat=6,
        cue_fuse={},
        asr_encoder=None,
        **kwargs,
    ):
        super(DAEBSRNN, self).__init__()

        self.sr = sr
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.cue_emb_dim = cue_emb_dim
        self.eps = torch.finfo(torch.float32).eps

        # ASR (TC-ASR) encoder — the only cue source
        if asr_encoder is None:
            raise ValueError("asr_encoder is required for DAEBSRNN")
        self.asr_encoder_name = asr_encoder['model_name']
        self.asr_encoder = get_asr_encoder(asr_encoder)

        # 0-1k (100 hop), 1k-4k (250 hop),
        # 4k-8k (500 hop), 8k-16k (1k hop),
        # 16k-20k (2k hop), 20k-inf

        # 0-8k (1k hop), 8k-16k (2k hop), 16k
        bandwidth_100 = int(np.floor(100 / (sr / 2.0) * self.enc_dim))
        bandwidth_200 = int(np.floor(200 / (sr / 2.0) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.0) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.0) * self.enc_dim))

        # add up to 8k
        self.band_width = [bandwidth_100] * 15
        self.band_width += [bandwidth_200] * 10
        self.band_width += [bandwidth_500] * 5
        self.band_width += [bandwidth_2k] * 1

        self.band_width.append(self.enc_dim - int(np.sum(self.band_width)))
        self.nband = len(self.band_width)

        self.nch = 1

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i] * 2, self.eps),
                    nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1),
                ))

        self.separator = TextAwareFuseSeparation(
            nband=self.nband,
            num_repeat=num_repeat,
            feature_dim=feature_dim,
            cue_emb_dim=cue_emb_dim,
            cue_fuse=cue_fuse,
        )

        # self.proj =  nn.Linear(hidden_size*2, input_size)

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim,
                                 torch.finfo(torch.float32).eps),
                    nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.band_width[i] * 4, 1),
                ))

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward_stft(self, wav_input: torch.Tensor):
        spec = torch.stft(
            wav_input,
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(wav_input.device).type(
                wav_input.type()),
            return_complex=True,
        )
        return spec

    def forward_band_split(self, spec: torch.Tensor):
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(spec_RI[:, :, band_idx:band_idx +
                                        self.band_width[i]].contiguous())
            subband_mix_spec.append(spec[:, band_idx:band_idx +
                                         self.band_width[i]])  # B*nch, BW, T
            band_idx += self.band_width[i]

        return subband_spec, subband_mix_spec

    def forward_subband_feature(self, subband_spec):
        batch_size = subband_spec[0].shape[0]

        subband_feature = []
        for i, bn_func in enumerate(self.BN):
            subband_feature.append(
                bn_func(subband_spec[i].view(batch_size * self.nch,
                                             self.band_width[i] * 2, -1)))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        return subband_feature

    def forward_mask(self, sep_output: torch.Tensor, subband_mix_spec: list):
        batch_size = sep_output.shape[0]

        sep_subband_spec = []
        for i, mask_func in enumerate(self.mask):
            this_output = mask_func(sep_output[:, i]).view(
                batch_size * self.nch, 2, 2, self.band_width[i], -1)
            this_mask = this_output[:, 0] * torch.sigmoid(
                this_output[:, 1])  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:, 0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:, 1]  # B*nch, K, BW, T
            est_spec_real = (subband_mix_spec[i].real * this_mask_real -
                             subband_mix_spec[i].imag * this_mask_imag
                             )  # B*nch, BW, T
            est_spec_imag = (subband_mix_spec[i].real * this_mask_imag +
                             subband_mix_spec[i].imag * this_mask_real
                             )  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real,
                                                  est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T

        return est_spec

    def forward_istft(self, wav_input, est_spec: torch.Tensor):
        batch_size, nsample = wav_input.shape

        output = torch.istft(
            est_spec.view(batch_size * self.nch, self.enc_dim, -1),
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(wav_input.device).type(
                wav_input.type()),
            length=nsample,
        )

        output = output.view(batch_size, self.nch, -1)
        s = torch.squeeze(output, dim=1)

        return s

    def forward_asr_encoder(self, mix_wav, additional: dict = {}):
        if self.asr_encoder is None:
            return None
        data_type = additional.pop('data_type')
        if self.asr_encoder_name == 'TCASREncoder':
            assert data_type == 'TC-ASR'
            text_input = [additional['data'][_] for _ in ['kw_label', 'kw_len']]
            if 'mix_enroll' in additional['data'].keys(): 
                # for test, ablation
                speaker_emb, speaker_pred = self.asr_encoder(additional['data']['mix_enroll'], text_input)
            else:
                speaker_emb, speaker_pred = self.asr_encoder(mix_wav, text_input)

            return speaker_emb.unsqueeze(1).unsqueeze(3), speaker_pred
        else:
            raise NotImplementedError(f"Unsupported ASR encoder: {self.asr_encoder_name}")

    def forward(self, wav_input, spk_emb_input, additional_input = {}):
        '''
            wav_input shape: (B, T), wav
            spk_emb_input shape: (B, T'), wav
        '''
        del spk_emb_input   # saving cuda memory

        # frequency-domain separation
        spec = self.forward_stft(wav_input)

        # concat real and imag, split to subbands
        subband_spec, subband_mix_spec = self.forward_band_split(spec)

        # normalization and bottleneck
        subband_feature = self.forward_subband_feature(subband_spec)

        # construct cue embedding from ASR encoder (TC-ASR)
        asr_encoded_mix, asr_speaker_pred = self.forward_asr_encoder(wav_input, additional_input)
        _, nband, _, T = subband_feature.shape
        embedding = asr_encoded_mix.repeat(1, nband, 1, T)

        # separator should not accept additional because cue should be fixed as embedding previously
        sep_output = self.separator(subband_feature, embedding, torch.tensor(self.nch))

        # forward mask, and generate output waveform with istft
        est_spec = self.forward_mask(sep_output, subband_mix_spec)
        s = self.forward_istft(wav_input, est_spec)

        return s, asr_speaker_pred
