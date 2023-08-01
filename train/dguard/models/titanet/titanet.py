from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import modules, losses


class DumbConvNet(nn.Module):
    """
    Simple convolutional model used to test the learning loop
    """

    def __init__(
        self,
        n_mels,
        loss_function=None,
        hidden_size=256,
        embedding_size=192,
        kernel_size=3,
        n_layers=1,
        device="cpu",
    ):
        super(DumbConvNet, self).__init__()

        # Define architecture
        channels = [n_mels] + [hidden_size] * n_layers
        self.conv = nn.Sequential(
            *[
                modules.ConvBlock1d(in_channels, out_channels, kernel_size=kernel_size)
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ]
        )
        self.fc = nn.Linear(hidden_size, embedding_size)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Store loss function
        self.loss_function = loss_function

        # Transfer to device
        self.to(device)

    def forward(self, spectrograms, speakers=None):
        """
        Given input spectrograms of shape [B, M, T], DumbConvNet returns
        utterance-level embeddings of shape [B, E]

        B: batch size
        M: number of mel frequency bands
        T: maximum number of time steps (frames)
        E: embedding size
        """
        encodings = self.conv(spectrograms)
        embeddings = self.fc(encodings.transpose(1, 2))
        embeddings = self.pool(embeddings.transpose(1, 2)).squeeze(-1)

        # Inference mode
        if speakers is None:
            return F.normalize(embeddings, p=2, dim=1)

        # Training mode
        assert (
            self.loss_function is not None
        ), "Loss function should not be None in training mode"
        return self.loss_function(embeddings, speakers)


class DVectorBaseline(nn.Module):
    """
    Simple recurrent module that splits spectrograms in segments
    and computes embeddings as the average embedding of consecutive
    segments

    "Generalized End-to-End Loss for Speaker Verification",
    Wan et al., https://arxiv.org/abs/1710.10467
    """

    def __init__(
        self,
        n_mels,
        loss_function=None,
        n_lstm_layers=3,
        hidden_size=768,
        lstm_average=True,
        embedding_size=256,
        segment_length=160,
        device="cpu",
    ):
        super(DVectorBaseline, self).__init__()

        # Define architecture
        self.recurrent = nn.LSTM(n_mels, hidden_size, n_lstm_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, embedding_size)

        # Store attributes
        self.lstm_average = lstm_average
        self.embedding_size = embedding_size
        self.segment_length = segment_length
        self.loss_function = loss_function

        # Transfer to device
        self.to(device)

    def forward(self, spectrograms, speakers=None):
        """
        Given input spectrograms of shape [B, M, T], it returns
        utterance-level embeddings of shape [B, E]

        B: batch size
        N: number of segments for each spectrogram
        M: number of mel frequency bands
        T: maximum number of time steps (frames)
        E: embedding size
        H: hidden size
        S: segment length
        """
        # Pad spectrograms to have at least a number of frames
        # equal to the selected segment length
        right_pad = np.clip(self.segment_length - spectrograms.size(-1), 0, None)
        padded_spectrograms = F.pad(spectrograms, (0, right_pad), "constant", 0)

        # Compute segments for each spectrogram and stack them on the batch dimension
        # to obtain a tensor of size [B * N, M, S]
        segments_list = [
            s.unfold(-1, self.segment_length, self.segment_length // 2).transpose(0, 1)
            for s in padded_spectrograms
        ]
        segments = torch.cat(segments_list)

        # Forward LSTM pass
        # [B * N, M, S] -> [B * N, S, H]
        outputs, _ = self.recurrent(segments.transpose(1, 2))

        # Collapse LSTM hidden states by either averaging or taking the last
        # [B * N, S, H] -> [B * N, H]
        outputs = outputs.mean(dim=1) if self.lstm_average else outputs[:, -1, :]

        # Project hidden states to the embedding size
        # [B * N, H] -> [B * N, E]
        segments_embedding = self.projection(outputs)

        # Compute the average embedding over all segments in each spectrogram
        # [B * N, E] -> [B, E]
        batch_size, num_segments = spectrograms.size(0), segments_list[0].size(0)
        embeddings = segments_embedding.reshape(
            batch_size, num_segments, self.embedding_size
        ).mean(dim=1)

        # Inference mode
        if speakers is None:
            return F.normalize(embeddings, p=2, dim=1)

        # Training mode
        assert (
            self.loss_function is not None
        ), "Loss function should not be None in training mode"
        return self.loss_function(embeddings, speakers)


class TitaNet(nn.Module):
    """
    TitaNet is a neural network for extracting speaker representations,
    by leveraging 1D depth-wise separable convolutions with SE layers
    and a channel attention based statistic pooling layer

    "TitaNet: Neural Model for speaker representation with 1D Depth-wise
    separable convolutions and global context", Kologuri et al.,
    https://arxiv.org/abs/2110.04410
    """

    TARGET_PARAMS = {"s": 6.4, "m": 13.4, "l": 25.3}

    def __init__(
        self,
        n_mels,
        n_mega_blocks,
        n_sub_blocks,
        encoder_hidden_size,
        encoder_output_size,
        embedding_size,
        mega_block_kernel_size,
        prolog_kernel_size=3,
        epilog_kernel_size=1,
        attention_hidden_size=128,
        se_reduction=16,
        simple_pool=False,
        loss_function=None,
        dropout=0.5,
        device="cpu",
    ):
        super(TitaNet, self).__init__()

        # Define encoder and decoder
        self.encoder = Encoder(
            n_mels,
            n_mega_blocks,
            n_sub_blocks,
            encoder_hidden_size,
            encoder_output_size,
            mega_block_kernel_size,
            prolog_kernel_size=prolog_kernel_size,
            epilog_kernel_size=epilog_kernel_size,
            se_reduction=se_reduction,
            dropout=dropout,
        )
        self.decoder = Decoder(
            encoder_output_size,
            attention_hidden_size,
            embedding_size,
            simple_pool=simple_pool,
        )

        # Store loss function
        self.loss_function = loss_function

        # Transfer to device
        self.to(device)

    def get_n_params(self, div=1):
        """
        Return the number of parameters in the model and possibly
        divide it by the given number
        """
        return (
            sum([np.prod(p.size()) for p in self.parameters() if p.requires_grad]) / div
        )

    @classmethod
    def find_n_mega_blocks(
        cls,
        embedding_size,
        n_mels,
        model_size,
        loss_function=None,
        n_mega_blocks_trials=None,
    ):
        """
        Find the best number of mega blocks s.t. the spawned TitaNet model
        has the closest number of parameters to the given target ones
        """
        if n_mega_blocks_trials is None:
            n_mega_blocks_trials = list(range(1, 20))
        target_params = cls.TARGET_PARAMS[model_size]
        best_value, min_distance = None, np.inf
        for n_mega_blocks in n_mega_blocks_trials:
            titanet = cls.get_titanet(
                embedding_size=embedding_size,
                n_mels=n_mels,
                n_mega_blocks=n_mega_blocks,
                model_size=model_size,
                loss_function=loss_function,
            )
            params = titanet.get_n_params(div=1e6)
            distance = abs(target_params - params)
            if distance < min_distance:
                best_value = n_mega_blocks
                min_distance = distance
        return best_value

    @classmethod
    def get_titanet(
        cls,
        embedding_size=192,
        n_mels=80,
        n_mega_blocks=None,
        model_size="s",
        attention_hidden_size=128,
        simple_pool=False,
        loss_function=None,
        dropout=0.5,
        device="cpu",
    ):
        """
        Return one of the three TitaNet instances described in the paper,
        i.e. TitaNet-S, TitaNet-M or TitaNet-L
        """
        assert isinstance(model_size, str) and model_size.lower() in (
            "s",
            "m",
            "l",
        ), "Unsupported model size"
        assert (
            isinstance(loss_function, losses.MetricLearningLoss)
            or loss_function is None
        ), "Unsupported loss function"

        # Get the best number of mega blocks
        if n_mega_blocks is None:
            n_mega_blocks = cls.find_n_mega_blocks(
                embedding_size, n_mels, model_size, loss_function=loss_function
            )

        # Assign parameters common to all model sizes
        titanet = partial(
            TitaNet,
            n_mels=n_mels,
            n_mega_blocks=n_mega_blocks,
            n_sub_blocks=3,
            encoder_output_size=1536,
            embedding_size=embedding_size,
            attention_hidden_size=attention_hidden_size,
            simple_pool=simple_pool,
            loss_function=loss_function,
            dropout=dropout,
            device=device,
        )

        # Return the selected model size
        if model_size.lower() == "s":
            return titanet(encoder_hidden_size=256, mega_block_kernel_size=3)
        elif model_size.lower() == "m":
            return titanet(encoder_hidden_size=512, mega_block_kernel_size=7)
        elif model_size.lower() == "l":
            return titanet(encoder_hidden_size=1024, mega_block_kernel_size=11)

    def forward(self, spectrograms, speakers=None):
        """
        Given input spectrograms of shape [B, M, T], TitaNet returns
        utterance-level embeddings of shape [B, E]

        B: batch size
        M: number of mel frequency bands
        T: maximum number of time steps (frames)
        E: embedding size
        """
        encodings = self.encoder(spectrograms)
        embeddings = self.decoder(encodings)

        # Inference mode
        if speakers is None:
            return F.normalize(embeddings, p=2, dim=1)

        # Training mode
        assert (
            self.loss_function is not None
        ), "Loss function should not be None in training mode"
        return self.loss_function(embeddings, speakers)


class Encoder(nn.Module):
    """
    The TitaNet encoder starts with a prologue block, followed by a number
    of mega blocks and ends with an epilogue block; all blocks comprise
    convolutions, batch normalization, activation and dropout, while mega
    blocks are also equipped with residual connections and SE modules

    "TitaNet: Neural Model for speaker representation with 1D Depth-wise
    separable convolutions and global context", Kologuri et al.,
    https://arxiv.org/abs/2110.04410
    """

    def __init__(
        self,
        n_mels,
        n_mega_blocks,
        n_sub_blocks,
        hidden_size,
        output_size,
        mega_block_kernel_size,
        prolog_kernel_size=3,
        epilog_kernel_size=1,
        se_reduction=16,
        dropout=0.5,
    ):
        super(Encoder, self).__init__()

        # Define encoder as sequence of prolog, mega blocks and epilog
        self.prolog = modules.ConvBlock1d(n_mels, hidden_size, prolog_kernel_size)
        self.mega_blocks = nn.Sequential(
            *[
                MegaBlock(
                    hidden_size,
                    hidden_size,
                    mega_block_kernel_size,
                    n_sub_blocks,
                    se_reduction=se_reduction,
                    dropout=dropout,
                )
                for _ in range(n_mega_blocks)
            ]
        )
        self.epilog = modules.ConvBlock1d(hidden_size, output_size, epilog_kernel_size)

    def forward(self, spectrograms):
        """
        Given input spectrograms of shape [B, M, T], return encodings
        of shape [B, DE, T]

        B: batch size
        M: number of mel frequency bands
        T: maximum number of time steps (frames)
        DE: encoding output size
        H: hidden size
        """
        # [B, M, T] -> [B, H, T]
        prolog_outputs = self.prolog(spectrograms)

        # [B, H, T] -> [B, H, T]
        mega_blocks_outputs = self.mega_blocks(prolog_outputs)

        # [B, H, T] -> [B, DE, T]
        return self.epilog(mega_blocks_outputs)


class MegaBlock(nn.Module):
    """
    The TitaNet mega block, part of its encoder, comprises a sequence
    of sub-blocks, where each one contains a time-channel separable
    convolution followed by batch normalization, activation and dropout;
    the output of the sequence of sub-blocks is then processed by a SE
    module and merged with the initial input through a skip connection

    "TitaNet: Neural Model for speaker representation with 1D Depth-wise
    separable convolutions and global context", Kologuri et al.,
    https://arxiv.org/abs/2110.04410
    """

    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        n_sub_blocks,
        se_reduction=16,
        dropout=0.5,
    ):
        super(MegaBlock, self).__init__()

        # Store attributes
        self.dropout = dropout

        # Define sub-blocks composed of depthwise convolutions
        channels = [input_size] + [output_size] * n_sub_blocks
        self.sub_blocks = nn.Sequential(
            *[
                modules.ConvBlock1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    activation="relu",
                    dropout=dropout,
                    depthwise=True,
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ],
            modules.SqueezeExcitation(output_size, reduction=se_reduction)
        )

        # Define the final skip connection
        self.skip_connection = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=1),
            nn.BatchNorm1d(output_size),
        )

    def forward(self, prolog_outputs):
        """
        Given prolog outputs of shape [B, H, T], return
        a feature tensor of shape [B, H, T]

        B: batch size
        H: hidden size
        T: maximum number of time steps (frames)
        """
        # [B, H, T] -> [B, H, T]
        mega_block_outputs = self.skip_connection(prolog_outputs) + self.sub_blocks(
            prolog_outputs
        )
        return F.dropout(
            F.relu(mega_block_outputs), p=self.dropout, training=self.training
        )


class Decoder(nn.Module):
    """
    The TitaNet decoder computes intermediate time-independent features
    using an attentive statistics pooling layer and downsamples such
    representation using two linear layers, to obtain a fixed-size
    embedding vector first and class logits afterwards

    "TitaNet: Neural Model for speaker representation with 1D Depth-wise
    separable convolutions and global context", Kologuri et al.,
    https://arxiv.org/abs/2110.04410
    """

    def __init__(
        self,
        encoder_output_size,
        attention_hidden_size,
        embedding_size,
        simple_pool=False,
    ):
        super(Decoder, self).__init__()

        # Define the attention/pooling layer
        if simple_pool:
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                modules.Squeeze(-1),
                nn.Linear(encoder_output_size, encoder_output_size * 2),
            )
        else:
            self.pool = nn.Sequential(
                AttentiveStatsPooling(encoder_output_size, attention_hidden_size),
                nn.BatchNorm1d(encoder_output_size * 2),
            )

        # Define the final classifier
        self.linear = nn.Sequential(
            nn.Linear(encoder_output_size * 2, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, encodings):
        """
        Given encoder outputs of shape [B, DE, T], return a tensor
        of shape [B, E]

        B: batch size
        T: maximum number of time steps (frames)
        DE: encoding output size
        E: embedding size
        """
        # [B, DE, T] -> [B, DE * 2]
        pooled = self.pool(encodings)

        # [B, DE * 2] -> [B, E]
        return self.linear(pooled)


class AttentiveStatsPooling(nn.Module):
    """
    The attentive statistics pooling layer uses an attention
    mechanism to give different weights to different frames and
    generates not only weighted means but also weighted variances,
    to form utterance-level features from frame-level features

    "Attentive Statistics Pooling for Deep Speaker Embedding",
    Okabe et al., https://arxiv.org/abs/1803.10963
    """

    def __init__(self, input_size, hidden_size, eps=1e-6):
        super(AttentiveStatsPooling, self).__init__()

        # Store attributes
        self.eps = eps

        # Define architecture
        self.in_linear = nn.Linear(input_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, input_size)

    def forward(self, encodings):
        """
        Given encoder outputs of shape [B, DE, T], return
        pooled outputs of shape [B, DE * 2]

        B: batch size
        T: maximum number of time steps (frames)
        DE: encoding output size
        """
        # Compute a scalar score for each frame-level feature
        # [B, DE, T] -> [B, DE, T]
        energies = self.out_linear(
            torch.tanh(self.in_linear(encodings.transpose(1, 2)))
        ).transpose(1, 2)

        # Normalize scores over all frames by a softmax function
        # [B, DE, T] -> [B, DE, T]
        alphas = torch.softmax(energies, dim=2)

        # Compute mean vector weighted by normalized scores
        # [B, DE, T] -> [B, DE]
        means = torch.sum(alphas * encodings, dim=2)

        # Compute std vector weighted by normalized scores
        # [B, DE, T] -> [B, DE]
        residuals = torch.sum(alphas * encodings ** 2, dim=2) - means ** 2
        stds = torch.sqrt(residuals.clamp(min=self.eps))

        # Concatenate mean and std vectors to produce
        # utterance-level features
        # [[B, DE]; [B, DE]] -> [B, DE * 2]
        return torch.cat([means, stds], dim=1)