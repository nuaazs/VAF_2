import speechbrain
import torch
import torchaudio
from speechbrain.pretrained import Pretrained
from speechbrain.pretrained.fetching import fetch
from speechbrain.utils.data_utils import split_path
import multiprocessing
import torch.nn.functional as F
import copy
from utils.oss import upload_files,upload_file
import cfg
class lyxx_VAD(Pretrained):
    """A ready-to-use class for Voice Activity Detection (VAD) using a
    pre-trained model.
    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.pretrained import VAD
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> VAD = VAD.from_hparams(
    ...     source="speechbrain/vad-crdnn-libriparty",
    ...     savedir=tmpdir,
    ... )
    >>> # Perform VAD
    >>> boundaries = VAD.get_speech_segments("tests/samples/single-mic/example1.wav")
    """

    HPARAMS_NEEDED = ["sample_rate", "time_resolution", "device"]

    MODULES_NEEDED = ["compute_features", "mean_var_norm", "model"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_resolution = self.hparams.time_resolution
        self.sample_rate = self.hparams.sample_rate
        self.device = self.hparams.device

    def get_speech_prob_file(
        self,
        wav_data,
        large_chunk_size=30,
        small_chunk_size=10,
        overlap_small_chunk=False,
    ):
        """Outputs the frame-level speech probability of the input audio file
        using the neural model specified in the hparam file. To make this code
        both parallelizable and scalable to long sequences, it uses a
        double-windowing approach.  First, we sequentially read non-overlapping
        large chunks of the input signal.  We then split the large chunks into
        smaller chunks and we process them in parallel.
        Arguments
        ---------
        wav_data: # (1,8000*n)
        large_chunk_size: float
            Size (in seconds) of the large chunks that are read sequentially
            from the input audio file.
        small_chunk_size:
            Size (in seconds) of the small chunks extracted from the large ones.
            The audio signal is processed in parallel within the small chunks.
            Note that large_chunk_size/small_chunk_size must be an integer.
        overlap_small_chunk: bool
            True, creates overlapped small chunks. The probabilities of the
            overlapped chunks are combined using hamming windows.
        Returns
        -------
        prob_vad: torch.Tensor
            Tensor containing the frame-level speech probabilities for the
            input audio file.
        """
        # Getting the total size of the input file
        sample_rate, audio_len = self.sample_rate, wav_data.shape[-1]


        # Computing the length (in samples) of the large and small chunks
        long_chunk_len = int(sample_rate * large_chunk_size)
        small_chunk_len = int(sample_rate * small_chunk_size)

        # Setting the step size of the small chunk (50% overlapping windows are supported)
        small_chunk_step = small_chunk_size
        if overlap_small_chunk:
            small_chunk_step = small_chunk_size / 2

        # Computing the length (in sample) of the small_chunk step size
        small_chunk_len_step = int(sample_rate * small_chunk_step)

        # Loop over big chunks
        prob_chunks = []
        last_chunk = False
        begin_sample = 0
        while True:


            large_chunk = wav_data[:, begin_sample : begin_sample + long_chunk_len]
            # print(f"large_chunk device {large_chunk.device}")
            # large_chunk = large_chunk.to(self.device)

            # Manage padding of the last small chunk
            if last_chunk or large_chunk.shape[-1] < small_chunk_len:
                padding = torch.zeros(
                    1, small_chunk_len, device=large_chunk.device
                )
                large_chunk = torch.cat([large_chunk, padding], dim=1)

            # Splitting the big chunk into smaller (overlapped) ones
            small_chunks = torch.nn.functional.unfold(
                large_chunk.unsqueeze(1).unsqueeze(2),
                kernel_size=(1, small_chunk_len),
                stride=(1, small_chunk_len_step),
            )
            small_chunks = small_chunks.squeeze(0).transpose(0, 1)

            # Getting (in parallel) the frame-level speech probabilities
            small_chunks_prob = self.get_speech_prob_chunk(small_chunks)
            small_chunks_prob = small_chunks_prob[:, :-1, :]

            # Manage overlapping chunks
            if overlap_small_chunk:
                small_chunks_prob = self._manage_overlapped_chunks(
                    small_chunks_prob
                )

            # Prepare for folding
            small_chunks_prob = small_chunks_prob.permute(2, 1, 0)

            # Computing lengths in samples
            out_len = int(
                large_chunk.shape[-1] / (sample_rate * self.time_resolution)
            )
            kernel_len = int(small_chunk_size / self.time_resolution)
            step_len = int(small_chunk_step / self.time_resolution)

            # Folding the frame-level predictions
            small_chunks_prob = torch.nn.functional.fold(
                small_chunks_prob,
                output_size=(1, out_len),
                kernel_size=(1, kernel_len),
                stride=(1, step_len),
            )

            # Appending the frame-level speech probabilities of the large chunk
            small_chunks_prob = small_chunks_prob.squeeze(1).transpose(-1, -2)
            prob_chunks.append(small_chunks_prob)

            # Check stop condition
            if last_chunk:
                break

            # Update counter to process the next big chunk
            begin_sample = begin_sample + long_chunk_len

            # Check if the current chunk is the last one
            if begin_sample + long_chunk_len > audio_len:
                last_chunk = True

        # Converting the list to a tensor
        prob_vad = torch.cat(prob_chunks, dim=1)
        last_elem = int(audio_len / (self.time_resolution * sample_rate))
        prob_vad = prob_vad[:, 0:last_elem, :]

        return prob_vad

    def _manage_overlapped_chunks(self, small_chunks_prob):
        """This support function manages overlapped the case in which the
        small chunks have a 50% overlap."""

        # Weighting the frame-level probabilities with a hamming window
        # reduces uncertainty when overlapping chunks are used.
        hamming_window = torch.hamming_window(
            small_chunks_prob.shape[1], device=small_chunks_prob.device
        )

        # First and last chunks require special care
        half_point = int(small_chunks_prob.shape[1] / 2)
        small_chunks_prob[0, half_point:] = small_chunks_prob[
            0, half_point:
        ] * hamming_window[half_point:].unsqueeze(1)
        small_chunks_prob[-1, 0:half_point] = small_chunks_prob[
            -1, 0:half_point
        ] * hamming_window[0:half_point].unsqueeze(1)

        # Applying the window to all the other probabilities
        small_chunks_prob[1:-1] = small_chunks_prob[
            1:-1
        ] * hamming_window.unsqueeze(0).unsqueeze(2)

        return small_chunks_prob

    def get_speech_prob_chunk(self, wavs, wav_lens=None):
        # print(f"wavs device {wavs.device}")
        """Outputs the frame-level posterior probability for the input audio chunks
        Outputs close to zero refers to time steps with a low probability of speech
        activity, while outputs closer to one likely contain speech.
        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()
        # print(f"wavs device {wavs.device}")

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        outputs = self.mods.cnn(feats)

        outputs = outputs.reshape(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] * outputs.shape[3],
        )

        outputs, h = self.mods.rnn(outputs)
        outputs = self.mods.dnn(outputs)
        output_prob = torch.sigmoid(outputs)
        # output_prob = output_prob.to("cpu")
        return output_prob

    def apply_threshold(
        self, vad_prob, activation_th=0.5, deactivation_th=0.25
    ):
        """Scans the frame-level speech probabilities and applies a threshold
        on them. Speech starts when a value larger than activation_th is
        detected, while it ends when observing a value lower than
        the deactivation_th.
        Arguments
        ---------
        vad_prob: torch.Tensor
            Frame-level speech probabilities.
        activation_th:  float
            Threshold for starting a speech segment.
        deactivation_th: float
            Threshold for ending a speech segment.
        Returns
        -------
        vad_th: torch.Tensor
            Tensor containing 1 for speech regions and 0 for non-speech regions.
       """
        # print("===========apply_threshold===========")
        # vad_activation = (vad_prob >= activation_th).int()
        # vad_deactivation = (vad_prob >= deactivation_th).int()
        # vad_th = vad_activation + vad_deactivation
        # vad_th_old = vad_th.clone()
        # vad_th_new = vad_th.clone()
        # # Loop over batches and time steps
        # for batch in range(vad_th_old.shape[0]):
        #     for time_step in range(vad_th_old.shape[1] - 1):
        #         if (
        #             vad_th_old[batch, time_step] == 2
        #             and vad_th_old[batch, time_step + 1] == 1
        #         ):
        #             vad_th_old[batch, time_step + 1] = 2

        # vad_th_old[vad_th_old == 1] = 0
        # vad_th_old[vad_th_old == 2] = 1
        # print(vad_th_old[0,:,0])
        vad_prob[vad_prob >= activation_th] = 1
        vad_prob[vad_prob < activation_th] = 0
        return vad_prob

    def get_boundaries(self, prob_th, output_value="seconds"):
        """Computes the time boundaries where speech activity is detected.
        It takes in input frame-level binary decisions
        (1 for speech, 0 for non-speech) and outputs the begin/end second
        (or sample) of each detected speech region.
        Arguments
        ---------
        prob_th: torch.Tensor
            Frame-level binary decisions (1 for speech frame, 0 for a
            non-speech one).  The tensor can be obtained from apply_threshold.
        output_value: 'seconds' or 'samples'
            When the option 'seconds' is set, the returned boundaries are in
            seconds, otherwise, it reports them in samples.
        Returns
        -------
        boundaries: torch.Tensor
            Tensor containing the start second (or sample) of speech segments
            in even positions and their corresponding end in odd positions
            (e.g, [1.0, 1.5, 5,.0 6.0] means that we have two speech segment;
             one from 1.0 to 1.5 seconds and another from 5.0 to 6.0 seconds).
       """
        # Shifting frame-levels binary decision by 1
        # This allows detecting changes in speech/non-speech activities
        prob_th_shifted = torch.roll(prob_th, dims=1, shifts=1)
        prob_th_shifted[:, 0, :] = 0
        prob_th = prob_th + prob_th_shifted

        # Needed to first and last time step
        prob_th[:, 0, :] = (prob_th[:, 0, :] >= 1).int()
        prob_th[:, -1, :] = (prob_th[:, -1, :] >= 1).int()

        # Fix edge cases (when a speech starts in the last frames)
        if (prob_th == 1).nonzero().shape[0] % 2 == 1:
            prob_th = torch.cat(
                (prob_th, torch.Tensor([1.0]).unsqueeze(0).unsqueeze(2).to(prob_th.device)), dim=1
            )

        # Where prob_th is 1 there is a change
        indexes = (prob_th == 1).nonzero()[:, 1].reshape(-1, 2)

        # Remove 1 from end samples
        indexes[:, -1] = indexes[:, -1] - 1

        # From indexes to samples
        seconds = (indexes * self.time_resolution).float()
        samples = (self.sample_rate * seconds).round().int()

        if output_value == "seconds":
            boundaries = seconds
        else:
            boundaries = samples
        return boundaries

    def merge_close_segments(self, boundaries, close_th=0.250):
        """Merges segments that are shorter than the given threshold.
        Arguments
        ---------
        boundaries : str
            Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        close_th: float
            If the distance between boundaries is smaller than close_th, the
            segments will be merged.
        Returns
        -------
        new_boundaries
            The new boundaries with the merged segments.
        """

        new_boundaries = []

        # Single segment case
        if boundaries.shape[0] == 0:
            return boundaries

        # Getting beg and end of previous segment
        prev_beg_seg = boundaries[0, 0].float()
        prev_end_seg = boundaries[0, 1].float()

        # Process all the segments
        for i in range(1, boundaries.shape[0]):
            beg_seg = boundaries[i, 0]
            segment_distance = beg_seg - prev_end_seg

            # Merging close segments
            if segment_distance <= close_th:
                prev_end_seg = boundaries[i, 1]

            else:
                # Appending new segments
                new_boundaries.append([prev_beg_seg, prev_end_seg])
                prev_beg_seg = beg_seg
                prev_end_seg = boundaries[i, 1]

        new_boundaries.append([prev_beg_seg, prev_end_seg])
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries

    def remove_short_segments(self, boundaries, len_th=0.250):
        """Removes segments that are too short.
        Arguments
        ---------
        boundaries : torch.Tensor
            Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        len_th: float
            If the length of the segment is smaller than close_th, the segments
            will be merged.
        Returns
        -------
        new_boundaries
            The new boundaries without the short segments.
        """
        new_boundaries = []

        # Process the segments
        for i in range(boundaries.shape[0]):
            # Computing segment length
            seg_len = boundaries[i, 1] - boundaries[i, 0]

            # Accept segment only if longer than len_th
            if seg_len > len_th:
                new_boundaries.append([boundaries[i, 0], boundaries[i, 1]])
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)

        return new_boundaries

    def energy_VAD_before_nn(self,
                             wav_data,
                             activation_th=0.5,
                             deactivation_th=0.0,
                             eps=1e-6):
        # Getting the total size of the input file
        sample_rate, audio_len = self.sample_rate, wav_data.shape[-1]

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        # Computing the chunk length of the energy window
        chunk_len = int(self.time_resolution * sample_rate)
        new_boundaries = []

        # Create chunks
        segment_chunks = self.create_chunks(
            wav_data, chunk_size=chunk_len, chunk_stride=chunk_len
        )

        # Energy computation within each chunk
        energy_chunks = segment_chunks.abs().sum(-1) + eps
        energy_chunks = energy_chunks.log()

        # Energy normalization
        energy_chunks = (
            (energy_chunks - energy_chunks.mean())
            / (2 * energy_chunks.std())
        ) + 0.5
        energy_chunks = energy_chunks.unsqueeze(0).unsqueeze(2)

        # Apply threshold based on the energy value
        energy_vad = self.apply_threshold(
            energy_chunks,
            activation_th=activation_th,
            deactivation_th=deactivation_th,
        )

        # Get the boundaries
        energy_boundaries = self.get_boundaries(
            energy_vad, output_value="seconds"
        )

        # Get the final boundaries in the original signal
        for j in range(energy_boundaries.shape[0]):
            start_en = energy_boundaries[j, 0]
            end_end = energy_boundaries[j, 1]
            new_boundaries.append([start_en, end_end])
        # Convert boundaries to tensor
        new_boundaries = torch.FloatTensor(new_boundaries).to(wav_data.device)
        return new_boundaries
    

    def energy_VAD(
        self,
        wav_data,
        boundaries,
        activation_th=0.5,
        deactivation_th=0.0,
        eps=1e-6,
    ):
        """Applies energy-based VAD within the detected speech segments.The neural
        network VAD often creates longer segments and tends to merge segments that
        are close with each other.
        The energy VAD post-processes can be useful for having a fine-grained voice
        activity detection.
        The energy VAD computes the energy within the small chunks. The energy is
        normalized within the segment to have mean 0.5 and +-0.5 of std.
        This helps to set the energy threshold.
        Arguments
        ---------
        audio_file: path
            Path of the audio file containing the recording. The file is read
            with torchaudio.
        boundaries : torch.Tensor
            Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        activation_th: float
            A new speech segment is started it the energy is above activation_th.
        deactivation_th: float
            The segment is considered ended when the energy is <= deactivation_th.
        eps: float
            Small constant for numerical stability.
        Returns
        -------
        new_boundaries
            The new boundaries that are post-processed by the energy VAD.
        """

        # Getting the total size of the input file
        sample_rate, audio_len = self.sample_rate, wav_data.shape[-1]

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        # Computing the chunk length of the energy window
        chunk_len = int(self.time_resolution * sample_rate)
        new_boundaries = []

        # Processing speech segments
        for i in range(boundaries.shape[0]):
            begin_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            seg_len = end_sample - begin_sample

            if seg_len/sample_rate < 0.1:
                continue
            segment, _ = wav_data[:, begin_sample:end_sample], self.sample_rate


            # Create chunks
            segment_chunks = self.create_chunks(
                segment, chunk_size=chunk_len, chunk_stride=chunk_len
            )

            # Energy computation within each chunk
            energy_chunks = segment_chunks.abs().sum(-1) + eps
            energy_chunks = energy_chunks.log()

            # Energy normalization
            energy_chunks = (
                (energy_chunks - energy_chunks.mean())
                / (2 * energy_chunks.std())
            ) + 0.5
            energy_chunks = energy_chunks.unsqueeze(0).unsqueeze(2)

            # Apply threshold based on the energy value
            energy_vad = self.apply_threshold(
                energy_chunks,
                activation_th=activation_th,
                deactivation_th=deactivation_th,
            )

            # Get the boundaries
            energy_boundaries = self.get_boundaries(
                energy_vad, output_value="seconds"
            )

            # Get the final boundaries in the original signal
            for j in range(energy_boundaries.shape[0]):
                start_en = boundaries[i, 0] + energy_boundaries[j, 0]
                end_end = boundaries[i, 0] + energy_boundaries[j, 1]
                new_boundaries.append([start_en, end_end])

        # Convert boundaries to tensor
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries

    def create_chunks(self, x, chunk_size=16384, chunk_stride=16384):
        """Splits the input into smaller chunks of size chunk_size with
        an overlap chunk_stride. The chunks are concatenated over
        the batch axis.
        Arguments
        ---------
        x: torch.Tensor
            Signal to split into chunks.
        chunk_size : str
            The size of each chunk.
        chunk_stride:
            The stride (hop) of each chunk.
        Returns
        -------
        x: torch.Tensor
            A new tensors with the chunks derived from the input signal.
        """
        x = x.unfold(1, chunk_size, chunk_stride)
        x = x.reshape(x.shape[0] * x.shape[1], -1)
        return x


    def upsample_VAD(self, vad_out, wav_data, time_resolution=0.01):
        """Upsamples the output of the vad to help visualization. It creates a
        signal that is 1 when there is speech and 0 when there is no speech.
        The vad signal has the same resolution as the input one and can be
        opened with it (e.g, using audacity) to visually figure out VAD regions.
        Arguments
        ---------
        vad_out: torch.Tensor
            Tensor containing 1 for each frame of speech and 0 for each non-speech
            frame.
        wav_data
        time_resolution : float
            Time resolution of the vad_out signal.
        Returns
        -------
        vad_signal
            The upsampled version of the vad_out tensor.
        """

        # Getting the total size of the input file
        sample_rate, sig_len = self.sample_rate, wav_data.shape[-1]


        beg_samp = 0
        step_size = int(time_resolution * sample_rate)
        end_samp = step_size
        index = 0

        # Initialize upsampled signal
        vad_signal = torch.zeros(1, sig_len, device=vad_out.device)

        # Upsample signal
        while end_samp < sig_len:
            vad_signal[0, beg_samp:end_samp] = vad_out[0, index, 0]
            index = index + 1
            beg_samp = beg_samp + step_size
            end_samp = beg_samp + step_size
        return vad_signal

    def upsample_boundaries(self, boundaries, wav_data):
        """Based on the input boundaries, this method creates a signal that is 1
        when there is speech and 0 when there is no speech.
        The vad signal has the same resolution as the input one and can be
        opened with it (e.g, using audacity) to visually figure out VAD regions.
        Arguments
        ---------
        boundaries: torch.Tensor
            Tensor containing the boundaries of the speech segments.
        wav_data
        Returns
        -------
        vad_signal
            The output vad signal with the same resolution of the input one.
        """

        # Getting the total size of the input file
        sample_rate, sig_len = self.sample_rate, wav_data.shape[-1]

        # Initialization of the output signal
        vad_signal = torch.zeros(1, sig_len, device=boundaries.device)

        # Composing the vad signal from boundaries
        for i in range(boundaries.shape[0]):
            beg_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            vad_signal[0, beg_sample:end_sample] = 1.0
        return vad_signal
    def double_check_speech_segments(
        self, boundaries, wav_data, speech_th=0.5
    ):
        """Takes in input the boundaries of the detected speech segments and
        double checks (using the neural VAD) that they actually contain speech.
        Arguments
        ---------
        boundaries: torch.Tensor
            Tensor containing the boundaries of the speech segments.
        audio_file: path
            The original audio file used to compute vad_out.
        speech_th: float
            Threshold on the mean posterior probability over which speech is
            confirmed. Below that threshold, the segment is re-assigned to a
            non-speech region.
        Returns
        -------
        new_boundaries
            The boundaries of the segments where speech activity is confirmed.
        """

        # Getting the total size of the input file
        # sample_rate, sig_len = self._get_audio_info(audio_file)
        sample_rate, sig_len = self.sample_rate, wav_data.shape[-1]

        # Double check the segments
        new_boundaries = []
        for i in range(boundaries.shape[0]):
            beg_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            len_seg = end_sample - beg_sample

            # Read the candidate speech segment
            # segment, fs = torchaudio.load(
            #     audio_file, frame_offset=beg_sample, num_frames=len_seg
            # )
            segment = wav_data[:, beg_sample:end_sample]
            speech_prob = self.get_speech_prob_chunk(segment)
            if speech_prob.mean() > speech_th:
                # Accept this as a speech segment
                new_boundaries.append([boundaries[i, 0], boundaries[i, 1]])

        # Convert boundaries from list to tensor
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries

    def get_segments(
        self, boundaries, wav_data, before_margin=0.1, after_margin=0.1
    ):
        """Returns a list containing all the detected speech segments.
        Arguments
        ---------
        boundaries: torch.Tensor
            Tensor containing the boundaries of the speech segments.
        wav_data
        before_margin: float
            Used to cut the segments samples a bit before the detected margin.
        after_margin: float
            Use to cut the segments samples a bit after the detected margin.
        Returns
        -------
        segments: list
            List containing the detected speech segments
        """

        sample_rate, sig_len = self.sample_rate, wav_data.shape[-1]

     

        segments = []
        for i in range(boundaries.shape[0]):
            beg_sample = boundaries[i, 0] * sample_rate
            end_sample = boundaries[i, 1] * sample_rate

            beg_sample = int(max(0, beg_sample - before_margin * sample_rate))
            end_sample = int(
                min(sig_len, end_sample + after_margin * sample_rate)
            )

            len_seg = end_sample - beg_sample
            vad_segment,_ = wav_data[:, beg_sample : beg_sample + len_seg]
            segments.append(vad_segment)
        return segments

    def get_speech_segments(
        self,
        wav_data,
        large_chunk_size=30,
        small_chunk_size=10,
        overlap_small_chunk=False,
        apply_energy_VAD=False,
        double_check=True,
        close_th=0.250,
        len_th=0.250,
        activation_th=0.5,
        deactivation_th=0.25,
        en_activation_th=0.5,
        en_deactivation_th=0.0,
        speech_th=0.50,
        apply_energy_VAD_before=False,
        en_activation_th_before=0.5,
        en_deactivation_th_before=0.0,
        outinfo=None,
    ):
        """Detects speech segments within the input file. The input signal can
        be both a short or a long recording. The function computes the
        posterior probabilities on large chunks (e.g, 30 sec), that are read
        sequentially (to avoid storing big signals in memory).
        Each large chunk is, in turn, split into smaller chunks (e.g, 10 seconds)
        that are processed in parallel. The pipeline for detecting the speech
        segments is the following:
            1- Compute posteriors probabilities at the frame level.
            2- Apply a threshold on the posterior probability.
            3- Derive candidate speech segments on top of that.
            4- Apply energy VAD within each candidate segment (optional).
            5- Merge segments that are too close.
            6- Remove segments that are too short.
            7- Double check speech segments (optional).
        Arguments
        ---------
        wav_data
        large_chunk_size: float
            Size (in seconds) of the large chunks that are read sequentially
            from the input audio file.
        small_chunk_size: float
            Size (in seconds) of the small chunks extracted from the large ones.
            The audio signal is processed in parallel within the small chunks.
            Note that large_chunk_size/small_chunk_size must be an integer.
        overlap_small_chunk: bool
            If True, it creates overlapped small chunks (with 50% overlap).
            The probabilities of the overlapped chunks are combined using
            hamming windows.
        apply_energy_VAD: bool
            If True, a energy-based VAD is used on the detected speech segments.
            The neural network VAD often creates longer segments and tends to
            merge close segments together. The energy VAD post-processes can be
            useful for having a fine-grained voice activity detection.
            The energy thresholds is  managed by activation_th and
            deactivation_th (see below).
        double_check: bool
            If True, double checks (using the neural VAD) that the candidate
            speech segments actually contain speech. A threshold on the mean
            posterior probabilities provided by the neural network is applied
            based on the speech_th parameter (see below).
        activation_th:  float
            Threshold of the neural posteriors above which starting a speech segment.
        deactivation_th: float
            Threshold of the neural posteriors below which ending a speech segment.
        en_activation_th: float
            A new speech segment is started it the energy is above activation_th.
            This is active only if apply_energy_VAD is True.
        en_deactivation_th: float
            The segment is considered ended when the energy is <= deactivation_th.
            This is active only if apply_energy_VAD is True.
        speech_th: float
            Threshold on the mean posterior probability within the candidate
            speech segment. Below that threshold, the segment is re-assigned to
            a non-speech region. This is active only if double_check is True.
        close_th: float
            If the distance between boundaries is smaller than close_th, the
            segments will be merged.
        len_th: float
            If the length of the segment is smaller than close_th, the segments
            will be merged.
        Returns
        -------
        boundaries: torch.Tensor
            Tensor containing the start second of speech segments in even
            positions and their corresponding end in odd positions
            (e.g, [1.0, 1.5, 5,.0 6.0] means that we have two speech segment;
             one from 1.0 to 1.5 seconds and another from 5.0 to 6.0 seconds).
        """
        wav_data = wav_data.to("cuda:0")
        assert wav_data.device == torch.device("cuda:0")
        # Computing speech vs non speech probabilities
        prob_chunks = self.get_speech_prob_file(
            wav_data,
            large_chunk_size=large_chunk_size,
            small_chunk_size=small_chunk_size,
            overlap_small_chunk=overlap_small_chunk,
        )
        if outinfo:
            # =========================LOG TIME=========================
            outinfo.log_time(name="vad:get_speech_prob_file_used_time")
        # prob_chunks = prob_chunks.to("cuda:0")
        assert prob_chunks.device == torch.device("cuda:0")
        # move prob_chunks to cpu
        # prob_chunks = prob_chunks.to("cuda:0")
        # print(prob_chunks.device)
        assert prob_chunks.device == torch.device("cuda:0")
        prob_th = self.apply_threshold(
            prob_chunks,
            activation_th=activation_th,
            deactivation_th=deactivation_th,
        ).float()
        assert prob_th.device == torch.device("cuda:0")
        # prob_th = prob_th.to("cuda:0")
        # Compute the boundaries of the speech segments
        # assert prob_th.device == torch.device("cuda:0")
        if outinfo:
            # =========================LOG TIME=========================
            outinfo.log_time(name="vad:apply_threshold_used_time")

        boundaries = self.get_boundaries(prob_th, output_value="seconds")
        assert boundaries.device == torch.device("cuda:0")
        if outinfo:
            # =========================LOG TIME=========================
            outinfo.log_time(name="vad:get_boundaries_used_time")

        if apply_energy_VAD:
            boundaries = self.energy_VAD(
                wav_data,
                boundaries,
                activation_th=en_activation_th,
                deactivation_th=en_deactivation_th,
            )
        if outinfo:
            # =========================LOG TIME=========================
            outinfo.log_time(name="vad:apply_energy_VAD_used_time")

        # Merge short segments
        boundaries = self.merge_close_segments(boundaries, close_th=close_th)
        assert boundaries.device == torch.device("cuda:0")
        if outinfo:
            # =========================LOG TIME=========================
            outinfo.log_time(name="vad:merge_close_segments_used_time")
        # # Remove short segments
        boundaries = self.remove_short_segments(boundaries, len_th=len_th)
        assert boundaries.device == torch.device("cuda:0")
        if outinfo:
            # =========================LOG TIME=========================
            outinfo.log_time(name="vad:remove_short_segments_used_time")

        # Double check speech segments
        if double_check:
            assert boundaries.device == torch.device("cuda:0")
            assert wav_data.device == torch.device("cuda:0")
            boundaries = self.double_check_speech_segments(
                boundaries, wav_data, speech_th=speech_th
            )
            if outinfo:
                # =========================LOG TIME=========================
                outinfo.log_time(name="vad:double_check_speech_segments_used_time")

        return boundaries

    def forward(self, wavs, wav_lens=None):
        """Gets frame-level speech-activity predictions"""
        return self.get_speech_prob_chunk(wavs, wav_lens)


VAD = lyxx_VAD.from_hparams(
        source=f"./models/vad_8k_en_phone_crdnns",
        savedir=f"./pretrained_models/vad_8k_en_phone_crdnns",
        run_opts={"device": cfg.CRDNN_DEVICE},
    )


def get_vad_result(wav,outinfo=None):
    # print(wav.shape)
    assert wav.shape[0] == 1
    assert len(wav.shape) == 2
    assert wav.device == torch.device("cuda:0")
    # wav.to(cfg.DEVICE)
    # if outinfo:
    #     # =========================LOG TIME=========================
    #     outinfo.log_time(name="vad:to_cuda_used_time")
    boundaries = VAD.get_speech_segments(
        wav_data=wav,
        large_chunk_size=cfg.large_chunk_size,
        small_chunk_size=cfg.small_chunk_size,
        overlap_small_chunk=cfg.overlap_small_chunk,
        apply_energy_VAD=cfg.apply_energy_VAD,
        double_check=cfg.double_check,
        close_th=cfg.close_th,
        len_th=cfg.len_th,
        activation_th=cfg.activation_th,
        deactivation_th=cfg.deactivation_th,
        en_activation_th=cfg.en_activation_th,
        en_deactivation_th=cfg.en_deactivation_th,
        speech_th=cfg.speech_th,
        apply_energy_VAD_before=cfg.apply_energy_VAD_before,
        outinfo=outinfo
    )
    if outinfo:
        # =========================LOG TIME=========================
        outinfo.log_time(name="vad:get_speech_segments_used_time")
    upsampled_boundaries = VAD.upsample_boundaries(boundaries, wav)
    output_wav = wav[upsampled_boundaries > 0.5]
    # torchaudio.save("test.wav",output_wav.reshape(1,-1),cfg.SR)
    return output_wav,upsampled_boundaries


def vad(wav, spkid, action_type=None, device=cfg.DEVICE,save=False):
    # wav = wav.to(device)
    assert wav.device == torch.device("cuda:0")
    before_vad_length = len(wav[0]) / cfg.SR

    spk_dir = os.path.join(cfg.TEMP_PATH, str(spkid))
    os.makedirs(spk_dir, exist_ok=True)
    spk_filelist = os.listdir(spk_dir)

    speech_number = len(spk_filelist) + 1
    save_name = f"preprocessed_{spkid}_{speech_number}.wav"
    final_save_path = os.path.join(spk_dir, save_name)

    # after vad wav (tensor)
    output_wav,upsampled_boundaries = get_vad_result(wav,outinfo=outinfo)

    # save
    if save:
        save_audio(final_save_path, output_wav.clone().detach().cpu(), sampling_rate=cfg.SR)
        preprocessed_file_path = upload_file(
            bucket_name="preprocessed",
            filepath=final_save_path,
            filename=save_name,
            save_days=cfg.MINIO["test_save_days"],
        )
    else:
        preprocessed_file_path = ""

    after_vad_length = len(output_wav) / cfg.SR
    # output_wav = torch.FloatTensor(output_wav)
    
    result = {
        "wav_torch": output_wav,
        "before_length": before_vad_length,
        "after_length": after_vad_length,
        "preprocessed_file_path": preprocessed_file_path,
        "boundaries":upsampled_boundaries
    }
    return result