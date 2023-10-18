from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio.functional as F
import torch
import gradio as gr

DEFAULT_SAMPLING_RATE = 16000


class AutomaticSpeechRecognition():
    def __init__(self):
        # load model and processor
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-tiny")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny")
        self.model.config.forced_decoder_ids = None

    def transcript(self, raw_audio, sampling_rate):
        input_features = self.processor(
            raw_audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
        # generate token ids
        predicted_ids = self.model.generate(input_features)
        # decode token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=False)

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True)
        return transcription


def resample_audio(raw_audio, sampling_rate, resampling_rate=DEFAULT_SAMPLING_RATE):
    return F.resample(torch.from_numpy(raw_audio), sampling_rate, resampling_rate).numpy()


def simple_ui(asr):
    def transcript(audio):
        sampling_rate, raw_audio = audio
        raw_audio = raw_audio / 2**32
        if sampling_rate != DEFAULT_SAMPLING_RATE:
            raw_audio = resample_audio(raw_audio, sampling_rate)
        return asr.transcript(raw_audio, DEFAULT_SAMPLING_RATE)[0]

    demo = gr.Interface(
        transcript,
        gr.Audio(source="microphone"),
        gr.Text(),
        interpretation="default",
    )

    demo.launch()
