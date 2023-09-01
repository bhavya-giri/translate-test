import torch
from transformers import pipeline
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AutoProcessor, BarkModel
import scipy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transcribe_model_ckpt = "openai/whisper-base"
transcribe_pipe = pipeline(
    task="automatic-speech-recognition",
    model=transcribe_model_ckpt,
    chunk_length_s=30,
    device=device,
)
lang = "en"
transcribe_pipe.model.config.forced_decoder_ids = transcribe_pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")

translate_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

processor = AutoProcessor.from_pretrained("suno/bark")
tts_model = BarkModel.from_pretrained("suno/bark").to(device)
voice_preset = "v2/hi_speaker_2"

def transcibe(audio):
    return transcribe_pipe(audio)["text"]

def translate(text,lang="hi_IN"):
    model_inputs = tokenizer(text, return_tensors="pt")

    generated_tokens = translate_model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[lang]
    )
    text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return text[0]

def pipeline(audio,lang):
    text = transcibe(audio)
    text = translate(text)
    inputs = processor(
    text=[text],
    return_tensors="pt",)
    speech_values = tts_model.generate(**inputs)
    sampling_rate = tts_model.generation_config.sample_rate
    scipy.io.wavfile.write("output.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())
    
pipeline("input.mp3",None)
        