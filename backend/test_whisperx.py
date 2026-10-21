"""Test transcription. Usage: python test_whisperx.py path/to/audio.wav"""
import sys
import whisperx
import torch

# Patch torch.load
_orig = torch.load
def _patch(*a, **kw):
    kw['weights_only'] = False
    return _orig(*a, **kw)
torch.load = _patch

if len(sys.argv) != 2:
    sys.exit('Usage: python test_whisperx.py path/to/audio.wav')
audio_path = sys.argv[1]

print('Loading model...')
model = whisperx.load_model('large-v3', device='cuda', compute_type='float16', language='en', vad_method='silero')

print('Loading alignment model...')
align_model, align_metadata = whisperx.load_align_model(language_code='en', device='cuda')

print('Loading audio...')
audio = whisperx.load_audio(audio_path)

print('Transcribing...')
result = model.transcribe(audio, batch_size=8, language='en')
print(f'Got {len(result.get("segments", []))} segments')

print('Aligning...')
aligned = whisperx.align(
    result["segments"],
    align_model,
    align_metadata,
    audio,
    'cuda',
    return_char_alignments=False
)
print(f'Aligned: {len(aligned.get("segments", []))} segments')

print('SUCCESS!')
