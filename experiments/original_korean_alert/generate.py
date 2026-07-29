from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
OUTPUT_DIR = Path(os.environ.get("VOICE_OUTPUT_DIR", "voice_output"))
TEXT = "아침, 한입!"

# These are broad performance traits for an original character voice.
# They intentionally avoid cloning or imitating any identifiable person.
CANDIDATES = [
    {
        "id": "a_human_bright",
        "seed": 21,
        "instruction": (
            "밝고 생기 있는 성인 한국 여성의 오리지널 캐릭터 목소리. "
            "약간 높은 중음역과 앞쪽 공명, 아주 가벼운 비음, 자연스러운 숨과 미세한 속도 변화를 준다. "
            "아침은 짧고 또렷하게 말하고 아주 잠깐 쉰 뒤, 한입은 미소를 머금고 살짝 올려 친근하게 끝낸다. "
            "광고 성우처럼 과장하지 말고 가까이서 실제 사람이 말하듯 자연스럽게. "
            "어린아이 흉내나 특정 실존 인물의 모사는 하지 않는다."
        ),
    },
    {
        "id": "b_lively_retro",
        "seed": 37,
        "instruction": (
            "명랑한 성인 한국 여성의 오리지널 알림 목소리. "
            "가벼운 복고 대중가요의 리듬감만 살리되 노래하지 말고 말하듯 표현한다. "
            "첫 단어는 통통 튀고 두 번째 단어는 한 박자 여유를 둔 뒤 부드럽게 상승한다. "
            "호흡, 자음 길이, 억양에 작은 인간적인 흔들림을 남기고 기계적인 일정 박자를 피한다. "
            "과도한 애교, 아동 음성, 특정 실존 인물 모사는 하지 않는다."
        ),
    },
    {
        "id": "c_soft_playful",
        "seed": 63,
        "instruction": (
            "따뜻하고 장난기 있는 성인 한국 여성의 오리지널 캐릭터 목소리. "
            "말하기 직전의 작은 미소가 느껴지고 숨결은 부드럽지만 발음은 선명하다. "
            "아침 뒤에 자연스러운 짧은 쉼을 두고 한입을 가볍고 둥글게 올려 말한다. "
            "스튜디오 광고 톤보다 일상 대화에 가깝고, 감정은 밝되 과장하지 않는다. "
            "유명인이나 어린아이를 흉내 내지 않는다."
        ),
    },
]


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    audio = np.asarray(waveform, dtype=np.float32).squeeze()
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"Unexpected waveform shape: {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError("Generated waveform contains NaN or Infinity")

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio * min(1.0, 0.92 / peak)

    fade_samples = min(240, audio.size // 10)
    if fade_samples > 1:
        fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        audio[:fade_samples] *= fade
        audio[-fade_samples:] *= fade[::-1]
    return audio


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )

    manifest = {
        "model": MODEL_ID,
        "speaker": "Sohee",
        "language": "Korean",
        "text": TEXT,
        "identity_policy": "original_non_identifying_voice_only",
        "candidates": [],
    }

    for candidate in CANDIDATES:
        torch.manual_seed(candidate["seed"])
        waveforms, sample_rate = model.generate_custom_voice(
            text=TEXT,
            language="Korean",
            speaker="Sohee",
            instruct=candidate["instruction"],
            max_new_tokens=512,
        )
        audio = normalize_audio(waveforms[0])
        output_path = OUTPUT_DIR / f"morning_bite_{candidate['id']}.wav"
        sf.write(output_path, audio, sample_rate, subtype="PCM_16")
        manifest["candidates"].append(
            {
                "id": candidate["id"],
                "seed": candidate["seed"],
                "duration_seconds": round(audio.size / sample_rate, 3),
                "sample_rate": int(sample_rate),
                "sha256": sha256_file(output_path),
                "file": output_path.name,
            }
        )

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
