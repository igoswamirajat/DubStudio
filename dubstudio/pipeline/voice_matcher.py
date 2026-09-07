from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from dubstudio.pipeline.voice_profile import AcousticProfile, extract_acoustic_profile

log = logging.getLogger("dubstudio.voice_matcher")


@dataclass
class VoiceCandidate:
    voice_id: str
    gender: str
    target_f0: float
    persona: str
    description: str


VEENA_VOICES = {
    "agastya": VoiceCandidate(
        voice_id="agastya",
        gender="male",
        target_f0=130.0,
        persona="tech_explainer",
        description="Energetic, confident, modern tech & tutorial Indian male voice",
    ),
    "vinaya": VoiceCandidate(
        voice_id="vinaya",
        gender="male",
        target_f0=105.0,
        persona="deep_narrator",
        description="Deep, resonant, authoritative, formal Indian male narrator",
    ),
    "kavya": VoiceCandidate(
        voice_id="kavya",
        gender="female",
        target_f0=225.0,
        persona="bright_dynamic",
        description="Dynamic, expressive, bright, lively Indian female voice",
    ),
    "maitri": VoiceCandidate(
        voice_id="maitri",
        gender="female",
        target_f0=185.0,
        persona="warm_conversational",
        description="Warm, soothing, calm, conversational Indian female voice",
    ),
}


def score_voice_match(profile: AcousticProfile, candidate: VoiceCandidate) -> float:
    """Calculate matching score between speaker acoustic profile and a voice candidate (higher is better)."""
    # Hard gender match penalty
    if profile.gender != candidate.gender:
        return -100.0

    score = 100.0

    # Pitch distance penalty (relative difference)
    pitch_diff = abs(profile.f0_median - candidate.target_f0)
    score -= min(50.0, pitch_diff * 0.7)

    # Spectral resonance preference
    if candidate.gender == "male":
        if candidate.voice_id == "vinaya" and profile.spectral_ratio > 1.3:
            score += 15.0  # Deep chest resonance bonus
        elif candidate.voice_id == "agastya" and profile.spectral_ratio <= 1.3:
            score += 15.0  # Modern clear mid-range bonus
    elif candidate.gender == "female":
        if candidate.voice_id == "kavya" and profile.f0_median >= 200.0:
            score += 15.0  # High pitch lively bonus
        elif candidate.voice_id == "maitri" and profile.f0_median < 200.0:
            score += 15.0  # Soothing warmer tone bonus

    return score


def match_voices_for_job(
    speakers: list[dict],
    job_dir: Path,
    tts_engine: str = "veena",
    overrides: dict | None = None,
) -> dict[str, dict]:
    """Intelligently assign the best voice to each speaker/character in the video.

    - For Single-Speaker Videos:
      Picks the single most fitting native voice matching the presenter's exact
      vocal profile and locks it across all segments.
    - For Multi-Speaker Videos:
      Assigns distinct, contrasting voices to avoid confusion between characters,
      guaranteeing character voice persistence.
    """
    overrides = overrides or {}
    results: dict[str, dict] = {}

    # Extract acoustic profile for each speaker from their ref.wav
    profiles: dict[str, AcousticProfile] = {}
    for sp in speakers:
        sid = sp["speaker_id"]
        ref_path = job_dir / "voices" / sid / "ref.wav"
        profiles[sid] = extract_acoustic_profile(ref_path, speaker_id=sid)

    # If engine is not Veena (e.g. OmniVoice clone), use clone mode
    if tts_engine != "veena":
        for sp in speakers:
            sid = sp["speaker_id"]
            prof = profiles[sid]
            results[sid] = {
                "voice_id": sid,
                "voice_mode": "clone",
                "gender": prof.gender,
                "f0_median": prof.f0_median,
                "persona": "cloned",
                "confidence": 0.95,
            }
        return results

    # Veena Engine Voice Assignment:
    # Check overrides first
    assigned_voices: set[str] = set()

    for sp in speakers:
        sid = sp["speaker_id"]
        ov = overrides.get(sid, {})
        req_voice = ov.get("voice_id")
        if req_voice and req_voice in VEENA_VOICES:
            results[sid] = {
                "voice_id": req_voice,
                "voice_mode": "native",
                "gender": VEENA_VOICES[req_voice].gender,
                "f0_median": profiles[sid].f0_median,
                "persona": VEENA_VOICES[req_voice].persona,
                "description": VEENA_VOICES[req_voice].description,
                "confidence": 1.0,
            }
            assigned_voices.add(req_voice)

    # For remaining speakers, rank candidates
    remaining_sids = [sp["speaker_id"] for sp in speakers if sp["speaker_id"] not in results]

    if len(speakers) == 1 and remaining_sids:
        # Single-speaker video: Find best matching candidate without constraint
        sid = remaining_sids[0]
        prof = profiles[sid]
        best_candidate = max(
            VEENA_VOICES.values(),
            key=lambda c: score_voice_match(prof, c),
        )
        conf = round(min(0.99, max(0.50, score_voice_match(prof, best_candidate) / 100.0)), 2)
        results[sid] = {
            "voice_id": best_candidate.voice_id,
            "voice_mode": "native",
            "gender": best_candidate.gender,
            "f0_median": prof.f0_median,
            "persona": best_candidate.persona,
            "description": best_candidate.description,
            "confidence": conf,
        }
        log.info(
            "Single-speaker matched: [%s] -> %s (%s, %.1f Hz, conf=%.2f)",
            sid,
            best_candidate.voice_id,
            best_candidate.persona,
            prof.f0_median,
            conf,
        )
        return results

    # Multi-speaker assignment:
    for sid in remaining_sids:
        prof = profiles[sid]
        # Sort candidate voices by matching score
        candidates = sorted(
            VEENA_VOICES.values(),
            key=lambda c: score_voice_match(prof, c),
            reverse=True,
        )

        # Pick best unassigned voice of matching gender, or fallback to best scored
        chosen = None
        for c in candidates:
            if c.voice_id not in assigned_voices and c.gender == prof.gender:
                chosen = c
                break

        if chosen is None:
            # Fallback to best unassigned of any gender
            for c in candidates:
                if c.voice_id not in assigned_voices:
                    chosen = c
                    break

        if chosen is None:
            # If more speakers than available voices, recycle highest scored
            chosen = candidates[0]

        assigned_voices.add(chosen.voice_id)
        conf = round(min(0.99, max(0.50, score_voice_match(prof, chosen) / 100.0)), 2)
        results[sid] = {
            "voice_id": chosen.voice_id,
            "voice_mode": "native",
            "gender": chosen.gender,
            "f0_median": prof.f0_median,
            "persona": chosen.persona,
            "description": chosen.description,
            "confidence": conf,
        }
        log.info(
            "Multi-speaker matched: [%s] -> %s (%s, %.1f Hz, conf=%.2f)",
            sid,
            chosen.voice_id,
            chosen.persona,
            prof.f0_median,
            conf,
        )

    return results
