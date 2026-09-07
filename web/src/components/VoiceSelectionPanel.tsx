import { useEffect, useState } from "react";
import {
  Sparkles,
  Volume2,
  Mic,
  Wand2,
  Bot,
  Check,
  Play,
  ArrowRight,
  AlertCircle,
  User,
  ShieldCheck,
} from "lucide-react";
import {
  applyVoiceSelections,
  autoApplyVoiceSelections,
  getVoiceOptions,
  type VoiceOption,
  type SpeakerVoiceInfo,
  type VoiceOptionsResponse,
} from "../api";

type Props = {
  jobId: string;
  onApplied: () => void;
};

type SelectionState = {
  voice_id: string;
  voice_mode: string;
  design_prompt?: string;
};

const DESIGN_PRESETS = [
  "energetic Indian male, conversational tech reviewer",
  "deep authoritative male, calm documentary narrator",
  "young expressive female, bright and cheerful tone",
  "warm soothing female, natural storytelling podcast",
];

export function VoiceSelectionPanel({ jobId, onApplied }: Props) {
  const [data, setData] = useState<VoiceOptionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState("");
  const [selections, setSelections] = useState<Record<string, SelectionState>>({});

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setErr("");
    getVoiceOptions(jobId)
      .then((res) => {
        if (!mounted) return;
        setData(res);
        const initial: Record<string, SelectionState> = {};
        for (const sp of res.speakers) {
          // Default to auto suggestion or clone_original
          const suggested = sp.auto_suggestion?.voice_id || sp.current_voice_id || "clone_original";
          const isClone = suggested === "clone_original" || sp.current_voice_mode === "clone";
          const isDesign = suggested === "design_custom" || sp.current_voice_mode === "design";
          initial[sp.speaker_id] = {
            voice_id: suggested,
            voice_mode: isClone ? "clone" : isDesign ? "design" : "native",
          };
        }
        setSelections(initial);
      })
      .catch((e) => {
        if (mounted) setErr(`Failed to load voice options: ${String(e)}`);
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [jobId]);

  function handleSelectVoice(speakerId: string, voice: VoiceOption) {
    setSelections((prev) => ({
      ...prev,
      [speakerId]: {
        voice_id: voice.voice_id,
        voice_mode: voice.category === "clone" ? "clone" : voice.category === "design" ? "design" : "native",
        design_prompt: prev[speakerId]?.design_prompt || "",
      },
    }));
  }

  function handleDesignPromptChange(speakerId: string, prompt: string) {
    setSelections((prev) => ({
      ...prev,
      [speakerId]: {
        ...prev[speakerId],
        design_prompt: prompt,
      },
    }));
  }

  async function handleApply() {
    setSubmitting(true);
    setErr("");
    try {
      await applyVoiceSelections(jobId, selections);
      onApplied();
    } catch (e) {
      setErr(`Failed to apply voice selections: ${String(e)}`);
      setSubmitting(false);
    }
  }

  async function handleAutoApply() {
    setSubmitting(true);
    setErr("");
    try {
      await autoApplyVoiceSelections(jobId);
      onApplied();
    } catch (e) {
      setErr(`Failed to apply auto suggestions: ${String(e)}`);
      setSubmitting(false);
    }
  }

  if (loading) {
    return (
      <div className="voice-selection-loading-card">
        <Sparkles className="spin-icon text-accent" size={24} />
        <p>Analyzing speaker vocal profiles and loading voice catalog...</p>
      </div>
    );
  }

  if (!data || !data.speakers.length) {
    return null;
  }

  const isSingleSpeaker = data.speakers.length === 1;

  return (
    <section className="studio-card voice-selection-panel-container">
      {/* Paused Banner */}
      <div className="voice-selection-pause-banner">
        <div className="banner-badge">
          <Sparkles size={16} />
          <span>Action Required</span>
        </div>
        <div className="banner-content">
          <h2 className="banner-title">
            {isSingleSpeaker
              ? "Choose Target Voice for Presenter"
              : `Choose Target Voices for ${data.speakers.length} Characters`}
          </h2>
          <p className="banner-desc">
            Video analysis, separation, and translation are complete. Select the exact voice timbre you want for each speaker before speech synthesis begins.
          </p>
        </div>
        <button
          type="button"
          className="btn secondary quick-auto-btn"
          disabled={submitting}
          onClick={handleAutoApply}
          title="Use intelligent AI-matched voices and continue immediately"
        >
          <Bot size={15} />
          <span>Use AI Suggestions</span>
        </button>
      </div>

      {err && (
        <div className="failure-alert" style={{ marginBottom: "1rem" }}>
          <AlertCircle size={18} />
          <div className="alert-content">
            <p>{err}</p>
          </div>
        </div>
      )}

      {/* Speakers List */}
      <div className="voice-selection-speakers-list">
        {data.speakers.map((speaker) => {
          const currentSel = selections[speaker.speaker_id] || {
            voice_id: "clone_original",
            voice_mode: "clone",
          };

          return (
            <div key={speaker.speaker_id} className="voice-speaker-selection-box">
              {/* Speaker Info Header */}
              <div className="voice-speaker-header">
                <div className="voice-speaker-meta-left">
                  <div className="spk-avatar">
                    {(speaker.label || "S").slice(0, 1).toUpperCase()}
                  </div>
                  <div>
                    <h3 className="voice-speaker-title">
                      {speaker.label || `Speaker ${speaker.speaker_id}`}
                    </h3>
                    <div className="voice-speaker-stats">
                      <code>{speaker.speaker_id}</code> · {speaker.segment_count} dialogue segments
                      {speaker.gender && ` · Detected ${speaker.gender}`}
                      {speaker.f0_median && ` (${speaker.f0_median.toFixed(0)} Hz)`}
                    </div>
                  </div>
                </div>

                {/* Vocal Reference Audio Player */}
                <div className="voice-speaker-audio-wrapper">
                  <div className="voice-ref-audio-pill">
                    <Volume2 size={13} className="text-accent" />
                    <span>Original Reference:</span>
                    <audio
                      controls
                      src={speaker.ref_audio_url}
                      preload="metadata"
                      className="inline-ref-audio"
                    />
                  </div>
                </div>
              </div>

              {/* Voice Options Grid */}
              <div className="voice-options-grid">
                {data.available_voices.map((voice) => {
                  const isSelected = currentSel.voice_id === voice.voice_id;
                  const isAutoSuggested = speaker.auto_suggestion?.voice_id === voice.voice_id;
                  const isClone = voice.category === "clone";

                  return (
                    <div
                      key={voice.voice_id}
                      className={`voice-option-card ${isSelected ? "selected" : ""} ${
                        isClone ? "clone-card" : ""
                      }`}
                      onClick={() => handleSelectVoice(speaker.speaker_id, voice)}
                    >
                      <div className="voice-card-top">
                        <div className="voice-card-identity">
                          <span className="voice-card-title">{voice.label}</span>
                          {voice.gender && (
                            <span className={`voice-gender-pill ${voice.gender}`}>
                              {voice.gender}
                            </span>
                          )}
                        </div>

                        <div className="voice-card-check">
                          {isSelected && <Check size={14} className="check-icon" />}
                        </div>
                      </div>

                      <p className="voice-card-desc">{voice.description}</p>

                      <div className="voice-card-badges">
                        {isClone && (
                          <span className="badge-pill highlight">
                            <ShieldCheck size={11} /> Highest Fidelity
                          </span>
                        )}
                        {isAutoSuggested && !isClone && (
                          <span className="badge-pill suggestion">
                            <Sparkles size={11} /> AI Suggested
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Voice Design Custom Prompt if selected */}
              {currentSel.voice_id === "design_custom" && (
                <div className="voice-custom-design-box">
                  <label className="design-label">
                    <Wand2 size={13} className="text-accent" /> Describe Desired Voice Timbre:
                  </label>
                  <input
                    type="text"
                    className="design-input"
                    placeholder="e.g. energetic male, young adult, Hindi conversational accent"
                    value={currentSel.design_prompt || ""}
                    onChange={(e) =>
                      handleDesignPromptChange(speaker.speaker_id, e.target.value)
                    }
                  />
                  <div className="preset-chips">
                    {DESIGN_PRESETS.map((preset) => (
                      <button
                        key={preset}
                        type="button"
                        className="preset-chip"
                        onClick={() =>
                          handleDesignPromptChange(speaker.speaker_id, preset)
                        }
                      >
                        {preset}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Sticky Bottom Actions */}
      <div className="voice-selection-actions-bar">
        <button
          type="button"
          className="btn secondary"
          disabled={submitting}
          onClick={handleAutoApply}
        >
          <Bot size={15} />
          <span>Use AI Suggested Defaults</span>
        </button>

        <button
          type="button"
          className="btn primary large-btn"
          disabled={submitting}
          onClick={handleApply}
        >
          <Sparkles size={16} />
          <span>{submitting ? "Applying & Starting Synthesis..." : "Apply Voices & Continue Dubbing"}</span>
          <ArrowRight size={16} />
        </button>
      </div>
    </section>
  );
}
