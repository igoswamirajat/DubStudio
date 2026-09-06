import { useEffect, useRef, useState } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Maximize2,
  Film,
  Sparkles,
  AlertCircle,
  Loader2,
  Subtitles,
} from "lucide-react";
import { mediaUrl, type Job } from "../api";

type Props = {
  job: Job;
  currentTime?: number;
  activeSegmentText?: string;
  mediaBust?: number;
  onTimeUpdate?: (t: number) => void;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
};

export function StudioPlayer({
  job,
  activeSegmentText,
  mediaBust = 0,
  onTimeUpdate,
  videoRef: externalVideoRef,
}: Props) {
  const internalRef = useRef<HTMLVideoElement | null>(null);
  const ref = externalVideoRef || internalRef;

  const [track, setTrack] = useState<"output" | "source">(
    job.state === "completed" || job.has_output ? "output" : "source"
  );
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [duration, setDuration] = useState(job.duration_s || 0);
  const [curTime, setCurTime] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [showSubtitles, setShowSubtitles] = useState(true);

  // Default to output once completed
  useEffect(() => {
    if (job.state === "completed" && job.has_output) {
      setTrack("output");
    }
  }, [job.state, job.has_output]);

  const currentUrl = `${mediaUrl(job.job_id, track)}?v=${mediaBust}`;

  useEffect(() => {
    setIsLoading(true);
    setHasError(false);
  }, [currentUrl, track]);

  function togglePlay() {
    const v = ref.current;
    if (!v) return;
    if (v.paused) {
      v.play().catch(() => void 0);
    } else {
      v.pause();
    }
  }

  function handleSeek(e: React.ChangeEvent<HTMLInputElement>) {
    const v = ref.current;
    if (!v) return;
    const t = parseFloat(e.target.value);
    v.currentTime = t;
    setCurTime(t);
  }

  function toggleMute() {
    const v = ref.current;
    if (!v) return;
    v.muted = !v.muted;
    setIsMuted(v.muted);
  }

  function handleVolumeChange(e: React.ChangeEvent<HTMLInputElement>) {
    const v = ref.current;
    if (!v) return;
    const val = parseFloat(e.target.value);
    v.volume = val;
    setVolume(val);
    if (val === 0) {
      v.muted = true;
      setIsMuted(true);
    } else if (v.muted) {
      v.muted = false;
      setIsMuted(false);
    }
  }

  function toggleFullscreen() {
    const v = ref.current?.parentElement;
    if (!v) return;
    if (!document.fullscreenElement) {
      v.requestFullscreen().catch(() => void 0);
    } else {
      document.exitFullscreen().catch(() => void 0);
    }
  }

  function fmtTime(s: number) {
    if (!s || Number.isNaN(s) || !Number.isFinite(s)) return "0:00";
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  }

  const outputReady = job.state === "completed" || job.has_output;

  return (
    <div className="player-wrapper">
      {/* Top track switcher bar */}
      <div className="player-track-tabs">
        <button
          className={`track-tab ${track === "output" ? "active" : ""}`}
          onClick={() => setTrack("output")}
        >
          <Sparkles size={14} className="tab-icon" />
          <span>Dubbed Output</span>
          {outputReady ? (
            <span className="tab-badge ready">Ready</span>
          ) : (
            <span className="tab-badge pending">Rendering…</span>
          )}
        </button>

        <button
          className={`track-tab ${track === "source" ? "active" : ""}`}
          onClick={() => setTrack("source")}
        >
          <Film size={14} className="tab-icon" />
          <span>Original Video</span>
          <span className="tab-badge ready">Source</span>
        </button>

        <div className="player-meta-tag">
          {track === "output"
            ? `Target: ${(job.target_language || "hi").toUpperCase()}`
            : `Original: ${(job.source_language || "en").toUpperCase()}`}
        </div>
      </div>

      {/* Main Video Stage */}
      <div className="video-viewport" onClick={togglePlay}>
        <video
          ref={ref as React.RefObject<HTMLVideoElement>}
          src={currentUrl}
          playsInline
          preload="metadata"
          className="stage-video"
          onTimeUpdate={(e) => {
            const t = (e.target as HTMLVideoElement).currentTime;
            setCurTime(t);
            onTimeUpdate?.(t);
          }}
          onDurationChange={(e) => {
            setDuration((e.target as HTMLVideoElement).duration || job.duration_s || 0);
          }}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onLoadedData={() => {
            setIsLoading(false);
            setHasError(false);
          }}
          onWaiting={() => setIsLoading(true)}
          onPlaying={() => setIsLoading(false)}
          onError={() => {
            setIsLoading(false);
            setHasError(true);
          }}
        />

        {/* Loading overlay */}
        {isLoading && !hasError && (
          <div className="viewport-overlay loading-overlay">
            <Loader2 size={36} className="spin accent-icon" />
            <p>Buffering video preview…</p>
          </div>
        )}

        {/* Error / Not ready state overlay */}
        {hasError && (
          <div
            className="viewport-overlay error-overlay"
            onClick={(e) => e.stopPropagation()}
          >
            {track === "output" && !outputReady ? (
              <div className="state-notice">
                <Sparkles size={32} className="accent-icon" />
                <h4>Dubbed Video Rendering in Progress</h4>
                <p>
                  The pipeline is currently at stage:{" "}
                  <strong>{job.state.replace(/_/g, " ")}</strong> (
                  {job.percent || 0}%).
                </p>
                <button
                  className="btn ghost small"
                  onClick={() => setTrack("source")}
                >
                  <Film size={14} /> Switch to Original Video
                </button>
              </div>
            ) : (
              <div className="state-notice">
                <AlertCircle size={32} className="err-icon" />
                <h4>Preview Stream Unavailable</h4>
                <p className="muted">
                  Could not load the {track} stream for this job.
                </p>
                <button
                  className="btn ghost small"
                  onClick={() => {
                    setHasError(false);
                    setIsLoading(true);
                    if (ref.current) {
                      ref.current.load();
                    }
                  }}
                >
                  <RotateCcw size={14} /> Retry loading
                </button>
              </div>
            )}
          </div>
        )}

        {/* Subtitle overlay */}
        {showSubtitles && activeSegmentText && (
          <div className="subtitle-overlay">
            <span className="subtitle-pill">{activeSegmentText}</span>
          </div>
        )}

        {/* Big play button overlay when paused */}
        {!isPlaying && !isLoading && !hasError && (
          <div className="viewport-overlay play-overlay">
            <div className="play-circle">
              <Play size={28} className="fill-icon" />
            </div>
          </div>
        )}
      </div>

      {/* Bottom custom control dock */}
      <div className="player-control-dock">
        <button
          className="dock-btn"
          onClick={togglePlay}
          title={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
        </button>

        <span className="dock-time tabular">{fmtTime(curTime)}</span>

        <div className="seek-track-wrapper">
          <input
            type="range"
            min={0}
            max={duration || 10}
            step={0.05}
            value={curTime}
            onChange={handleSeek}
            className="seek-slider"
          />
        </div>

        <span className="dock-time muted tabular">{fmtTime(duration)}</span>

        <div className="volume-group">
          <button
            className="dock-btn"
            onClick={toggleMute}
            title={isMuted ? "Unmute" : "Mute"}
          >
            {isMuted || volume === 0 ? <VolumeX size={17} /> : <Volume2 size={17} />}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={isMuted ? 0 : volume}
            onChange={handleVolumeChange}
            className="volume-slider"
          />
        </div>

        <button
          className={`dock-btn ${showSubtitles ? "active" : ""}`}
          onClick={() => setShowSubtitles((s) => !s)}
          title="Toggle Subtitles"
        >
          <Subtitles size={17} />
        </button>

        <button
          className="dock-btn"
          onClick={toggleFullscreen}
          title="Fullscreen"
        >
          <Maximize2 size={17} />
        </button>
      </div>
    </div>
  );
}
