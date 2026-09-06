import { useEffect, useState } from "react";
import {
  HardDrive,
  Download,
  CheckCircle2,
  AlertTriangle,
  Folder,
  RefreshCw,
  X,
  Sparkles,
  Cpu,
  Layers,
  Check,
} from "lucide-react";
import {
  downloadModel,
  getModels,
  updateStorageDir,
  type ModelItem,
  type StorageInfo,
} from "../api";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onModelsChanged?: () => void;
}

export function ModelManagerModal({ isOpen, onClose, onModelsChanged }: Props) {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [storage, setStorage] = useState<StorageInfo | null>(null);
  const [newPath, setNewPath] = useState("");
  const [savingPath, setSavingPath] = useState(false);
  const [loading, setLoading] = useState(true);
  const [toastMsg, setToastMsg] = useState("");
  const [toastErr, setToastErr] = useState("");

  async function loadData() {
    try {
      const res = await getModels();
      setModels(res.models);
      setStorage(res.storage);
      if (!newPath) {
        setNewPath(res.storage.storage_path);
      }
    } catch (e) {
      setToastErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!isOpen) return;
    loadData();

    // Poll every 3s if any download is active
    const interval = setInterval(() => {
      getModels()
        .then((res) => {
          setModels(res.models);
          setStorage(res.storage);
        })
        .catch(() => void 0);
    }, 3000);

    return () => clearInterval(interval);
  }, [isOpen]);

  if (!isOpen) return null;

  async function handleSavePath() {
    if (!newPath.trim()) return;
    setSavingPath(true);
    setToastErr("");
    setToastMsg("");
    try {
      const res = await updateStorageDir(newPath.trim());
      setStorage(res);
      setToastMsg(`Model directory updated to ${res.storage_path}`);
      setTimeout(() => setToastMsg(""), 3500);
      loadData();
      onModelsChanged?.();
    } catch (e) {
      setToastErr(String(e));
    } finally {
      setSavingPath(false);
    }
  }

  async function handleDownload(modelId: string) {
    setToastErr("");
    try {
      await downloadModel(modelId);
      setToastMsg(`Download started in background. Monitor progress here.`);
      setTimeout(() => setToastMsg(""), 3500);
      loadData();
    } catch (e) {
      setToastErr(String(e));
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-content model-manager-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div className="modal-title-group">
            <div className="modal-icon-badge">
              <HardDrive size={20} />
            </div>
            <div>
              <h3>AI Models &amp; Storage Manager</h3>
              <p>Zero-CLI weight management, disk drive routing, and offline AI readiness</p>
            </div>
          </div>
          <button className="icon-btn close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        {toastMsg && (
          <div className="modal-toast success">
            <Check size={14} /> {toastMsg}
          </div>
        )}
        {toastErr && (
          <div className="modal-toast error">
            <AlertTriangle size={14} /> {toastErr}
          </div>
        )}

        {/* Section 1: Storage & Drive Card */}
        {storage && (
          <div className="storage-hero-card">
            <div className="storage-info-row">
              <div className="drive-badge">
                <Folder size={18} />
                <span>Drive {storage.drive}</span>
              </div>
              <div className="drive-stats">
                <span className="free-stat">
                  <strong>{storage.free_gb} GB</strong> Free
                </span>
                <span className="total-stat">of {storage.total_gb} GB Total</span>
              </div>
            </div>

            <div className="disk-meter-track">
              <div
                className={`disk-meter-fill ${storage.is_low_space ? "warning" : ""}`}
                style={{
                  width: `${Math.min(100, Math.max(5, 100 - storage.free_percent))}%`,
                }}
              />
            </div>

            <div className="storage-path-editor">
              <label>Model Storage Directory</label>
              <div className="path-input-group">
                <input
                  type="text"
                  value={newPath}
                  onChange={(e) => setNewPath(e.target.value)}
                  placeholder="e.g. D:\hf_cache"
                />
                <button
                  className="btn secondary"
                  disabled={savingPath || newPath === storage.storage_path}
                  onClick={handleSavePath}
                >
                  {savingPath ? "Saving..." : "Update Directory"}
                </button>
              </div>
              {storage.is_low_space && (
                <p className="low-space-warning">
                  <AlertTriangle size={14} /> Warning: Free disk space is below 5 GB on this drive.
                  Select a drive with more free space (e.g. D:\hf_cache) before downloading large models.
                </p>
              )}
            </div>
          </div>
        )}

        {/* Section 2: Model Registry Grid */}
        <div className="model-cards-container">
          <h4>Installed &amp; Available Models</h4>
          {loading ? (
            <div className="models-loading">
              <RefreshCw className="spin" size={20} />
              <span>Scanning models…</span>
            </div>
          ) : (
            <div className="models-grid">
              {models.map((m) => {
                const isDownloaded = m.status === "downloaded";
                const isDownloading = m.status === "downloading";

                return (
                  <div key={m.id} className={`model-card ${isDownloaded ? "installed" : ""}`}>
                    <div className="model-card-header">
                      <div className="model-badge-row">
                        <span className="category-pill">{m.category.toUpperCase()}</span>
                        {m.recommended_for === "hi" && (
                          <span className="recommended-pill">
                            <Sparkles size={11} /> Best for Hindi
                          </span>
                        )}
                        <span className="size-pill">~{Math.round(m.size_mb / 1024 * 10) / 10} GB</span>
                      </div>
                      <h4 className="model-name">{m.name}</h4>
                      <p className="model-repo">{m.repo_id}</p>
                    </div>

                    <p className="model-desc">{m.description}</p>

                    <div className="model-card-footer">
                      {isDownloading ? (
                        <div className="download-progress-wrap">
                          <div className="progress-info">
                            <span>{m.progress_message || "Downloading..."}</span>
                            <span>{m.progress}%</span>
                          </div>
                          <div className="progress-bar-small">
                            <div
                              className="progress-fill-small"
                              style={{ width: `${m.progress}%` }}
                            />
                          </div>
                        </div>
                      ) : isDownloaded ? (
                        <div className="status-downloaded">
                          <CheckCircle2 size={16} className="text-success" />
                          <span>
                            Ready {m.size_on_disk_mb ? `(${Math.round(m.size_on_disk_mb / 1024 * 10) / 10} GB)` : ""}
                          </span>
                        </div>
                      ) : (
                        <button
                          className="btn primary download-btn"
                          onClick={() => handleDownload(m.id)}
                        >
                          <Download size={14} />
                          <span>Download Model</span>
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
