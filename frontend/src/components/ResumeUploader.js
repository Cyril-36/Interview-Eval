import React, { useRef, useState } from "react";

const MAX_FILE_BYTES = 2 * 1024 * 1024;

/**
 * @typedef {Object} ResumeData
 * @property {string[]} skills
 * @property {string} summary
 * @property {string} fileName
 * @property {number} pageCount
 * @property {boolean} truncated
 */

/**
 * @typedef {Object} ResumeUploaderProps
 * @property {ResumeData | null} resumeData
 * @property {(file: File) => Promise<void>} onUpload
 * @property {() => void} onClear
 * @property {(skill: string) => void} onRemoveSkill
 * @property {boolean} disabled
 */

/** @param {ResumeUploaderProps} props */
export default function ResumeUploader({
  resumeData,
  onUpload,
  onClear,
  onRemoveSkill,
  disabled,
}) {
  const inputRef = useRef(null);
  const [error, setError] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  const handlePick = () => {
    if (disabled || isUploading) return;
    inputRef.current?.click();
  };

  const handleChange = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = ""; // allow re-upload of the same file
    if (!file) return;

    if (file.type && file.type !== "application/pdf") {
      setError("Only PDF resumes are supported.");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      setError(`Resume too large (max ${Math.round(MAX_FILE_BYTES / 1024 / 1024)} MB).`);
      return;
    }

    setError("");
    setIsUploading(true);
    try {
      await onUpload(file);
    } catch (err) {
      setError(err?.message || "Failed to parse resume.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleClear = () => {
    setError("");
    onClear();
  };

  return (
    <div className="resume-uploader" data-testid="resume-uploader">
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        onChange={handleChange}
        style={{ display: "none" }}
        data-testid="resume-file-input"
      />

      {!resumeData ? (
        <div className="resume-uploader-empty">
          <button
            type="button"
            className="resume-upload-btn"
            onClick={handlePick}
            disabled={disabled || isUploading}
          >
            {isUploading ? "Parsing resume…" : "📄 Upload resume (optional)"}
          </button>
          <span className="resume-uploader-hint">
            Tailor questions to your background — PDF, max 2 MB
          </span>
        </div>
      ) : (
        <div className="resume-uploader-loaded">
          <div className="resume-uploader-header">
            <span className="resume-file-name" title={resumeData.fileName}>
              📄 {resumeData.fileName}
            </span>
            <button
              type="button"
              className="resume-clear-btn"
              onClick={handleClear}
              disabled={disabled}
              aria-label="Remove resume"
            >
              ✕
            </button>
          </div>
          {resumeData.skills.length > 0 ? (
            <div className="resume-skill-chips" data-testid="resume-skill-chips">
              {resumeData.skills.map((skill) => (
                <button
                  key={skill}
                  type="button"
                  className="resume-skill-chip"
                  onClick={() => onRemoveSkill(skill)}
                  disabled={disabled}
                  title="Click to remove"
                >
                  {skill} <span className="resume-skill-chip-x">×</span>
                </button>
              ))}
            </div>
          ) : (
            <span className="resume-uploader-hint">
              No skills detected — questions will use the resume summary only.
            </span>
          )}
          {resumeData.truncated && (
            <span className="resume-uploader-hint">
              Resume was long; only the first portion will be used.
            </span>
          )}
        </div>
      )}

      {error && <div className="resume-uploader-error" role="alert">{error}</div>}
    </div>
  );
}
