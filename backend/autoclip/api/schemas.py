"""Pydantic request and response models for the REST API.

Kept separate from the dataclasses in ``db.models``: those mirror SQLite rows,
these are the wire contract. Splitting them means a schema change doesn't
silently reshape the API, and the API can expose derived fields (durations,
export URLs) that don't belong in storage.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..db import models


class SourceOut(BaseModel):
    id: str
    type: str
    title: str
    url: str | None = None
    filename: str | None = None
    channel: str | None = None
    duration_s: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    has_audio: bool
    has_video: bool
    created_at: str
    timings: dict[str, Any] | None = None

    @classmethod
    def of(cls, source: models.Source) -> SourceOut:
        # Built field by field rather than from vars(): the row carries an
        # absolute filesystem path that has no business leaving the process.
        return cls(
            id=source.id,
            type=source.type,
            title=source.title,
            url=source.url,
            filename=source.filename,
            channel=source.channel,
            duration_s=source.duration_s,
            width=source.width,
            height=source.height,
            fps=source.fps,
            has_audio=source.has_audio,
            has_video=source.has_video,
            created_at=source.created_at,
            timings=source.timings or None,
        )


class YouTubeIngestIn(BaseModel):
    url: str
    cookies_from_browser: str | None = None


class JobSettingsIn(BaseModel):
    """Per-job overrides. Anything omitted falls back to saved settings."""

    provider: str | None = None
    whisper_model: str | None = None
    language: str | None = None
    diarization: bool | None = None
    min_duration_s: float | None = Field(default=None, gt=0)
    max_duration_s: float | None = Field(default=None, gt=0)
    max_clips: int | None = Field(default=None, ge=1, le=50)
    caption_style: str | None = None
    ratio: Literal["9:16", "1:1", "16:9"] | None = None


class JobCreateIn(BaseModel):
    source_id: str
    settings: JobSettingsIn = Field(default_factory=JobSettingsIn)


class StageTiming(BaseModel):
    duration_seconds: float
    status: Literal["completed", "failed", "cancelled", "cached", "skipped"]
    details: dict[str, Any] | None = None


class JobTimings(BaseModel):
    pipeline_duration_seconds: float | None = None
    total_duration_seconds: float | None = None
    stages: dict[str, StageTiming] = Field(default_factory=dict)


class JobOut(BaseModel):
    id: str
    source_id: str
    status: str
    current_stage: str
    progress: float
    error: str | None = None
    provider: str
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    source: SourceOut | None = None
    timings: JobTimings | None = None

    @classmethod
    def of(cls, job: models.Job, source: models.Source | None = None) -> JobOut:
        timings_obj: JobTimings | None = None
        if job.timings and isinstance(job.timings, dict):
            try:
                timings_obj = JobTimings.model_validate(job.timings)
            except ValidationError:
                stages_dict: dict[str, StageTiming] = {}
                raw_stages = job.timings.get("stages")
                stages_map = raw_stages if isinstance(raw_stages, dict) else {}
                for k, v in stages_map.items():
                    if isinstance(v, dict) and "duration_seconds" in v and "status" in v:
                        try:
                            stages_dict[k] = StageTiming(**v)
                        except ValidationError:
                            pass

                pipeline_dur = job.timings.get("pipeline_duration_seconds")
                try:
                    pipeline_dur = float(pipeline_dur) if pipeline_dur is not None else None
                except (ValueError, TypeError):
                    pipeline_dur = None

                total_dur = job.timings.get("total_duration_seconds")
                try:
                    total_dur = float(total_dur) if total_dur is not None else None
                except (ValueError, TypeError):
                    total_dur = None

                timings_obj = JobTimings(
                    pipeline_duration_seconds=pipeline_dur,
                    total_duration_seconds=total_dur,
                    stages=stages_dict,
                )

        return cls(
            id=job.id,
            source_id=job.source_id,
            status=job.status,
            current_stage=job.current_stage,
            progress=job.progress,
            error=job.error,
            provider=job.provider,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            source=SourceOut.of(source) if source else None,
            timings=timings_obj,
        )


class WordOut(BaseModel):
    text: str
    start: float
    end: float
    speaker: str | None = None


class ExportOut(BaseModel):
    id: str
    clip_id: str
    ratio: str
    style: str
    size_bytes: int
    created_at: str
    download_url: str

    @classmethod
    def of(cls, export: models.Export) -> ExportOut:
        return cls(
            id=export.id,
            clip_id=export.clip_id,
            ratio=export.ratio,
            style=export.style,
            size_bytes=export.size_bytes,
            created_at=export.created_at,
            download_url=f"/api/exports/{export.id}/download",
        )


class ClipOut(BaseModel):
    id: str
    job_id: str
    rank: int
    start_s: float
    end_s: float
    duration_s: float
    start_word: int
    end_word: int
    title: str
    hook: str
    score: int
    reason: str
    status: str
    user_trimmed: bool
    caption_style: str = "bold_pop"
    ratio: str = "9:16"
    exports: list[ExportOut] = Field(default_factory=list)

    @classmethod
    def of(
        cls,
        clip: models.Clip,
        *,
        edit: models.ClipEdit | None = None,
        exports: list[models.Export] | None = None,
    ) -> ClipOut:
        return cls(
            id=clip.id,
            job_id=clip.job_id,
            rank=clip.rank,
            start_s=clip.start_s,
            end_s=clip.end_s,
            duration_s=clip.duration_s,
            start_word=clip.start_word,
            end_word=clip.end_word,
            title=clip.title,
            hook=clip.hook,
            score=clip.score,
            reason=clip.reason,
            status=clip.status,
            user_trimmed=clip.user_trimmed,
            caption_style=edit.caption_style if edit else "bold_pop",
            ratio=edit.ratio if edit else "9:16",
            exports=[ExportOut.of(e) for e in (exports or [])],
        )


class ClipPatchIn(BaseModel):
    start_s: float | None = Field(default=None, ge=0)
    end_s: float | None = Field(default=None, gt=0)
    title: str | None = None
    status: Literal["candidate", "kept", "discarded", "exported"] | None = None


class CaptionPatchIn(BaseModel):
    """Word-level caption edits for one clip."""

    words: list[WordOut] | None = None
    caption_style: str | None = None
    ratio: Literal["9:16", "1:1", "16:9"] | None = None


class ExportRequestIn(BaseModel):
    ratio: Literal["9:16", "1:1", "16:9"] = "9:16"
    style: str = "bold_pop"
    write_srt: bool = False


class CaptionStyleOut(BaseModel):
    key: str
    label: str
    description: str
    #: Enough for the UI to approximate the look in HTML/CSS before rendering.
    preview: dict[str, Any]


class ProviderStatusOut(BaseModel):
    name: str
    available: bool
    detail: str = ""
    models: list[str] = Field(default_factory=list)
    requires_key: bool = True
    has_key: bool = False


class SettingsOut(BaseModel):
    active_provider: str
    providers: dict[str, dict[str, Any]]
    whisper: dict[str, Any]
    clips: dict[str, Any]
    ingest: dict[str, Any]
    export: dict[str, Any]
    insecure_secret_storage: bool
    #: Which providers have a key stored. The keys themselves never leave the
    #: keyring, so the UI shows presence, not value.
    keys_present: dict[str, bool] = Field(default_factory=dict)


class SettingsIn(BaseModel):
    active_provider: str | None = None
    providers: dict[str, dict[str, Any]] | None = None
    whisper: dict[str, Any] | None = None
    clips: dict[str, Any] | None = None
    ingest: dict[str, Any] | None = None
    export: dict[str, Any] | None = None


class SecretIn(BaseModel):
    key: str
    value: str


class SystemOut(BaseModel):
    ready: bool
    python_version: str
    platform: str
    ffmpeg_version: str | None
    has_libass: bool
    nvenc_works: bool
    accel: str
    gpu_name: str | None
    compute_type: str
    diarization_available: bool
