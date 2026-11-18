"""Focused unit tests for stage-level execution timing."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autoclip.config import Settings
from autoclip.db import connection, schema, store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline import Stage
from autoclip.pipeline.reframe.croppath import CropPath
from autoclip.pipeline.runner import JobCancelled, PipelineRunner
from autoclip.pipeline.transcript import Segment, Transcript, Word


@pytest.fixture
def mock_source(initialised_db: int) -> Source:
    return store.create_source(
        Source(
            id=new_id(),
            type="youtube",
            path="C:/media/video.mp4",
            title="Timing Test Video",
            duration_s=60.0,
            has_video=True,
            has_audio=True,
        )
    )


@pytest.fixture
def mock_audio_source(initialised_db: int) -> Source:
    return store.create_source(
        Source(
            id=new_id(),
            type="upload",
            path="C:/media/audio.mp3",
            title="Audio Only Source",
            duration_s=45.0,
            has_video=False,
            has_audio=True,
        )
    )


class TestDatabaseMigrationV2:
    def test_migration_from_v1_to_v2_preserves_old_records(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(schema._V1)
        conn.execute("PRAGMA user_version = 1")

        # Insert raw rows into v1 tables without timings_json
        conn.execute(
            """
            INSERT INTO sources (id, type, path, title, created_at)
            VALUES ('s_old', 'youtube', 'C:/test.mp4', 'Old Source', '2026-01-01T00:00:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO jobs (id, source_id, status, created_at, updated_at)
            VALUES ('j_old', 's_old', 'done', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
            """
        )
        conn.commit()

        # Migrate to v2
        version = schema.migrate(conn)
        assert version == 2
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2

        # Verify old records load and default to empty timings
        s_row = conn.execute("SELECT * FROM sources WHERE id = 's_old'").fetchone()
        j_row = conn.execute("SELECT * FROM jobs WHERE id = 'j_old'").fetchone()

        loaded_source = Source.from_row(s_row)
        loaded_job = Job.from_row(j_row)

        assert loaded_source.timings == {}
        assert loaded_job.timings == {}

    def test_store_timings_updates_safely(self, mock_source: Source) -> None:
        job = store.create_job(Job(id=new_id(), source_id=mock_source.id))
        assert job.timings == {}

        # Record stage 1
        store.record_job_timing(job.id, "prepare", {"duration_seconds": 1.25, "status": "completed"})
        # Record stage 2
        store.record_job_timing(job.id, "transcribe", {"duration_seconds": 5.4, "status": "completed"})

        loaded = store.get_job(job.id)
        assert loaded is not None
        assert "prepare" in loaded.timings["stages"]
        assert "transcribe" in loaded.timings["stages"]
        assert loaded.timings["stages"]["prepare"]["duration_seconds"] == 1.25

        # Unrelated update_job does not overwrite timings
        store.update_job(job.id, progress=0.8, status="running")
        loaded_after_update = store.get_job(job.id)
        assert loaded_after_update is not None
        assert loaded_after_update.progress == pytest.approx(0.8)
        assert "prepare" in loaded_after_update.timings["stages"]
        assert "transcribe" in loaded_after_update.timings["stages"]

        # Set job durations
        store.set_job_durations(job.id, pipeline_duration_seconds=10.5, total_duration_seconds=15.0)
        final = store.get_job(job.id)
        assert final is not None
        assert final.timings["pipeline_duration_seconds"] == 10.5
        assert final.timings["total_duration_seconds"] == 15.0
        assert "prepare" in final.timings["stages"]

    @pytest.mark.parametrize(
        "malformed_json",
        [
            "[1, 2, 3]",
            '"just a string"',
            "42.5",
            "true",
            "null",
            "{not valid json",
        ],
    )
    def test_from_row_normalizes_non_dict_json_to_empty_dict(self, malformed_json: str) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(schema._V1)
        schema._migration_v2(conn)

        conn.execute(
            """
            INSERT INTO sources (id, type, path, title, created_at, timings_json)
            VALUES ('s_mal', 'youtube', 'C:/test.mp4', 'Test', '2026-01-01T00:00:00Z', ?)
            """,
            (malformed_json,),
        )
        conn.execute(
            """
            INSERT INTO jobs (id, source_id, status, created_at, updated_at, timings_json)
            VALUES ('j_mal', 's_mal', 'done', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', ?)
            """,
            (malformed_json,),
        )
        conn.commit()

        s_row = conn.execute("SELECT * FROM sources WHERE id = 's_mal'").fetchone()
        j_row = conn.execute("SELECT * FROM jobs WHERE id = 'j_mal'").fetchone()

        assert Source.from_row(s_row).timings == {}
        assert Job.from_row(j_row).timings == {}

    def test_store_timing_methods_handle_malformed_structures_safely(
        self, mock_source: Source
    ) -> None:
        job = store.create_job(Job(id=new_id(), source_id=mock_source.id))

        # 1. When timings_json is a top-level list
        with connection() as conn:
            conn.execute(
                "UPDATE jobs SET timings_json = ? WHERE id = ?",
                (json.dumps(["item1", "item2"]), job.id),
            )

        updated = store.record_job_timing(job.id, "prepare", {"duration_seconds": 1.2, "status": "completed"})
        assert isinstance(updated, dict)
        assert isinstance(updated.get("stages"), dict)
        assert updated["stages"]["prepare"]["status"] == "completed"

        # 2. When timings_json has a non-dictionary stages field
        with connection() as conn:
            conn.execute(
                "UPDATE jobs SET timings_json = ? WHERE id = ?",
                (json.dumps({"stages": ["not", "a", "dict"], "other_key": 123}), job.id),
            )

        updated2 = store.record_job_timing(job.id, "transcribe", {"duration_seconds": 3.4, "status": "completed"})
        assert isinstance(updated2, dict)
        assert isinstance(updated2.get("stages"), dict)
        assert updated2["stages"]["transcribe"]["status"] == "completed"
        # Other valid keys at the top level are preserved
        assert updated2.get("other_key") == 123

        # 3. set_job_durations when timings_json is a top-level string
        with connection() as conn:
            conn.execute(
                "UPDATE jobs SET timings_json = ? WHERE id = ?",
                (json.dumps("invalid string"), job.id),
            )

        dur_result = store.set_job_durations(job.id, pipeline_duration_seconds=5.0, total_duration_seconds=7.5)
        assert isinstance(dur_result, dict)
        assert dur_result["pipeline_duration_seconds"] == 5.0
        assert dur_result["total_duration_seconds"] == 7.5


class TestPipelineTimingExecution:
    @pytest.mark.asyncio
    async def test_successful_stage_timings(self, mock_source: Source, monkeypatch: pytest.MonkeyPatch) -> None:
        job = store.create_job(Job(id=new_id(), source_id=mock_source.id))
        runner = PipelineRunner(job, mock_source, settings=Settings())

        # Mock all underlying stages with small delays
        def fake_extract_audio(*args, **kwargs):
            time.sleep(0.01)

        def fake_transcribe(*args, **kwargs):
            time.sleep(0.01)
            t = Transcript(language="en", model="small")
            t.words.append(Word(text="hello", start=0.0, end=1.0))
            t.segments.append(Segment(text="hello", start=0.0, end=1.0, first_word=0, last_word=0))
            return t

        def fake_detect_silences(*args, **kwargs):
            time.sleep(0.005)
            return []

        async def fake_detect(*args, **kwargs):
            time.sleep(0.01)
            return [Clip(id=new_id(), job_id=job.id, start_s=0.0, end_s=5.0, start_word=0, end_word=0)]

        def fake_build_crop_path(*args, **kwargs):
            time.sleep(0.005)
            return CropPath(source_width=1920, source_height=1080, segments=[])

        def fake_export_clip(*args, **kwargs):
            time.sleep(0.01)
            dest = kwargs["request"].destination if "request" in kwargs else args[0].destination
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"dummy mp4")

        from autoclip.pipeline import captions, export, highlights, prepare, transcribe
        from autoclip.pipeline import runner as runner_module

        monkeypatch.setattr(prepare, "extract_audio", fake_extract_audio)
        monkeypatch.setattr(prepare, "generate_thumbnails", lambda *a, **k: None)
        monkeypatch.setattr(transcribe, "transcribe", fake_transcribe)
        monkeypatch.setattr(prepare, "detect_silences", fake_detect_silences)
        monkeypatch.setattr(highlights, "detect", fake_detect)
        monkeypatch.setattr(runner_module, "build_provider", lambda *a, **k: MagicMock(name="mock"))
        monkeypatch.setattr(runner_module, "build_crop_path", fake_build_crop_path)
        monkeypatch.setattr(export, "export_clip", fake_export_clip)

        clips = await runner.run()
        assert len(clips) == 1

        loaded = store.get_job(job.id)
        assert loaded is not None
        assert loaded.status == "done"
        timings = loaded.timings
        assert timings["pipeline_duration_seconds"] > 0
        assert timings["total_duration_seconds"] >= timings["pipeline_duration_seconds"]

        stages = timings["stages"]
        for expected in ("prepare", "transcribe", "highlights", "reframe", "captions", "export"):
            assert expected in stages
            assert stages[expected]["duration_seconds"] >= 0
            assert stages[expected]["status"] == "completed"

        # Check highlight details for silence accounting
        assert "silence_detection_seconds" in stages["highlights"]["details"]
        assert "detection_seconds" in stages["highlights"]["details"]
        assert stages["highlights"]["details"]["clip_count"] == 1

    @pytest.mark.asyncio
    async def test_failed_stage_timing(self, mock_source: Source, monkeypatch: pytest.MonkeyPatch) -> None:
        job = store.create_job(Job(id=new_id(), source_id=mock_source.id))
        runner = PipelineRunner(job, mock_source, settings=Settings())

        def explode(*args, **kwargs):
            time.sleep(0.01)
            raise RuntimeError("Model download crashed")

        from autoclip.pipeline import prepare, transcribe

        monkeypatch.setattr(prepare, "extract_audio", lambda *a, **k: None)
        monkeypatch.setattr(prepare, "generate_thumbnails", lambda *a, **k: None)
        monkeypatch.setattr(transcribe, "transcribe", explode)

        with pytest.raises(RuntimeError, match="Model download crashed"):
            await runner.run()

        loaded = store.get_job(job.id)
        assert loaded is not None
        assert loaded.status == "failed"
        assert loaded.timings["pipeline_duration_seconds"] > 0
        assert "prepare" in loaded.timings["stages"]
        assert loaded.timings["stages"]["prepare"]["status"] == "completed"

        # Failed stage must retain its elapsed time and status
        assert "transcribe" in loaded.timings["stages"]
        assert loaded.timings["stages"]["transcribe"]["status"] == "failed"
        assert loaded.timings["stages"]["transcribe"]["duration_seconds"] > 0

    @pytest.mark.asyncio
    async def test_cancelled_stage_timing(self, mock_source: Source, monkeypatch: pytest.MonkeyPatch) -> None:
        job = store.create_job(Job(id=new_id(), source_id=mock_source.id))
        cancelled_flag = False
        runner = PipelineRunner(job, mock_source, settings=Settings(), is_cancelled=lambda: cancelled_flag)

        def cancellable_transcribe(*args, **kwargs):
            nonlocal cancelled_flag
            time.sleep(0.01)
            cancelled_flag = True
            # Runner checks cancellation at various points; simulate check
            if kwargs.get("cancelled") and kwargs["cancelled"]():
                raise JobCancelled("Cancelled by user")
            raise JobCancelled("Cancelled by user")

        from autoclip.pipeline import prepare, transcribe

        monkeypatch.setattr(prepare, "extract_audio", lambda *a, **k: None)
        monkeypatch.setattr(prepare, "generate_thumbnails", lambda *a, **k: None)
        monkeypatch.setattr(transcribe, "transcribe", cancellable_transcribe)

        with pytest.raises(JobCancelled):
            await runner.run()

        loaded = store.get_job(job.id)
        assert loaded is not None
        assert loaded.status == "cancelled"
        assert loaded.timings["pipeline_duration_seconds"] > 0
        assert "transcribe" in loaded.timings["stages"]
        assert loaded.timings["stages"]["transcribe"]["status"] == "cancelled"
        assert loaded.timings["stages"]["transcribe"]["duration_seconds"] > 0

    @pytest.mark.asyncio
    async def test_cached_stages_reporting(self, mock_source: Source, tmp_path: Path) -> None:
        job = store.create_job(Job(id=new_id(), source_id=mock_source.id))
        runner = PipelineRunner(job, mock_source, settings=Settings())

        # Pre-seed cached files
        runner.workspace.audio.write_bytes(b"existing audio")
        runner.workspace.thumbnails.mkdir(parents=True, exist_ok=True)
        (runner.workspace.thumbnails / "0001.jpg").write_bytes(b"jpg")

        transcript_data = {
            "language": "en",
            "model": "small",
            "source": "whisper",
            "words": [{"text": "word", "start": 0.0, "end": 1.0}],
            "segments": [{"text": "word", "start": 0.0, "end": 1.0, "first_word": 0, "last_word": 0}],
        }
        runner.workspace.transcript.write_text(json.dumps(transcript_data), encoding="utf-8")

        # Pre-seed clips
        clip = Clip(
            id=new_id(),
            job_id=job.id,
            rank=1,
            start_s=0.0,
            end_s=5.0,
            start_word=0,
            end_word=0,
            title="Cached Clip",
        )
        store.replace_clips(job.id, [clip])
        crop_dir = runner.workspace.root / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        (crop_dir / f"{clip.id}.json").write_text(
            json.dumps({"source_width": 1920, "source_height": 1080, "segments": []}), encoding="utf-8"
        )

        audio = runner._stage_prepare()
        assert runner.job.timings["stages"]["prepare"]["status"] == "cached"

        tr = runner._stage_transcribe(audio)
        assert runner.job.timings["stages"]["transcribe"]["status"] == "cached"

        clips = await runner._stage_highlights(tr, [], silence_seconds=0.25, silence_cached=True)
        hl_timing = runner.job.timings["stages"]["highlights"]
        assert hl_timing["status"] == "cached"
        assert hl_timing["duration_seconds"] >= 0.25
        assert hl_timing["details"]["silence_detection_seconds"] == 0.25
        assert hl_timing["details"]["silence_status"] == "cached"

        crops = runner._stage_reframe(clips, tr)
        assert runner.job.timings["stages"]["reframe"]["status"] == "cached"

    @pytest.mark.asyncio
    async def test_audio_only_skips_reframe(self, mock_audio_source: Source) -> None:
        job = store.create_job(Job(id=new_id(), source_id=mock_audio_source.id))
        runner = PipelineRunner(job, mock_audio_source, settings=Settings())

        clips = [Clip(id=new_id(), job_id=job.id, start_s=0.0, end_s=5.0, start_word=0, end_word=0)]
        transcript = Transcript()

        crops = runner._stage_reframe(clips, transcript)
        assert crops == {}

        loaded = store.get_job(job.id)
        assert loaded is not None
        reframe_timing = loaded.timings["stages"]["reframe"]
        assert reframe_timing["status"] == "skipped"
        assert reframe_timing["details"]["reason"] == "audio_only"

    @pytest.mark.asyncio
    async def test_multiple_clip_exports_aggregate_timing(
        self, mock_source: Source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = store.create_job(Job(id=new_id(), source_id=mock_source.id))
        runner = PipelineRunner(job, mock_source, settings=Settings())

        clips = [
            Clip(id=new_id(), job_id=job.id, start_s=0.0, end_s=5.0, start_word=0, end_word=0, rank=1),
            Clip(id=new_id(), job_id=job.id, start_s=5.0, end_s=10.0, start_word=1, end_word=1, rank=2),
            Clip(id=new_id(), job_id=job.id, start_s=10.0, end_s=15.0, start_word=2, end_word=2, rank=3),
        ]
        transcript = Transcript()
        for i in range(3):
            transcript.words.append(Word(text=f"w{i}", start=float(i), end=float(i + 1)))

        export_count = 0

        def fake_export_clip(*args, **kwargs):
            nonlocal export_count
            export_count += 1
            time.sleep(0.005)
            dest = kwargs["request"].destination if "request" in kwargs else args[0].destination
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"dummy mp4")

        from autoclip.pipeline import export

        monkeypatch.setattr(export, "export_clip", fake_export_clip)

        store.replace_clips(job.id, clips)
        fake_crops = {c.id: CropPath(source_width=1920, source_height=1080, segments=[]) for c in clips}
        runner._stage_export(clips, transcript, fake_crops)
        assert export_count == 3

        loaded = store.get_job(job.id)
        assert loaded is not None
        export_timing = loaded.timings["stages"]["export"]
        assert export_timing["status"] == "completed"
        assert export_timing["duration_seconds"] > 0
        assert export_timing["details"]["clip_count"] == 3

    @pytest.mark.asyncio
    async def test_cached_highlights_includes_silence_detection_time(
        self, mock_source: Source
    ) -> None:
        job = store.create_job(Job(id=new_id(), source_id=mock_source.id))
        runner = PipelineRunner(job, mock_source, settings=Settings())

        clip = Clip(id=new_id(), job_id=job.id, rank=1, start_s=0.0, end_s=5.0, start_word=0, end_word=0)
        store.replace_clips(job.id, [clip])

        transcript = Transcript()
        silence_time = 0.35
        clips = await runner._stage_highlights(
            transcript, [], silence_seconds=silence_time, silence_cached=False
        )
        assert len(clips) == 1

        loaded = store.get_job(job.id)
        assert loaded is not None
        timing = loaded.timings["stages"]["highlights"]
        assert timing["status"] == "cached"
        assert timing["duration_seconds"] >= silence_time
        assert timing["details"]["silence_detection_seconds"] == 0.35
        assert timing["details"]["silence_status"] == "completed"

    @pytest.mark.asyncio
    async def test_silence_detection_not_double_counted(
        self, mock_source: Source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = store.create_job(Job(id=new_id(), source_id=mock_source.id))
        runner = PipelineRunner(job, mock_source, settings=Settings())

        from autoclip.pipeline import highlights
        from autoclip.pipeline import runner as runner_module

        monkeypatch.setattr(runner_module, "build_provider", lambda *a, **k: MagicMock(name="mock"))

        async def fake_detect(*args, **kwargs):
            time.sleep(0.02)
            return [Clip(id=new_id(), job_id=job.id, start_s=0.0, end_s=5.0, start_word=0, end_word=0)]

        monkeypatch.setattr(highlights, "detect", fake_detect)

        transcript = Transcript()
        silence_time = 0.05
        clips = await runner._stage_highlights(
            transcript, [], silence_seconds=silence_time, silence_cached=False
        )

        loaded = store.get_job(job.id)
        assert loaded is not None
        timing = loaded.timings["stages"]["highlights"]
        assert timing["status"] == "completed"
        # Total duration must equal detect time + silence_time, not double-counted
        detect_duration = timing["details"]["detection_seconds"]
        silence_duration = timing["details"]["silence_detection_seconds"]
        assert silence_duration == silence_time
        assert detect_duration >= 0.015
        assert timing["duration_seconds"] >= silence_duration + detect_duration
        # Double counting silence detection would add silence_duration twice
        assert timing["duration_seconds"] < (silence_duration * 2) + detect_duration

    @pytest.mark.asyncio
    async def test_stage_timing_persistence_error_does_not_mask_pipeline_exception(
        self, mock_source: Source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = store.create_job(Job(id=new_id(), source_id=mock_source.id))
        runner = PipelineRunner(job, mock_source, settings=Settings())

        from autoclip.pipeline import prepare

        def boom(*args, **kwargs):
            raise ValueError("Audio extraction failed")

        monkeypatch.setattr(prepare, "extract_audio", boom)

        # Make store.record_job_timing fail when recording failure
        def fail_store(*args, **kwargs):
            raise sqlite3.OperationalError("Database disk image is malformed")

        monkeypatch.setattr(store, "record_job_timing", fail_store)

        # The original exception (ValueError) must not be masked by the persistence error
        with pytest.raises(ValueError, match="Audio extraction failed"):
            await runner.run()


class TestIngestAssociationAndAPI:
    def test_ingest_timing_associated_with_job(self, initialised_db: int) -> None:
        source = store.create_source(
            Source(
                id=new_id(),
                type="youtube",
                path="C:/media/video.mp4",
                title="Ingested video",
                duration_s=120.0,
                timings={
                    "ingest": {
                        "duration_seconds": 14.52,
                        "status": "completed",
                        "details": {"type": "youtube"},
                    }
                },
            )
        )

        from autoclip.api.jobs import create_job
        from autoclip.api.schemas import JobCreateIn

        import asyncio

        job_out = asyncio.run(create_job(JobCreateIn(source_id=source.id)))
        assert job_out.timings is not None
        assert "ingest" in job_out.timings.stages
        assert job_out.timings.stages["ingest"].duration_seconds == 14.52
        assert job_out.timings.stages["ingest"].status == "completed"

    def test_job_out_serialization_for_old_jobs(self) -> None:
        source = Source(id=new_id(), type="upload", path="C:/test.mp4", title="Old")
        job = Job(id=new_id(), source_id=source.id, timings={})

        from autoclip.api.schemas import JobOut

        out = JobOut.of(job, source)
        assert out.timings is None
        data = out.model_dump(mode="json")
        assert data["timings"] is None

    def test_job_out_serialization_with_malformed_timings(self) -> None:
        from autoclip.api.schemas import JobOut

        source = Source(id=new_id(), type="upload", path="C:/test.mp4", title="Test")

        # 1. Non-dict timings at top level (e.g. list, string, int)
        for bad_top in (["a", "b"], "invalid-string", 42):
            job = Job(id=new_id(), source_id=source.id, timings=bad_top)  # type: ignore[arg-type]
            out = JobOut.of(job, source)
            assert out.timings is None

        # 2. Non-dict stages value in timings dict
        job_bad_stages = Job(
            id=new_id(),
            source_id=source.id,
            timings={"stages": ["not", "a", "dict"], "pipeline_duration_seconds": 12.3},
        )
        out = JobOut.of(job_bad_stages, source)
        assert out.timings is not None
        assert out.timings.pipeline_duration_seconds == 12.3
        assert out.timings.stages == {}

        # 3. Stages dict with invalid individual stage records (filters them out while keeping valid ones)
        job_mixed_stages = Job(
            id=new_id(),
            source_id=source.id,
            timings={
                "pipeline_duration_seconds": 20.0,
                "stages": {
                    "prepare": {"duration_seconds": 1.5, "status": "completed"},
                    "bad_record_1": "not-a-dict",
                    "bad_record_2": {"duration_seconds": "invalid_number", "status": "completed"},
                    "bad_record_3": {"missing_status": True},
                },
            },
        )
        out_mixed = JobOut.of(job_mixed_stages, source)
        assert out_mixed.timings is not None
        assert "prepare" in out_mixed.timings.stages
        assert out_mixed.timings.stages["prepare"].duration_seconds == 1.5
        assert len(out_mixed.timings.stages) == 1

    def test_job_out_serialization_with_valid_timings(self) -> None:
        from autoclip.api.schemas import JobOut

        source = Source(id=new_id(), type="upload", path="C:/test.mp4", title="Valid")
        job = Job(
            id=new_id(),
            source_id=source.id,
            timings={
                "pipeline_duration_seconds": 45.2,
                "total_duration_seconds": 58.6,
                "stages": {
                    "prepare": {"duration_seconds": 1.2, "status": "completed"},
                    "transcribe": {"duration_seconds": 25.4, "status": "completed"},
                    "highlights": {
                        "duration_seconds": 8.0,
                        "status": "completed",
                        "details": {"clip_count": 2},
                    },
                },
            },
        )
        out = JobOut.of(job, source)
        assert out.timings is not None
        assert out.timings.pipeline_duration_seconds == 45.2
        assert out.timings.total_duration_seconds == 58.6
        assert len(out.timings.stages) == 3
        assert out.timings.stages["highlights"].details == {"clip_count": 2}
