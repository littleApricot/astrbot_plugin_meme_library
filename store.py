"""Persist short image metadata and a shared collection queue."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import time
from pathlib import Path

NAME_LIMIT = 24
DESCRIPTION_LIMIT = 120


def validate_metadata(name: str, description: str) -> tuple[str, str]:
    """Validate concise editable metadata without silently discarding corrections.

    Args:
        name: Short image name, at most 24 characters.
        description: Practical description, at most 120 characters.

    Returns:
        Trimmed name and single-line description.

    Raises:
        ValueError: Either field is missing, invalid or too long.
    """
    if not isinstance(name, str) or not isinstance(description, str):
        raise ValueError("invalid_metadata")
    name, description = " ".join(name.split()), " ".join(description.split())
    if not name or not description:
        raise ValueError("name_and_description_required")
    if len(name) > NAME_LIMIT or len(description) > DESCRIPTION_LIMIT:
        raise ValueError("name_max_24_description_max_120")
    return name, description


class Library:
    """Own image names, descriptions, revisions and notification cursors."""

    def __init__(self, root: Path):
        self.root = root
        (root / "images").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "library.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memes (
                id TEXT PRIMARY KEY, digest TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL, status TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '', description TEXT NOT NULL DEFAULT '',
                document TEXT NOT NULL DEFAULT '',
                revision INTEGER NOT NULL DEFAULT 0,
                origin TEXT NOT NULL, provider_id TEXT NOT NULL, source_text TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '', added_seq INTEGER NOT NULL DEFAULT 0,
                source_ref TEXT NOT NULL DEFAULT '',
                use_count INTEGER NOT NULL DEFAULT 0,
                last_used_at REAL NOT NULL DEFAULT 0,
                recommendation_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS revisions (
                meme_id TEXT NOT NULL, revision INTEGER NOT NULL,
                name TEXT NOT NULL DEFAULT '', description TEXT NOT NULL DEFAULT '',
                document TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL, actor TEXT NOT NULL, created_at REAL NOT NULL,
                PRIMARY KEY (meme_id, revision)
            );
            CREATE TABLE IF NOT EXISTS cursors (session TEXT PRIMARY KEY, seq INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS sent_previews (meme_id TEXT PRIMARY KEY, frames TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tool_history_archive (
                id TEXT PRIMARY KEY, origin TEXT NOT NULL, cid TEXT NOT NULL,
                records TEXT NOT NULL, created_at REAL NOT NULL
            );
        """)
        self.db.execute("UPDATE memes SET status='queued' WHERE status='processing'")
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(memes)")}
        if "source_ref" not in columns:
            self.db.execute(
                "ALTER TABLE memes ADD COLUMN source_ref TEXT NOT NULL DEFAULT ''"
            )
        for field, declaration in (
            ("use_count", "INTEGER NOT NULL DEFAULT 0"),
            ("last_used_at", "REAL NOT NULL DEFAULT 0"),
            ("recommendation_count", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if field not in columns:
                self.db.execute(f"ALTER TABLE memes ADD COLUMN {field} {declaration}")
        # Preserve original legacy text in its existing column and revisions.
        for table in ("memes", "revisions"):
            columns = {row[1] for row in self.db.execute(f"PRAGMA table_info({table})")}
            for field in ("name", "description"):
                if field not in columns:
                    self.db.execute(
                        f"ALTER TABLE {table} ADD COLUMN {field} TEXT NOT NULL DEFAULT ''"
                    )
            if "document" in columns:
                for row in self.db.execute(
                    f"SELECT rowid,document FROM {table} WHERE name='' AND description='' AND document<>''"
                ).fetchall():
                    description = " ".join(row["document"].split())
                    name = re.split(r"[。！？；，:：]", description, maxsplit=1)[
                        0
                    ].strip("#* -")[:NAME_LIMIT]
                    self.db.execute(
                        f"UPDATE {table} SET name=?,description=? WHERE rowid=?",
                        (
                            name or "未命名表情",
                            description[:DESCRIPTION_LIMIT],
                            row["rowid"],
                        ),
                    )
        self.db.commit()

    def add(
        self, data: bytes, suffix: str, origin: str, provider_id: str, text: str
    ) -> tuple[dict, bool]:
        """Snapshot an image once across all groups.

        Args:
            data: Validated original image bytes.
            suffix: File extension obtained from image decoding.
            origin: Source session used to resolve the reply model.
            provider_id: Explicit provider override, or an empty string.
            text: Caption accompanying this image, used only for first recognition.

        Returns:
            Catalog record and whether this image is new.
        """
        digest = hashlib.sha256(data).hexdigest()
        existing = self.db.execute(
            "SELECT * FROM memes WHERE digest=?", (digest,)
        ).fetchone()
        if existing:
            return dict(existing), False
        meme_id = "m_" + digest[:24]
        filename = f"{meme_id}{suffix}"
        path = self.root / "images" / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        now = time.time()
        with self.db:
            self.db.execute(
                "INSERT INTO memes(id,digest,filename,status,origin,provider_id,source_text,created_at,updated_at) VALUES(?,?,?,'queued',?,?,?,?,?)",
                (meme_id, digest, filename, origin, provider_id, text, now, now),
            )
        return self.get(meme_id), True

    def get(self, meme_id: str) -> dict:
        """Return an exact record or fail without accepting file paths.

        Args:
            meme_id: Stable catalog identifier.

        Returns:
            Current catalog record.

        Raises:
            ValueError: The record does not exist.
        """
        row = self.db.execute("SELECT * FROM memes WHERE id=?", (meme_id,)).fetchone()
        if row is None:
            raise ValueError("image_not_found")
        return dict(row)

    def download_error(
        self, reference: str, origin: str, provider_id: str, text: str, error: str
    ) -> dict:
        """Retain a failed download job for visibility and explicit retry.

        Args:
            reference: Original media reference, kept only in local runtime data.
            origin: Source conversation.
            provider_id: Selected provider override.
            text: Accompanying message text.
            error: Safe failure code without URLs or response contents.

        Returns:
            Failed download job, available for dashboard retry.
        """
        digest = "download:" + hashlib.sha256((origin + reference).encode()).hexdigest()
        meme_id = "download_" + digest[-24:]
        now = time.time()
        with self.db:
            self.db.execute(
                "INSERT INTO memes(id,digest,filename,status,origin,provider_id,source_text,error,source_ref,created_at,updated_at) VALUES(?,?,'','failed',?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET error=excluded.error,updated_at=excluded.updated_at",
                (
                    meme_id,
                    digest,
                    origin,
                    provider_id,
                    text,
                    error,
                    reference,
                    now,
                    now,
                ),
            )

        return self.get(meme_id)

    def path(self, record: dict) -> Path:
        return self.root / "images" / record["filename"]

    def sent_preview(self, meme_id: str) -> list[str] | None:
        row = self.db.execute(
            "SELECT frames FROM sent_previews WHERE meme_id=?", (meme_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def cache_sent_preview(self, meme_id: str, frames: list[str]) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO sent_previews VALUES (?,?)",
                (meme_id, json.dumps(frames)),
            )

    def archive_tool_history(self, origin, cid, history, entries):
        kept = dict(entries)
        records = [
            {"index": i, "message": message}
            for i, message in enumerate(history)
            if kept.get(i) != message
        ]
        if not records:
            return
        encoded = json.dumps(records, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(
            json.dumps(
                [origin, cid, history], ensure_ascii=False, sort_keys=True
            ).encode()
        ).hexdigest()
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO tool_history_archive VALUES (?,?,?,?,?)",
                (digest, origin, cid, encoded, time.time()),
            )

    def metadata(self, record: dict) -> dict:
        """Expose the same current metadata and original-file path to every client.

        Args:
            record: Current catalog row.

        Returns:
            Stable ID, name, short description, revision and absolute local path.
        """
        return {
            "id": record["id"],
            "name": record["name"],
            "description": record["description"],
            "revision": record["revision"],
            "local_path": str(self.path(record).resolve())
            if record["filename"]
            else "",
        }

    def records(
        self, status: str = "", query: str = "", limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """Read a stable catalog page or active metadata for semantic search.

        Args:
            status: Optional exact processing state.
            query: Optional literal substring of the name, description or ID.
            limit: Maximum records to return.
            offset: Number of matching records to skip.

        Returns:
            Matching current catalog records.
        """
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM memes WHERE (?='' OR status=?) AND (?='' OR instr(name,?)>0 OR instr(description,?)>0 OR instr(id,?)>0) ORDER BY created_at DESC,id LIMIT ? OFFSET ?",
                (status, status, query, query, query, query, limit, offset),
            )
        ]

    def record_use(self, meme_id: str) -> None:
        """Count one received image or successful bot send, independently of exposure.

        Args:
            meme_id: Stable ID of the image actually used in a message.

        Raises:
            ValueError: The image no longer exists.
        """
        with self.db:
            cursor = self.db.execute(
                "UPDATE memes SET use_count=use_count+1,last_used_at=? WHERE id=?",
                (time.time(), meme_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("image_not_found")

    def recommend(self) -> list[dict]:
        """Select four ranked memes and one uniform random remaining meme locally.

        Each exposure reduces the new-image bonus by one step, reaching zero
        after four recommendations. Within a step, use in the last 24 hours and
        lifetime usage frequency provide smaller bonuses. Recency and frequency
        take over once the new-image bonus is exhausted. Only actual received
        images and successful sends update usage; recommending only counts as
        exposure. Existing images start with zero statistics on upgrade.

        Returns:
            Up to five distinct ready records, with the random choice last.
            Every selected record receives one exposure, including the random
            choice. Original metadata, revisions and notice cursors stay intact.
        """
        now = time.time()
        with self.db:
            candidates = [
                row
                for row in self.records(status="ready", limit=2_000_000_000)
                if row["filename"]
                and row["name"]
                and row["description"]
                and self.path(row).is_file()
            ]
            # Compare priorities in order; no model, text matching or paid call.
            ranked = sorted(
                candidates,
                key=lambda row: (
                    max(0, 4 - row["recommendation_count"]),
                    row["last_used_at"]
                    if row["last_used_at"] > 0 and row["last_used_at"] >= now - 86400
                    else 0,
                    row["use_count"],
                    row["created_at"],
                    row["id"],
                ),
                reverse=True,
            )
            selected = ranked[:4]
            if len(ranked) > 4:
                selected.append(random.SystemRandom().choice(ranked[4:]))
            self.db.executemany(
                "UPDATE memes SET recommendation_count=recommendation_count+1 WHERE id=?",
                [(row["id"],) for row in selected],
            )
        return selected

    def counts(self) -> dict:
        return dict(
            self.db.execute("SELECT status,COUNT(*) FROM memes GROUP BY status")
        )

    def set_status(self, meme_id: str, status: str, error: str = "") -> None:
        with self.db:
            row = self.get(meme_id)
            seq = (
                self.latest_seq() + 1
                if status == "ready" and not row["added_seq"]
                else row["added_seq"]
            )
            self.db.execute(
                "UPDATE memes SET status=?,error=?,added_seq=?,updated_at=? WHERE id=?",
                (status, error, seq, time.time(), meme_id),
            )

    def recognize(
        self, meme_id: str, name: str, description: str, is_meme: bool
    ) -> None:
        """Commit first recognition only while the original job is unchanged.

        Args:
            meme_id: Image being recognized.
            name: Short image name produced by vision.
            description: Concise visual meaning and practical usage.
            is_meme: Whether the picture is usable as a meme.
        """
        name, description = validate_metadata(name, description)
        with self.db:
            row = self.get(meme_id)
            if row["revision"] or row["status"] != "processing":
                if row["status"] == "processing":
                    self.set_status(
                        meme_id, "failed", "metadata_changed_during_recognition"
                    )
                return
            seq = self.latest_seq() + 1 if is_meme else 0
            self.db.execute(
                "UPDATE memes SET name=?,description=?,revision=1,status=?,added_seq=?,error='',updated_at=? WHERE id=?",
                (
                    name,
                    description,
                    "ready" if is_meme else "rejected",
                    seq,
                    time.time(),
                    meme_id,
                ),
            )
            self.db.execute(
                "INSERT INTO revisions(meme_id,revision,name,description,document,reason,actor,created_at) VALUES(?,1,?,?,'','首次识别','vision',?)",
                (meme_id, name, description, time.time()),
            )

    def edit(
        self,
        meme_id: str,
        name: str,
        description: str,
        revision: int,
        reason: str,
        actor: str,
    ) -> dict:
        """Replace the two editable fields with optimistic concurrency control.

        Args:
            meme_id: Image whose metadata is edited.
            name: Complete short image name.
            description: Complete short description.
            revision: Revision the editor has actually read.
            reason: Explanation retained in the revision log.
            actor: Editing actor, such as the model or dashboard user.

        Returns:
            Updated record, including the new revision.

        Raises:
            ValueError: Missing content or a revision conflict.
        """
        name, description = validate_metadata(name, description)
        with self.db:
            row = self.get(meme_id)
            if row["revision"] != revision:
                raise ValueError("revision_conflict")
            next_revision = revision + 1
            self.db.execute(
                "UPDATE memes SET name=?,description=?,revision=?,updated_at=? WHERE id=?",
                (name, description, next_revision, time.time(), meme_id),
            )
            self.db.execute(
                "INSERT INTO revisions(meme_id,revision,name,description,document,reason,actor,created_at) VALUES(?,?,?,?,'',?,?,?)",
                (meme_id, next_revision, name, description, reason, actor, time.time()),
            )
        return self.get(meme_id)

    def disable(self, meme_id: str, revision: int, reason: str, actor: str) -> dict:
        """Retire a reviewed meme while preserving its file and metadata.

        Args:
            meme_id: Stable ID of the meme to retire from the shared library.
            revision: Integer revision the caller has just reviewed.
            reason: Nonempty explanation retained in the revision log.
            actor: Model and source conversation making the change.

        Returns:
            Disabled record with its new revision, or the unchanged record if
            already disabled at the expected revision.

        Raises:
            ValueError: Invalid input, stale revision or unavailable image.
        """
        if type(revision) is not int or revision < 1:
            raise ValueError("invalid_revision")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason_required")
        with self.db:
            row = self.get(meme_id)
            if row["revision"] != revision:
                raise ValueError("revision_conflict")
            if row["status"] == "disabled":
                return row
            if row["status"] != "ready":
                raise ValueError("image_not_available")
            now = time.time()
            next_revision = revision + 1
            self.db.execute(
                "UPDATE memes SET status='disabled',revision=?,error='',updated_at=? WHERE id=?",
                (next_revision, now, meme_id),
            )
            self.db.execute(
                "INSERT INTO revisions(meme_id,revision,name,description,document,reason,actor,created_at) VALUES(?,?,?,?,'',?,?,?)",
                (
                    meme_id,
                    next_revision,
                    row["name"],
                    row["description"],
                    "停用：" + reason.strip(),
                    actor,
                    now,
                ),
            )
        return self.get(meme_id)

    def revisions(self, meme_id: str) -> list[dict]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM revisions WHERE meme_id=? ORDER BY revision DESC",
                (meme_id,),
            )
        ]

    def latest_seq(self) -> int:
        return self.db.execute(
            "SELECT COALESCE(MAX(added_seq),0) FROM memes"
        ).fetchone()[0]

    def unseen(self, session: str) -> int:
        row = self.db.execute(
            "SELECT seq FROM cursors WHERE session=?", (session,)
        ).fetchone()
        return self.latest_seq() if row is None else max(0, self.latest_seq() - row[0])

    def acknowledge(self, session: str, seq: int) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO cursors VALUES(?,?) ON CONFLICT(session) DO UPDATE SET seq=MAX(cursors.seq,excluded.seq)",
                (session, seq),
            )

    def close(self) -> None:
        self.db.close()
