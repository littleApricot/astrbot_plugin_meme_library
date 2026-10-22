"""Persist one editable document per image and a shared collection queue."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path


class Library:
    """Own the shared image catalog, document revisions and notification cursors."""

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
                document TEXT NOT NULL DEFAULT '', revision INTEGER NOT NULL DEFAULT 0,
                origin TEXT NOT NULL, provider_id TEXT NOT NULL, source_text TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '', added_seq INTEGER NOT NULL DEFAULT 0,
                source_ref TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS revisions (
                meme_id TEXT NOT NULL, revision INTEGER NOT NULL, document TEXT NOT NULL,
                reason TEXT NOT NULL, actor TEXT NOT NULL, created_at REAL NOT NULL,
                PRIMARY KEY (meme_id, revision)
            );
            CREATE TABLE IF NOT EXISTS cursors (session TEXT PRIMARY KEY, seq INTEGER NOT NULL);
        """)
        self.db.execute("UPDATE memes SET status='queued' WHERE status='processing'")
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(memes)")}
        if "source_ref" not in columns:
            self.db.execute(
                "ALTER TABLE memes ADD COLUMN source_ref TEXT NOT NULL DEFAULT ''"
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

    def records(
        self, status: str = "", query: str = "", limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """Read a stable catalog page or all active documents for semantic search.

        Args:
            status: Optional exact processing state.
            query: Optional literal substring of the document or identifier.
            limit: Maximum records to return.
            offset: Number of matching records to skip.

        Returns:
            Matching current catalog records.
        """
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM memes WHERE (?='' OR status=?) AND (?='' OR instr(document,?)>0 OR instr(id,?)>0) ORDER BY created_at DESC,id LIMIT ? OFFSET ?",
                (status, status, query, query, query, limit, offset),
            )
        ]

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

    def recognize(self, meme_id: str, document: str, is_meme: bool) -> None:
        """Commit first recognition only while the original job is unchanged.

        Args:
            meme_id: Image being recognized.
            document: Single unified description produced by vision.
            is_meme: Whether the picture is usable as a meme.
        """
        with self.db:
            row = self.get(meme_id)
            if row["revision"] or row["status"] != "processing":
                if row["status"] == "processing":
                    self.set_status(
                        meme_id, "failed", "document_changed_during_recognition"
                    )
                return
            seq = self.latest_seq() + 1 if is_meme else 0
            self.db.execute(
                "UPDATE memes SET document=?,revision=1,status=?,added_seq=?,error='',updated_at=? WHERE id=?",
                (
                    document,
                    "ready" if is_meme else "rejected",
                    seq,
                    time.time(),
                    meme_id,
                ),
            )
            self.db.execute(
                "INSERT INTO revisions VALUES(?,1,?,'首次识别','vision',?)",
                (meme_id, document, time.time()),
            )

    def edit(
        self, meme_id: str, document: str, revision: int, reason: str, actor: str
    ) -> dict:
        """Replace the same document with optimistic concurrency control.

        Args:
            meme_id: Image whose document is edited.
            document: Complete new unified document.
            revision: Revision the editor has actually read.
            reason: Explanation retained in the revision log.
            actor: Editing actor, such as the model or dashboard user.

        Returns:
            Updated record, including the new revision.

        Raises:
            ValueError: Missing content or a revision conflict.
        """
        if not document.strip() or len(document) > 12000:
            raise ValueError("invalid_document")
        with self.db:
            row = self.get(meme_id)
            if row["revision"] != revision:
                raise ValueError("revision_conflict")
            next_revision = revision + 1
            self.db.execute(
                "UPDATE memes SET document=?,revision=?,updated_at=? WHERE id=?",
                (document.strip(), next_revision, time.time(), meme_id),
            )
            self.db.execute(
                "INSERT INTO revisions VALUES(?,?,?,?,?,?)",
                (meme_id, next_revision, document.strip(), reason, actor, time.time()),
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
