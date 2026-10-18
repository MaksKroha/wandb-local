"""Local, versioned artifacts.

Layout::

    <wandb dir>/artifacts/<project>/<name>/
        index.json          # {"versions": [...], "aliases": {"latest": "v3", ...}}
        v0/manifest.json    # metadata, aliases, entries (sha256/size per file)
        v0/files/...        # the artifact content

Versions are content addressed: logging an artifact whose digest equals an
existing version only moves aliases, exactly like the real service.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from . import env as _env
from .errors import Error
from .util import atomic_write, read_json, sha256_file, utc_iso

try:  # pragma: no cover - platform specific
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

_NAME_RE = re.compile(r"^(?:(?P<entity>[^/:]+)/)?(?:(?P<project>[^/:]+)/)?(?P<name>[^/:]+)(?::(?P<alias>[^/:]+))?$")


def parse_artifact_path(path: str) -> tuple[str | None, str | None, str, str]:
    m = _NAME_RE.match(path.strip())
    if not m:
        raise ValueError(f"Invalid artifact name: {path!r}")
    entity, project = m.group("entity"), m.group("project")
    if entity and not project:  # "project/name"
        entity, project = None, entity
    return entity, project, m.group("name"), m.group("alias") or "latest"


def artifacts_root(wandb_dir: str | os.PathLike | None = None) -> Path:
    override = _env.get(_env.LOCAL_ARTIFACTS_DIR)
    if override:
        return Path(override)
    if wandb_dir is None:
        root = _env.get_dir() or os.getcwd()
        wandb_dir = Path(root) / "wandb"
    return Path(wandb_dir) / "artifacts"


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        if fcntl is not None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)


class ArtifactManifestEntry:
    def __init__(self, artifact: "Artifact", path: str, digest: str | None, size: int | None, ref: str | None = None, local_path: str | None = None):
        self._artifact = artifact
        self.path = path
        self.digest = digest
        self.size = size
        self.ref = ref
        self.local_path = local_path

    def download(self, root: str | None = None, skip_cache: bool | None = None) -> str:
        src = self._artifact._resolve_entry_file(self.path)
        if root is None:
            return src
        dst = Path(root) / self.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst)

    def ref_target(self) -> str | None:
        return self.ref

    def __repr__(self) -> str:
        return f"<ArtifactManifestEntry {self.path} size={self.size}>"


class _NewFileWriter:
    def __init__(self, artifact: "Artifact", name: str, mode: str, encoding: str | None):
        self.artifact, self.name, self.mode, self.encoding = artifact, name, mode, encoding


class Artifact:
    """A versioned bundle of files (datasets, models, code...) stored locally."""

    def __init__(
        self,
        name: str,
        type: str,
        description: str | None = None,
        metadata: dict | None = None,
        incremental: bool = False,
        use_as: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not re.match(r"^[\w\-.]+$", name):
            raise ValueError(f"Artifact name may only contain alphanumeric characters, dashes, underscores and dots: {name!r}")
        self._name = name
        self.type = type
        self.description = description
        self.metadata = dict(metadata or {})
        self.incremental = incremental
        self._entries: dict[str, dict] = {}  # staged: path -> {"src": abs path | None, "ref": uri | None}
        self._staging: str | None = None
        self._state = "PENDING"
        self._version: str | None = None
        self._aliases: list[str] = []
        self._tags: list[str] = []
        self._dir: Path | None = None  # materialised version dir
        self._project: str | None = None
        self._entity: str | None = None
        self._manifest: dict | None = None
        self._ttl = kwargs.get("ttl")

    # -- identity ------------------------------------------------------------
    @property
    def name(self) -> str:
        return f"{self._name}:{self._version}" if self._version else self._name

    @property
    def source_name(self) -> str:
        return self.name

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def aliases(self) -> list[str]:
        return list(self._aliases)

    @aliases.setter
    def aliases(self, value: list[str]) -> None:
        self._aliases = list(value)

    @property
    def tags(self) -> list[str]:
        return list(self._tags)

    @tags.setter
    def tags(self, value: list[str]) -> None:
        self._tags = list(value)

    @property
    def state(self) -> str:
        return self._state

    @property
    def id(self) -> str | None:
        return (self._manifest or {}).get("id")

    @property
    def digest(self) -> str | None:
        return (self._manifest or {}).get("digest")

    @property
    def size(self) -> int:
        if self._manifest:
            return int(self._manifest.get("size", 0))
        return sum(os.path.getsize(e["src"]) for e in self._entries.values() if e.get("src") and os.path.exists(e["src"]))

    @property
    def qualified_name(self) -> str:
        return f"{self._entity or 'local'}/{self._project or 'uncategorized'}/{self.name}"

    @property
    def project(self) -> str | None:
        return self._project

    @property
    def entity(self) -> str | None:
        return self._entity

    @property
    def created_at(self) -> str | None:
        return (self._manifest or {}).get("created_at")

    @property
    def file_count(self) -> int:
        return len(self._manifest["entries"]) if self._manifest else len(self._entries)

    @property
    def manifest(self) -> dict:
        return self._manifest or {"entries": {k: {"local_path": v.get("src"), "ref": v.get("ref")} for k, v in self._entries.items()}}

    def __repr__(self) -> str:
        return f"<Artifact {self.qualified_name} type={self.type}>"

    # -- building ----------------------------------------------------------------
    def _require_draft(self) -> None:
        if self._state != "PENDING":
            raise Error("This artifact is already logged; call artifact.new_draft() to modify it.")

    def _staging_dir(self) -> str:
        if self._staging is None:
            self._staging = tempfile.mkdtemp(prefix=f"wandb-artifact-{self._name}-")
        return self._staging

    @property
    def files_dir(self) -> str:
        """Lets media objects serialise themselves into the artifact (``artifact.add``)."""
        return self._staging_dir()

    @contextlib.contextmanager
    def new_file(self, name: str, mode: str = "x", encoding: str | None = None):
        self._require_draft()
        path = Path(self._staging_dir()) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if "b" not in mode and encoding is None:
            encoding = "utf-8"
        if "x" in mode and name in self._entries:
            raise ValueError(f"File with name {name!r} already exists")
        with open(path, mode.replace("x", "w") if name in self._entries else mode, encoding=encoding) as fh:
            yield fh
        self._entries[name] = {"src": str(path)}

    def add_file(self, local_path: str, name: str | None = None, is_tmp: bool | None = False, skip_cache: bool | None = False, policy: str | None = "mutable", overwrite: bool = False) -> ArtifactManifestEntry:
        self._require_draft()
        if not os.path.isfile(local_path):
            raise ValueError(f"Path is not a file: {local_path}")
        name = name or os.path.basename(local_path)
        if is_tmp:
            stem, ext = os.path.splitext(name)
            name = f"{stem}.{sha256_file(local_path)[:20]}{ext}"
        if name in self._entries and not overwrite and self._entries[name].get("src") != os.path.abspath(local_path):
            raise ValueError(f"File with name {name!r} already exists in the artifact (pass overwrite=True)")
        # The file is copied into the artifact store when the artifact is logged
        # (synchronously), so there is no need to snapshot it into staging here.
        src = os.path.abspath(local_path)
        self._entries[name] = {"src": src}
        return ArtifactManifestEntry(self, name, None, os.path.getsize(src), local_path=src)

    def add_dir(self, local_path: str, name: str | None = None, skip_cache: bool | None = False, policy: str | None = "mutable", merge: bool = False) -> None:
        self._require_draft()
        if not os.path.isdir(local_path):
            raise ValueError(f"Path is not a directory: {local_path}")
        prefix = (name.strip("/") + "/") if name else ""
        for root, _dirs, files in os.walk(local_path):
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, local_path).replace(os.sep, "/")
                self.add_file(full, prefix + rel, policy=policy, overwrite=True)

    def add_reference(self, uri: str, name: str | None = None, checksum: bool = True, max_objects: int | None = None) -> list[ArtifactManifestEntry]:
        self._require_draft()
        local = uri[len("file://"):] if uri.startswith("file://") else None
        entries = []
        if local and os.path.isdir(local):
            for root, _d, files in os.walk(local):
                for fname in files:
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, local).replace(os.sep, "/")
                    key = f"{name}/{rel}" if name else rel
                    self._entries[key] = {"ref": "file://" + os.path.abspath(full), "checksum": checksum}
                    entries.append(ArtifactManifestEntry(self, key, None, None, ref=self._entries[key]["ref"]))
        else:
            key = name or os.path.basename(uri.rstrip("/")) or uri
            self._entries[key] = {"ref": uri, "checksum": checksum and bool(local)}
            entries.append(ArtifactManifestEntry(self, key, None, None, ref=uri))
        return entries

    def add(self, obj: Any, name: str, overwrite: bool = False) -> ArtifactManifestEntry:
        from .data_types import WBValue, save_value_into

        self._require_draft()
        if not isinstance(obj, WBValue):
            raise TypeError("artifact.add() expects a wandb media object (Table, Image, ...)")
        fname = save_value_into(self._staging_dir(), name, obj)
        # Media referenced by the object (e.g. images inside a table) live in staging/media.
        for root, _d, files in os.walk(self._staging_dir()):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, self._staging_dir()).replace(os.sep, "/")
                self._entries.setdefault(rel, {"src": full})
        return ArtifactManifestEntry(self, fname, None, os.path.getsize(os.path.join(self._staging_dir(), fname)))

    def remove(self, item: str) -> None:
        self._require_draft()
        prefix = item.rstrip("/") + "/"
        for key in [k for k in self._entries if k == item or k.startswith(prefix)]:
            del self._entries[key]

    def __setitem__(self, name: str, item: Any) -> None:
        self.add(item, name)

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    # -- reading -------------------------------------------------------------------
    def _entry_map(self) -> dict:
        return self._manifest["entries"] if self._manifest else {k: {"local_path": v.get("src"), "ref": v.get("ref")} for k, v in self._entries.items()}

    def _resolve_entry_file(self, name: str) -> str:
        entries = self._entry_map()
        if name not in entries:
            raise KeyError(f"{name!r} not found in artifact {self.name}")
        entry = entries[name]
        if entry.get("ref"):
            ref = entry["ref"]
            if ref.startswith("file://"):
                return ref[len("file://"):]
            return ref
        if self._dir is not None:
            return str(self._dir / "files" / name)
        return entry["local_path"]

    def get_entry(self, name: str) -> ArtifactManifestEntry:
        entry = self._entry_map().get(name)
        if entry is None:
            raise KeyError(f"{name!r} not found in artifact {self.name}")
        return ArtifactManifestEntry(self, name, entry.get("sha256"), entry.get("size"), ref=entry.get("ref"), local_path=self._resolve_entry_file(name))

    get_path = get_entry

    def get(self, name: str) -> Any:
        from .data_types import Table

        entries = self._entry_map()
        candidates = [name, f"{name}.table.json"]
        for cand in candidates:
            if cand in entries and cand.endswith(".table.json"):
                with open(self._resolve_entry_file(cand), encoding="utf-8") as fh:
                    return Table.from_json(json.load(fh))
        for key in entries:
            if key.startswith(name + ".") and key.endswith(".json"):
                with open(self._resolve_entry_file(key), encoding="utf-8") as fh:
                    return json.load(fh)
        return None

    def files(self, names: list[str] | None = None, per_page: int = 50) -> list[ArtifactManifestEntry]:
        entries = self._entry_map()
        keys = names if names is not None else sorted(entries)
        return [self.get_entry(k) for k in keys if k in entries]

    def download(self, root: str | None = None, allow_missing_references: bool = False, skip_cache: bool | None = None, path_prefix: str | None = None, **kwargs: Any) -> str:
        if self._dir is None:
            raise Error("Artifact must be logged (or fetched with use_artifact) before it can be downloaded")
        src_root = self._dir / "files"
        if root is None:
            return str(src_root)
        dst = Path(root)
        for name in self._entry_map():
            if path_prefix and not name.startswith(path_prefix):
                continue
            entry_file = self._resolve_entry_file(name)
            if not os.path.exists(entry_file):
                if allow_missing_references:
                    continue
                raise Error(f"Missing file for artifact entry {name}: {entry_file}")
            out = dst / name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry_file, out)
        return str(dst)

    def checkout(self, root: str | None = None) -> str:
        return self.download(root)

    def file(self, root: str | None = None) -> str:
        names = list(self._entry_map())
        if len(names) != 1:
            raise ValueError("artifact.file() can only be used on artifacts containing exactly one file")
        path = self.download(root)
        return os.path.join(path, names[0])

    def verify(self, root: str | None = None) -> None:
        for name, entry in self._entry_map().items():
            if entry.get("ref") or not entry.get("sha256"):
                continue
            if sha256_file(self._resolve_entry_file(name)) != entry["sha256"]:
                raise ValueError(f"Digest mismatch for {name}")

    def wait(self, timeout: int | None = None) -> "Artifact":
        return self

    def is_draft(self) -> bool:
        return self._state == "PENDING"

    def new_draft(self) -> "Artifact":
        draft = Artifact(self._name, self.type, self.description, dict(self.metadata))
        for name in self._entry_map():
            entry = self._entry_map()[name]
            if entry.get("ref"):
                draft._entries[name] = {"ref": entry["ref"]}
            else:
                draft._entries[name] = {"src": self._resolve_entry_file(name)}
        return draft

    def logged_by(self) -> Any:
        return (self._manifest or {}).get("logged_by")

    def used_by(self) -> list:
        return []

    def save(self, project: str | None = None, settings: Any = None) -> None:
        import wandb

        if wandb.run is not None and not getattr(wandb.run, "disabled", False):
            wandb.run.log_artifact(self)
        else:
            commit_artifact(self, artifacts_root(), project=project or _env.get_project() or "uncategorized", entity=_env.get_entity() or "local", run_info=None, aliases=None)

    def link(self, target_path: str, aliases: list[str] | None = None) -> None:
        """Registry links have no local meaning; the aliases are added to this version instead."""
        if self._dir is not None and aliases:
            _add_aliases(self._dir.parent, self._version or "v0", aliases)
            self._aliases = sorted(set(self._aliases) | set(aliases))

    def delete(self, delete_aliases: bool = False) -> None:
        if self._dir is None:
            return
        index_path = self._dir.parent / "index.json"
        with _locked(self._dir.parent / ".lock"):
            index = read_json(index_path, {"versions": [], "aliases": {}})
            if not delete_aliases and any(v == self._version for v in index.get("aliases", {}).values()):
                raise Error("Artifact has aliases; pass delete_aliases=True to delete it")
            index["versions"] = [v for v in index.get("versions", []) if v != self._version]
            index["aliases"] = {a: v for a, v in index.get("aliases", {}).items() if v != self._version}
            atomic_write(index_path, json.dumps(index, indent=2))
        shutil.rmtree(self._dir, ignore_errors=True)

    def json_encode(self) -> dict:
        return {"_type": "artifactVersion", "name": self.qualified_name}


# ---------------------------------------------------------------- storage --
def _artifact_mode() -> str:
    mode = (_env.get(_env.LOCAL_ARTIFACT_MODE) or "copy").lower()
    return mode if mode in ("copy", "hardlink", "symlink") else "copy"


def _place(src: str, dst: str, mode: str) -> None:
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if os.path.lexists(dst):
        os.unlink(dst)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    elif mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
        return
    shutil.copy2(src, dst)


def _compute_entries(artifact: Artifact) -> dict:
    entries = {}
    for name, staged in sorted(artifact._entries.items()):
        if staged.get("ref"):
            ref = staged["ref"]
            entry: dict = {"ref": ref}
            local = ref[len("file://"):] if ref.startswith("file://") else None
            if local and os.path.isfile(local):
                entry["size"] = os.path.getsize(local)
                if staged.get("checksum", True):
                    entry["sha256"] = sha256_file(local)
            entries[name] = entry
        else:
            src = staged["src"]
            entries[name] = {"sha256": sha256_file(src), "size": os.path.getsize(src), "local_path": src}
    return entries


def _digest(entries: dict) -> str:
    h = hashlib.sha256()
    for name, entry in sorted(entries.items()):
        h.update(name.encode())
        h.update(str(entry.get("sha256") or entry.get("ref")).encode())
    return h.hexdigest()


def _add_aliases(art_dir: Path, version: str, aliases: list[str]) -> None:
    index_path = art_dir / "index.json"
    with _locked(art_dir / ".lock"):
        index = read_json(index_path, {"versions": [], "aliases": {}})
        for alias in aliases:
            index.setdefault("aliases", {})[alias] = version
        atomic_write(index_path, json.dumps(index, indent=2))
        _sync_manifest_aliases(art_dir, index)


def _sync_manifest_aliases(art_dir: Path, index: dict) -> None:
    for version in index.get("versions", []):
        mpath = art_dir / version / "manifest.json"
        manifest = read_json(mpath)
        if manifest is None:
            continue
        aliases = sorted(a for a, v in index.get("aliases", {}).items() if v == version)
        if manifest.get("aliases") != aliases:
            manifest["aliases"] = aliases
            atomic_write(mpath, json.dumps(manifest, indent=2))


def commit_artifact(artifact: Artifact, root: Path, project: str, entity: str, run_info: dict | None, aliases: list[str] | None, tags: list[str] | None = None) -> Artifact:
    """Materialise a draft artifact into ``root`` and return it in its logged state."""
    art_dir = Path(root) / project / artifact._name
    art_dir.mkdir(parents=True, exist_ok=True)
    entries = _compute_entries(artifact)
    digest = _digest(entries)
    wanted_aliases = list(dict.fromkeys(["latest", *(aliases or []), *artifact._aliases]))
    with _locked(art_dir / ".lock"):
        index = read_json(art_dir / "index.json", {"versions": [], "aliases": {}})
        version = None
        for v in index.get("versions", []):
            m = read_json(art_dir / v / "manifest.json", {})
            if m.get("digest") == digest:
                version = v
                break
        if version is None:
            numbers = [int(v[1:]) for v in index.get("versions", []) if v[1:].isdigit()]
            version = f"v{max(numbers) + 1 if numbers else 0}"
            vdir = art_dir / version
            files_dir = vdir / "files"
            files_dir.mkdir(parents=True, exist_ok=True)
            mode = _artifact_mode()
            staging = os.path.abspath(artifact._staging) + os.sep if artifact._staging else None
            for name, entry in entries.items():
                local = entry.pop("local_path", None)
                if local is None:
                    continue
                dst = files_dir / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if staging and os.path.abspath(local).startswith(staging):
                    shutil.move(local, dst)  # staged files are temporary: move instead of copy
                else:
                    _place(local, str(dst), mode)
            manifest = {
                "id": hashlib.sha256(f"{project}/{artifact._name}/{version}/{digest}".encode()).hexdigest()[:24],
                "name": artifact._name,
                "version": version,
                "type": artifact.type,
                "description": artifact.description,
                "metadata": artifact.metadata,
                "digest": digest,
                "size": sum(int(e.get("size") or 0) for e in entries.values()),
                "created_at": utc_iso(),
                "project": project,
                "entity": entity,
                "logged_by": run_info,
                "tags": list(tags or artifact._tags),
                "entries": entries,
                "aliases": [],
            }
            atomic_write(vdir / "manifest.json", json.dumps(manifest, indent=2, default=str))
            index.setdefault("versions", []).append(version)
        else:
            for entry in entries.values():
                entry.pop("local_path", None)
        latest = index.get("aliases", {}).get("latest")
        for alias in wanted_aliases:
            if alias == "latest" and latest is not None and version != latest:
                # only move "latest" forward
                if int(version[1:]) < int(latest[1:]):
                    continue
            index.setdefault("aliases", {})[alias] = version
        atomic_write(art_dir / "index.json", json.dumps(index, indent=2))
        _sync_manifest_aliases(art_dir, index)
    if artifact._staging:
        shutil.rmtree(artifact._staging, ignore_errors=True)
        artifact._staging = None
    # Update the caller's object in place (``art.version`` etc. work after log_artifact).
    artifact.__dict__.update(load_artifact(art_dir, version).__dict__)
    return artifact


def load_artifact(art_dir: Path, version: str) -> Artifact:
    manifest = read_json(Path(art_dir) / version / "manifest.json")
    if manifest is None:
        raise Error(f"Artifact version not found: {art_dir}/{version}")
    art = Artifact.__new__(Artifact)
    Artifact.__init__(art, manifest["name"], manifest.get("type", "unspecified"), manifest.get("description"), manifest.get("metadata"))
    art._state = "COMMITTED"
    art._version = version
    art._aliases = list(manifest.get("aliases", []))
    art._tags = list(manifest.get("tags", []))
    art._dir = Path(art_dir) / version
    art._project = manifest.get("project")
    art._entity = manifest.get("entity")
    art._manifest = manifest
    return art


def find_artifact(path: str, roots: list[Path], default_project: str | None = None) -> Artifact:
    _entity, project, name, alias = parse_artifact_path(path)
    searched = []
    for root in roots:
        root = Path(root)
        projects = [project] if project else ([default_project] if default_project else [])
        projects += [p.name for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
        for proj in dict.fromkeys(p for p in projects if p):
            art_dir = root / proj / name
            searched.append(str(art_dir))
            index = read_json(art_dir / "index.json")
            if not index:
                continue
            version = index.get("aliases", {}).get(alias)
            if version is None and re.match(r"^v\d+$", alias) and alias in index.get("versions", []):
                version = alias
            if version is not None:
                return load_artifact(art_dir, version)
    raise Error(f"Artifact {path!r} not found locally (searched: {', '.join(searched) or 'nothing'})")


def list_artifacts(roots: list[Path]) -> list[dict]:
    out = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for proj in sorted(p for p in root.iterdir() if p.is_dir()):
            for art_dir in sorted(a for a in proj.iterdir() if a.is_dir()):
                index = read_json(art_dir / "index.json")
                if not index:
                    continue
                versions = []
                for v in index.get("versions", []):
                    m = read_json(art_dir / v / "manifest.json", {})
                    versions.append({
                        "version": v,
                        "aliases": m.get("aliases", []),
                        "size": m.get("size"),
                        "created_at": m.get("created_at"),
                        "digest": m.get("digest"),
                        "metadata": m.get("metadata"),
                        "files": len(m.get("entries", {})),
                        "logged_by": m.get("logged_by"),
                        "dir": str(art_dir / v / "files"),
                    })
                type_ = read_json(art_dir / index["versions"][-1] / "manifest.json", {}).get("type") if index.get("versions") else None
                out.append({"project": proj.name, "name": art_dir.name, "type": type_, "versions": versions, "root": str(root)})
    return out


def wait_all() -> None:
    """Artifacts are written synchronously, nothing to wait for."""
    time.sleep(0)
