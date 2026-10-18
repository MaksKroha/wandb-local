"""Run metadata (wandb-metadata.json) and periodic system metrics (wandb-events.jsonl).

Metric keys follow the real client (``cpu``, ``memory_percent``, ``gpu.0.gpu``,
``gpu.0.memoryAllocated`` ...).  psutil / pynvml are used when importable;
otherwise /proc and ``nvidia-smi`` are read directly.
"""

from __future__ import annotations

import getpass
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

from .util import utc_iso


def _run(cmd: list[str], cwd: str | None = None, timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _git_info(root: str) -> dict:
    if shutil.which("git") is None:
        return {}
    top = _run(["git", "rev-parse", "--show-toplevel"], cwd=root, timeout=3)
    if not top:
        return {}
    info = {"root": top}
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root, timeout=3)
    if commit:
        info["commit"] = commit
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, timeout=3)
    if branch:
        info["branch"] = branch
    remote = _run(["git", "config", "--get", "remote.origin.url"], cwd=root, timeout=3)
    if remote:
        info["remote"] = remote
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, timeout=5)
    if dirty is not None:
        info["dirty"] = bool(dirty)
    return info


def _meminfo() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                out[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return out


def query_gpus() -> list[dict]:
    """Static GPU description (name, memory, uuid)."""
    gpus: list[dict] = []
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            gpus.append({
                "name": name.decode() if isinstance(name, bytes) else name,
                "memoryTotal": str(pynvml.nvmlDeviceGetMemoryInfo(h).total),
                "uuid": pynvml.nvmlDeviceGetUUID(h) if hasattr(pynvml, "nvmlDeviceGetUUID") else None,
            })
        return gpus
    except Exception:
        pass
    if shutil.which("nvidia-smi") is None:
        return gpus
    text = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,uuid,driver_version", "--format=csv,noheader,nounits"])
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                mem = str(int(float(parts[2]) * 1024 * 1024))
            except ValueError:
                mem = None
            gpus.append({"name": parts[1], "memoryTotal": mem, "uuid": parts[3], "driver": parts[4] if len(parts) > 4 else None})
    return gpus


def collect_metadata(start_time: float, program: str | None, root: str) -> dict:
    meta: dict[str, Any] = {
        "os": platform.platform(),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "startedAt": utc_iso(start_time),
        "args": list(sys.argv[1:]),
        "host": socket.gethostname(),
        "executable": sys.executable,
        "cpu_count_logical": os.cpu_count(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
    }
    try:
        meta["username"] = getpass.getuser()
    except Exception:
        pass
    if program:
        meta["program"] = program
        try:
            meta["codePath"] = os.path.relpath(program, root)
        except ValueError:
            meta["codePath"] = program
    git = _git_info(root)
    meta["root"] = git.get("root", root)
    if git:
        meta["git"] = {k: v for k, v in git.items() if k != "root"}
    try:
        import psutil  # type: ignore

        meta["cpu_count"] = psutil.cpu_count(logical=False)
        meta["memory"] = {"total": str(psutil.virtual_memory().total)}
    except Exception:
        mem = _meminfo().get("MemTotal")
        if mem:
            meta["memory"] = {"total": str(mem)}
    try:
        du = shutil.disk_usage("/")
        meta["disk"] = {"/": {"total": str(du.total), "used": str(du.used)}}
    except OSError:
        pass
    gpus = query_gpus()
    if gpus:
        meta["gpu"] = gpus[0]["name"]
        meta["gpu_count"] = len(gpus)
        meta["gpu_nvidia"] = gpus
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch.version, "cuda", None):
        meta["cudaVersion"] = torch.version.cuda
    return meta


def requirements() -> str:
    try:
        from importlib import metadata as md
    except ImportError:  # pragma: no cover
        return ""
    seen = {}
    for dist in md.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:
            continue
        if name and name.lower() not in seen:
            seen[name.lower()] = f"{name}=={dist.version}"
    return "\n".join(sorted(seen.values(), key=str.lower)) + "\n"


class SystemSampler:
    """Takes one sample of system metrics per call to :meth:`sample`."""

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._ncpu = os.cpu_count() or 1
        self._last_wall = time.monotonic()
        self._last_proc = self._proc_cpu_time()
        self._last_stat = self._read_stat()
        self._net0 = self._net_bytes()
        self._gpu_ok = True
        self._nvml = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception:
            self._nvml = None
            self._gpu_ok = shutil.which("nvidia-smi") is not None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._proc = psutil.Process(self._pid)
            self._proc.cpu_percent(None)
            psutil.cpu_percent(None)
            net = psutil.net_io_counters()
            self._net0 = (net.bytes_sent, net.bytes_recv)
        except Exception:
            self._psutil = None
            self._proc = None

    @staticmethod
    def _proc_cpu_time() -> float:
        t = os.times()
        return t.user + t.system

    @staticmethod
    def _read_stat() -> tuple[int, int] | None:
        try:
            with open("/proc/stat") as fh:
                parts = [int(x) for x in fh.readline().split()[1:]]
            idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
            return sum(parts), idle
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def _net_bytes() -> tuple[int, int] | None:
        try:
            sent = recv = 0
            with open("/proc/net/dev") as fh:
                for line in fh.readlines()[2:]:
                    iface, _, rest = line.partition(":")
                    if iface.strip() == "lo":
                        continue
                    cols = rest.split()
                    recv += int(cols[0])
                    sent += int(cols[8])
            return sent, recv
        except (OSError, ValueError, IndexError):
            return None

    def sample(self) -> dict:
        m: dict[str, Any] = {}
        now = time.monotonic()
        if self._psutil is not None:
            ps = self._psutil
            try:
                m["cpu"] = self._proc.cpu_percent(None) / self._ncpu
                m["cpu.system"] = ps.cpu_percent(None)
                vm = ps.virtual_memory()
                m["memory_percent"] = vm.percent
                m["proc.memory.availableMB"] = vm.available / 1048576
                mi = self._proc.memory_info()
                m["proc.memory.rssMB"] = mi.rss / 1048576
                m["proc.memory.percent"] = self._proc.memory_percent()
                m["proc.cpu.threads"] = self._proc.num_threads()
                net = ps.net_io_counters()
                if self._net0 is None:
                    self._net0 = (net.bytes_sent, net.bytes_recv)
                m["network.sent"] = net.bytes_sent - self._net0[0]
                m["network.recv"] = net.bytes_recv - self._net0[1]
            except Exception:
                pass
        else:
            proc = self._proc_cpu_time()
            wall = now - self._last_wall
            if wall > 0:
                m["cpu"] = 100.0 * (proc - self._last_proc) / wall / self._ncpu
            self._last_proc = proc
            stat = self._read_stat()
            if stat and self._last_stat:
                total, idle = stat[0] - self._last_stat[0], stat[1] - self._last_stat[1]
                if total > 0:
                    m["cpu.system"] = 100.0 * (1 - idle / total)
            self._last_stat = stat
            mem = _meminfo()
            if mem.get("MemTotal"):
                avail = mem.get("MemAvailable", mem.get("MemFree", 0))
                m["memory_percent"] = 100.0 * (1 - avail / mem["MemTotal"])
                m["proc.memory.availableMB"] = avail / 1048576
            try:
                with open(f"/proc/{self._pid}/status") as fh:
                    for line in fh:
                        if line.startswith("VmRSS:"):
                            rss = int(line.split()[1]) * 1024
                            m["proc.memory.rssMB"] = rss / 1048576
                            if mem.get("MemTotal"):
                                m["proc.memory.percent"] = 100.0 * rss / mem["MemTotal"]
                        elif line.startswith("Threads:"):
                            m["proc.cpu.threads"] = int(line.split()[1])
            except (OSError, ValueError):
                pass
            net = self._net_bytes()
            if net and self._net0:
                m["network.sent"] = net[0] - self._net0[0]
                m["network.recv"] = net[1] - self._net0[1]
        self._last_wall = now
        try:
            du = shutil.disk_usage("/")
            m["disk./.usagePercent"] = 100.0 * du.used / du.total
            m["disk./.usageGB"] = du.used / 1073741824
        except OSError:
            pass
        m.update(self._gpu())
        return m

    def _gpu(self) -> dict:
        if not self._gpu_ok:
            return {}
        out: dict[str, Any] = {}
        if self._nvml is not None:
            nv = self._nvml
            try:
                for i in range(nv.nvmlDeviceGetCount()):
                    h = nv.nvmlDeviceGetHandleByIndex(i)
                    util = nv.nvmlDeviceGetUtilizationRates(h)
                    mem = nv.nvmlDeviceGetMemoryInfo(h)
                    out[f"gpu.{i}.gpu"] = util.gpu
                    out[f"gpu.{i}.memory"] = util.memory
                    out[f"gpu.{i}.memoryAllocated"] = 100.0 * mem.used / mem.total
                    out[f"gpu.{i}.memoryAllocatedBytes"] = mem.used
                    try:
                        out[f"gpu.{i}.temp"] = nv.nvmlDeviceGetTemperature(h, 0)
                        power = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
                        limit = nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                        out[f"gpu.{i}.powerWatts"] = power
                        if limit:
                            out[f"gpu.{i}.powerPercent"] = 100.0 * power / limit
                    except Exception:
                        pass
                return out
            except Exception:
                self._nvml = None
        text = _run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
        )
        if text is None:
            self._gpu_ok = False
            return out

        def num(s: str) -> float | None:
            try:
                return float(s)
            except ValueError:
                return None

        for line in text.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 8:
                continue
            i = p[0]
            vals = [num(x) for x in p[1:]]
            util, mutil, used, total, temp, power, limit = vals
            if util is not None:
                out[f"gpu.{i}.gpu"] = util
            if mutil is not None:
                out[f"gpu.{i}.memory"] = mutil
            if used is not None and total:
                out[f"gpu.{i}.memoryAllocated"] = 100.0 * used / total
                out[f"gpu.{i}.memoryAllocatedBytes"] = used * 1048576
            if temp is not None:
                out[f"gpu.{i}.temp"] = temp
            if power is not None:
                out[f"gpu.{i}.powerWatts"] = power
                if limit:
                    out[f"gpu.{i}.powerPercent"] = 100.0 * power / limit
        return out
