"""In-process NCCL communicator revoke for Corex (ncclCommRevoke).

Corex 4.5.0 corrupts the device VA space (XID 24) when a process is torn down
with NCCL collectives still in flight. SIGTERM, SIGKILL and implicit
``ncclCommAbort`` in ProcessGroup destructors all take that path. NCCL 2.28.7+
has a gentler primitive: ``ncclCommRevoke`` (ixcclCommRevoke in Corex) is a
LOCAL operation (not a collective) that stops every in-flight operation on the
communicator and moves it to ``ncclInProgress``; once quiescent the comm can be
destroyed safely.

PyTorch here has no revoke binding, but ``ProcessGroupNCCL._comm_ptr`` exposes
the raw ``ncclComm_t`` for the current device, so we can call the C API with
ctypes. The library is already loaded by torch (LD_PRELOAD), so this does not
load a second copy.

Typical use as a stuck-job kill switch, without sending any signal::

    from nccl_revoke import CommWatchdog
    wd = CommWatchdog(timeout_s=120)
    wd.start()          # 每秒检查心跳和停止文件
    ...
    wd.touch()          # 训练循环每步调用

If no ``touch()`` for ``timeout_s``, or ``$UDTCA_REVOKE_FILE`` appears, the
watchdog revokes and destroys every NCCL comm of this rank and flushes logs.
Shutdown is coordinated, not signalled: the rank that revokes first writes a
notice file, every rank of the node revokes its own comms and records that it
is quiescent, and only once all of them are quiescent does each process call
``os._exit``. Exiting while a peer still has collectives in flight is what
makes torchrun SIGTERM that peer mid-op and wedges the GPU (XID 24); the
barrier removes that window. Nothing ever sends a signal to another process.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import tempfile
import threading
import time

_NCCL_SUCCESS = 0
# ncclResult_t: ... ncclInProgress = 7
_NCCL_IN_PROGRESS = 7
_NCCL_REVOKE_DEFAULT = 0


def _log(msg: str) -> None:
    print(f"[nccl-revoke][pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def _load_nccl() -> ctypes.CDLL:
    """Load the nccl library torch already uses and bind a cancel primitive.

    The bare soname comes first so the loader returns the *already loaded*
    instance, which owns our ProcessGroups' ``ncclComm_t`` pointers; a second
    copy from another Corex install would get the wrong pointers. Corex 4.4
    exports only ``ncclCommAbort`` (same local cancel, no quiescence handle),
    Corex 4.5 exports both; prefer revoke.
    """
    candidates = ["libnccl.so.2", "libnccl.so"]
    preload = os.environ.get("LD_PRELOAD", "")
    candidates += [p for p in preload.split(":") if "libnccl" in p]
    for env in ("COREX_PATH", "COREX_TOOLKIT"):
        corex = os.environ.get(env)
        if not corex:
            continue
        candidates.append(os.path.join(corex, "lib64", "libnccl.so.2"))
        candidates.append(os.path.join(corex, "lib", "libnccl.so.2"))
        candidates.append(os.path.join(corex, "ixccl", "lib64", "libnccl.so.2"))
    found = ctypes.util.find_library("nccl")
    if found:
        candidates.append(found)
    candidates.append("/usr/local/corex/lib64/libnccl.so.2")

    last_err = None
    for path in candidates:
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path)
        except OSError as exc:
            last_err = exc
            continue
        bound = None
        for name, argtypes in (
            ("ncclCommRevoke", [ctypes.c_void_p, ctypes.c_int]),
            ("ncclCommAbort", [ctypes.c_void_p]),
        ):
            fn = getattr(lib, name, None)
            if fn is None:
                continue
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
            bound = bound or name
        if bound is None:
            last_err = AttributeError(
                f"{path} exports neither ncclCommRevoke nor ncclCommAbort"
            )
            continue
        try:
            lib.ncclCommGetAsyncError.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.ncclCommGetAsyncError.restype = ctypes.c_int
        except AttributeError:
            # Without it we cannot tell when the cancel took effect; try the
            # next candidate instead of failing outright.
            last_err = AttributeError(f"{path} exports no ncclCommGetAsyncError")
            continue
        try:
            lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
            lib.ncclCommDestroy.restype = ctypes.c_int
        except AttributeError:
            lib.ncclCommDestroy = None  # optional: only used with destroy=True
        lib._udtca_revoke_symbol = bound  # type: ignore[attr-defined]
        _log(f"loaded {path} (cancel primitive: {bound})")
        return lib
    raise RuntimeError(f"cannot load an nccl library with revoke support: {last_err}")


_lib = None
_lib_lock = threading.Lock()


def _nccl_lib() -> ctypes.CDLL:
    global _lib
    with _lib_lock:
        if _lib is None:
            _lib = _load_nccl()
        return _lib


def _nccl_comms(device_index: int | None = None):
    """Return [(name, comm_ptr, revoker)] for every PG this rank owns.

    ``revoker`` is ``None`` for ordinary NCCL groups. bitscom's
    ``ProcessGroupLowBit`` keeps its own raw ``ncclComm_t`` that torch's backend
    enumeration cannot see, so for it we capture the backend's bound
    ``revoke_lowbit_comm`` method and call that instead.


    ``_comm_ptr`` returns the communicator of one device; our ranks are
    one-process-one-device, so one pointer per group is correct here.
    ``device_index`` must be captured on the main thread: the CUDA "current
    device" is thread-local, and a watchdog thread otherwise sees device 0.
    """
    import torch
    from torch._C import _distributed_c10d as _c10d
    from torch.distributed import distributed_c10d as dc

    if not dc.is_initialized():
        return []
    # torch 2.7 keeps the registry on a _World object, newer builds also expose
    # the flat _pg_map; probe both so the same file works on 2.7.1 and 2.10.0.
    registry = getattr(dc, "_pg_map", None)
    if isinstance(registry, dict) and registry:
        pg_map = list(registry.keys())
    else:
        try:
            pg_map = list(dc._world.pg_map.keys())
        except Exception:
            pg_map = [dc.group.WORLD]

    if device_index is None:
        device_index = torch.cuda.current_device()
    # _comm_ptr() internally looks up the thread-local current device, so the
    # watchdog thread must select the same device as the main thread.
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    comms = []
    _last_revoke_summary["groups"] = len(pg_map)
    for pg in pg_map:
        # torch 2.10 wraps ProcessGroup in Python; the NCCL backend object
        # (the one exposing _comm_ptr) is reached through _get_backend.
        try:
            backend = pg._get_backend(device)
        except Exception as exc:
            _log(f"skip {type(pg).__name__}: {exc!r}")
            continue
        if not isinstance(backend, _c10d.ProcessGroupNCCL):
            # bitscom's lowbit backend owns a private ncclComm_t; ask it to
            # revoke in-process instead of skipping it.
            revoker = getattr(backend, "revoke_lowbit_comm", None)
            if callable(revoker):
                ranks = dc.get_process_group_ranks(pg)
                comms.append(
                    (f"rank{dc.get_rank()}-lowbit-group{ranks}", None, revoker)
                )
                _log(f"captured lowbit backend for ranks {ranks}")
                continue
            try:
                backend_name = pg._get_backend_name()
            except Exception:
                backend_name = "?"
            _log(
                f"skip non-NCCL backend {type(backend).__name__} "
                f"for ranks {dc.get_process_group_ranks(pg)} "
                f"(backend_name={backend_name})"
            )
            continue
        ranks = dc.get_process_group_ranks(pg)
        name = f"rank{dc.get_rank()}-group{ranks}"
        # Preferred: the raw ncclComm_t, so we can call ncclCommRevoke and poll
        # for quiescence. Only some torch builds expose it.
        ptr = None
        try:
            ptr = backend._comm_ptr()
        except Exception as exc:
            ptr_err = exc
        else:
            ptr_err = None
        if ptr and int(ptr) > 0:
            comms.append((name, int(ptr), None))
            continue
        # Fallback: ProcessGroupNCCL.abort() uses the comm handle internally.
        # It is the same local cancel primitive, without a quiescence handle,
        # so callers sleep briefly instead of polling GetAsyncError.
        aborter = getattr(backend, "abort", None)
        if callable(aborter):
            comms.append((name, None, lambda f=aborter: (f() or 0)))
            _log(
                f"captured {name} via ProcessGroupNCCL.abort "
                f"(ptr unavailable: {ptr_err!r})"
            )
            continue
        _log(f"skip {type(backend).__name__}: no _comm_ptr and no abort() ({ptr_err!r})")
    if not comms:
        _log(f"no NCCL backend found among {len(pg_map)} groups on device {device_index}")
    return comms


# Set by revoke_all so coordinated_shutdown can tell "revoked everything" from
# "there was nothing to revoke" and "revoke is not available on this host".
_last_revoke_summary = {"available": False, "groups": 0, "comms": 0, "revoked": 0}
# Wall-clock start of this process: used to reject stale revoke notices.
_PROCESS_START = time.time()


def revoke_summary() -> dict:
    return dict(_last_revoke_summary)


def revoke_all(
    quiesce_timeout_s: float = 30.0,
    destroy: bool = True,
    device_index: int | None = None,
    comms: list | None = None,
) -> list:
    """Revoke every NCCL comm of this rank; destroy them once quiescent.

    ``comms`` may be a snapshot taken earlier on the main thread (see
    ``CommWatchdog.capture``); enumerating them from the watchdog thread while
    a collective is stuck can block on ProcessGroup locks, so the cached list
    is the reliable path.

    Returns a list of (name, ptr, revoke_rc, quiesce_ok, destroy_rc) for logging.
    Never raises: this runs on the failure path.
    """
    _last_revoke_summary.update(available=False, groups=0, comms=0, revoked=0)

    if comms is None:
        comms = _nccl_comms(device_index)
    if not comms:
        _log("no NCCL communicators to revoke")
        return []
    _last_revoke_summary["comms"] = len(comms)

    # The ctypes path needs the library; the abort()-based entries do not.
    lib = None
    if any(entry[1] for entry in comms):
        try:
            lib = _nccl_lib()
            _last_revoke_summary["available"] = True
        except Exception as exc:
            _log(f"revoke-by-pointer unavailable: {exc!r}")

    # [name, ptr, revoke_rc, quiesced, destroy_rc]
    results = []
    for entry in comms:
        name, ptr = entry[0], entry[1]
        revoker = entry[2] if len(entry) > 2 else None
        if revoker is not None:
            # LowBit backend: revoke its private ncclComm_t in-process. There is
            # no torch Work/quiescence handle for it, so only log the rc.
            try:
                rc = int(revoker())
            except Exception as exc:
                rc = -1
                _log(f"lowbit revoke {name} failed: {exc!r}")
            _log(f"revoke {name} via backend abort rc={rc}")
            if rc == _NCCL_SUCCESS:
                _last_revoke_summary["revoked"] += 1
            continue
        if lib is None:
            _log(f"skip {name}: no revoke library for a raw comm pointer")
            continue
        if getattr(lib, "_udtca_revoke_symbol", "ncclCommRevoke") == "ncclCommAbort":
            rc = lib.ncclCommAbort(ctypes.c_void_p(ptr))
        else:
            rc = lib.ncclCommRevoke(ctypes.c_void_p(ptr), _NCCL_REVOKE_DEFAULT)
        _log(f"revoke {name} ptr=0x{ptr:x} rc={rc}")
        if rc == _NCCL_SUCCESS:
            _last_revoke_summary["revoked"] += 1
        results.append([name, ptr, rc, False, None])

    if lib is None:
        # abort()-based entries have no quiescence handle; give the driver a
        # moment to cancel the in-flight kernels before the caller exits.
        time.sleep(min(1.0, max(0.1, quiesce_timeout_s / 4)))
        return []

    deadline = time.monotonic() + quiesce_timeout_s
    pending = [r for r in results if r[2] == _NCCL_SUCCESS]
    while pending and time.monotonic() < deadline:
        still = []
        for r in pending:
            state = ctypes.c_int(-1)
            erc = lib.ncclCommGetAsyncError(
                ctypes.c_void_p(r[1]), ctypes.byref(state)
            )
            if erc == _NCCL_SUCCESS and state.value != _NCCL_IN_PROGRESS:
                r[3] = True
            elif erc != _NCCL_SUCCESS:
                _log(f"async-error query {r[0]} failed rc={erc}")
            else:
                still.append(r)
        pending = still
        if pending:
            time.sleep(0.05)

    for r in results:
        if r[3] and destroy and lib.ncclCommDestroy is not None:
            r[4] = lib.ncclCommDestroy(ctypes.c_void_p(r[1]))
            _log(f"destroy {r[0]} rc={r[4]}")
        elif not r[3]:
            _log(f"{r[0]} did not quiesce within {quiesce_timeout_s}s; not destroying")
    return [tuple(r) for r in results]


def _run_token() -> str:
    """Stable within one torchrun job, unique across jobs.

    torch 2.7 leaves TORCHELASTIC_RUN_ID as the literal "none" without
    --rdzv-id, which made every job share /tmp/udtca_revoke_none and a stale
    notice killed the next run at step 0. Reject those values; the elastic
    agent's pid is the per-run fallback all local ranks agree on.
    """
    for key in ("TORCHELASTIC_RUN_ID", "MASTER_PORT"):
        value = (os.environ.get(key) or "").strip()
        if value and value.lower() not in {"none", "0"}:
            return value
    return f"ppid{os.getppid()}"


def _rendezvous_dir() -> str:
    """Per-run directory used for the shutdown handshake."""
    root = os.environ.get("UDTCA_REVOKE_DIR", "")
    if root:
        return root
    return os.path.join(tempfile.gettempdir(), f"udtca_revoke_{_run_token()}")


def _notice_path() -> str:
    return os.path.join(_rendezvous_dir(), "revoke.notice")


def _local_rank_world() -> tuple[int, int]:
    """Rank and local-world-size; the handshake is per node (per /tmp).

    Falls back to the global values when torchrun's LOCAL_* are absent, which
    is also the single-process case.
    """
    world = int(os.environ.get("WORLD_SIZE", "1") or 1)
    rank = int(os.environ.get("RANK", "-1") or -1)
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            world = int(dist.get_world_size())
            rank = int(dist.get_rank())
    except Exception:
        pass
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)) or world)
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)) or rank)
    return local_rank, max(1, local_world)


def _write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def revoke_requested() -> bool:
    """True once any rank of this node has started the shutdown handshake.

    Notices older than this process are stale: the rendezvous directory is
    per-run, but a reused port or a leftover /tmp file must never make a fresh
    run decide to exit at step 0.
    """
    try:
        path = _notice_path()
        if not os.path.exists(path):
            return False
        return os.stat(path).st_mtime >= _PROCESS_START - 30.0
    except Exception:
        return False


def request_revoke(reason: str) -> None:
    """Tell the other ranks that this job is going down, before we exit.

    Best effort and side-effect free on failure: if the file cannot be
    written the peers still find out through their own timeout.
    """
    try:
        os.makedirs(_rendezvous_dir(), exist_ok=True)
        if not os.path.exists(_notice_path()):
            _write_atomic(_notice_path(), f"{reason}\npid={os.getpid()}\n")
    except Exception as exc:
        _log(f"cannot write revoke notice: {exc!r}")


def _flush_logs() -> None:
    try:
        import logging

        for handler in logging.getLogger().handlers + logging.getLogger("psgd").handlers:
            try:
                handler.flush()
            except Exception:
                pass
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass


# A shutdown can be started from several threads (watchdog, op wait, signal
# handler). The first one runs it; the others wait so they cannot race it. If
# the owner has to abort (revoke unavailable), the flag clears and the waiters
# return so their caller can decide what to do.
_exit_lock = threading.Lock()
_exit_started = False
_shutdown_active = threading.Event()


def coordinated_shutdown(
    reason: str,
    exit_code: int = 3,
    device_index: int | None = None,
    quiesce_timeout_s: float = 10.0,
    barrier_timeout_s: float | None = None,
    comms: list | None = None,
) -> None:
    """Revoke this rank's NCCL comms, wait for the node to be quiescent, exit.

    1. write the notice file (peers revoke immediately instead of waiting for
       their own timeout),
    2. ``ncclCommRevoke`` every comm of this rank and wait for ncclInProgress
       to clear,
    3. write this rank's ``ready`` marker,
    4. wait until every local rank is ready, then ``os._exit``.

    Step 3-4 is the "rollback" part: no process is torn down while a peer can
    still have a collective in flight. If a peer never reports (crashed, or
    the node is wedged) we exit after the barrier timeout anyway; our own
    comms are already quiescent, so that exit cannot cause XID 24.
    """
    global _exit_started
    with _exit_lock:
        first = not _shutdown_active.is_set()
        if first:
            _shutdown_active.set()
            _exit_started = True
    if not first:
        # The owner exits the process on success; if it aborts, wake up.
        while _shutdown_active.is_set():
            time.sleep(0.2)
        return

    if barrier_timeout_s is None:
        barrier_timeout_s = float(os.environ.get("UDTCA_REVOKE_BARRIER_S", "5"))
    rank, world = _local_rank_world()
    _log(
        f"coordinated shutdown rank={rank}/{world} code={exit_code}: {reason}"
    )
    request_revoke(reason)

    try:
        revoke_all(
            quiesce_timeout_s=quiesce_timeout_s,
            destroy=False,  # os._exit skips destructors; don't risk a double free
            device_index=device_index,
            comms=comms,
        )
    except Exception as exc:
        _log(f"revoke failed: {exc!r}")

    summary = revoke_summary()
    if (
        (summary["comms"] > 0 or summary["groups"] > 0)
        and summary["revoked"] == 0
        and os.environ.get("UDTCA_REVOKE_DEGRADED_EXIT", "0") != "1"
    ):
        # Nothing was cancelled: exiting now would tear down the CUDA context
        # with collectives still in flight -- exactly the XID 24 path this
        # module exists to avoid. Stay alive (torchrun will kill us if it must)
        # and let the caller retry.
        _log(
            "ERROR: no communicator could be cancelled "
            f"(lib available={summary['available']}, groups={summary['groups']}, "
            f"comms={summary['comms']}); "
            "refusing to exit. Set UDTCA_REVOKE_DEGRADED_EXIT=1 to exit anyway."
        )
        _flush_logs()
        with _exit_lock:
            _exit_started = False
            _shutdown_active.clear()
        return

    try:
        os.makedirs(_rendezvous_dir(), exist_ok=True)
        _write_atomic(
            os.path.join(_rendezvous_dir(), f"rank-{rank}.ready"), str(os.getpid())
        )
    except Exception as exc:
        _log(f"cannot write ready marker: {exc!r}")
        # Cannot synchronise with peers: do not exit underneath them.
        _flush_logs()
        with _exit_lock:
            _exit_started = False
            _shutdown_active.clear()
        return

    deadline = time.monotonic() + barrier_timeout_s
    while time.monotonic() < deadline:
        try:
            ready = sum(
                os.path.exists(os.path.join(_rendezvous_dir(), f"rank-{r}.ready"))
                for r in range(world)
            )
        except Exception:
            break
        if ready >= world:
            _log(f"all {world} local ranks quiescent")
            break
        time.sleep(0.2)
    else:
        _log(f"barrier timeout after {barrier_timeout_s:.0f}s; exiting anyway")

    _flush_logs()
    os._exit(exit_code)


def exit_now(exit_code: int) -> None:
    """Exit this process, joining an in-progress rollback first.

    A revoked collective returns an error, so the main thread can reach a
    normal exit path while other ranks are still revoking. Every ordinary exit
    goes through here so that it cannot tear down underneath them.
    """
    if _exit_started or revoke_requested():
        # Parks forever when another thread owns the shutdown; it ends with
        # os._exit once every local rank is quiescent.
        coordinated_shutdown(
            "process exit requested after a revoke", exit_code=exit_code
        )
        # It only returns when the rollback was refused (nothing cancelled):
        # exiting here would tear down with collectives in flight, so stay
        # alive exactly like coordinated_shutdown does.
        summary = revoke_summary()
        if (
            (summary["comms"] > 0 or summary["groups"] > 0)
            and summary["revoked"] == 0
            and os.environ.get("UDTCA_REVOKE_DEGRADED_EXIT", "0") != "1"
        ):
            _log("exit suppressed: rollback degraded, staying alive")
            _flush_logs()
            while True:
                time.sleep(60)
    _flush_logs()
    os._exit(exit_code)


def revoke_and_exit(
    reason: str,
    exit_code: int = 124,
    device_index: int | None = None,
    quiesce_timeout_s: float = 10.0,
    grace_s: float = 5.0,
    comms: list | None = None,
) -> None:
    """Backwards-compatible name for ``coordinated_shutdown``.

    ``grace_s`` is kept for call-site compatibility; the wait for peers is
    controlled by ``UDTCA_REVOKE_BARRIER_S``.
    """
    coordinated_shutdown(
        reason,
        exit_code=exit_code,
        device_index=device_index,
        quiesce_timeout_s=quiesce_timeout_s,
        comms=comms,
    )


class CommWatchdog(threading.Thread):
    """Daemon thread: revoke-and-exit on missing heartbeat or stop file."""

    def __init__(
        self,
        timeout_s: float = 120.0,
        poll_s: float = 1.0,
        stop_file: str | None = None,
        exit_code: int = 124,
        quiesce_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(name="udtca-comm-watchdog", daemon=True)
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.stop_file = stop_file or os.environ.get("UDTCA_REVOKE_FILE", "")
        self.exit_code = exit_code
        self.quiesce_timeout_s = quiesce_timeout_s
        self._last = time.monotonic()
        self._last_capture = 0.0
        self._capture_interval_s = 5.0
        self._comms: list = []
        self._comms_lock = threading.Lock()
        self._stop = threading.Event()
        # 必须在本线程（主线程）记录设备号：CUDA current device 是线程局部的
        try:
            import torch

            self.device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        except Exception:
            self.device_index = 0

    def capture(self, force: bool = False) -> None:
        """Snapshot comm pointers. Must run on the thread that owns the PG."""
        now = time.monotonic()
        if not force and now - self._last_capture < self._capture_interval_s:
            return
        self._last_capture = now
        try:
            comms = _nccl_comms(self.device_index)
        except Exception as exc:
            _log(f"comm capture failed: {exc!r}")
            return
        with self._comms_lock:
            self._comms = comms

    def _comm_snapshot(self) -> list:
        with self._comms_lock:
            return list(self._comms)

    def touch(self) -> None:
        self._last = time.monotonic()
        self.capture(force=False)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        _log(
            f"watchdog started timeout={self.timeout_s}s device={self.device_index} "
            f"stop_file={self.stop_file or '-'}"
        )
        while not self._stop.wait(self.poll_s):
            if revoke_requested():
                coordinated_shutdown(
                    "another rank requested revoke",
                    self.exit_code,
                    self.device_index,
                    self.quiesce_timeout_s,
                    comms=self._comm_snapshot(),
                )
                self._stop.set()
                return
            if self.stop_file and os.path.exists(self.stop_file):
                coordinated_shutdown(
                    f"stop file {self.stop_file}",
                    self.exit_code,
                    self.device_index,
                    self.quiesce_timeout_s,
                    comms=self._comm_snapshot(),
                )
                self._stop.set()
                return
            idle = time.monotonic() - self._last
            if idle > self.timeout_s:
                coordinated_shutdown(
                    f"no progress for {idle:.1f}s",
                    self.exit_code,
                    self.device_index,
                    self.quiesce_timeout_s,
                    comms=self._comm_snapshot(),
                )
                self._stop.set()
                return
