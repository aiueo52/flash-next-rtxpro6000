#!/usr/bin/env python3
"""Minimal HumanEval executor: runs prompt+completion+test inside bwrap (no network, tmpfs) with a timeout.

The generated program is untrusted. It runs with a cleared environment (bwrap --clearenv, and the bwrap
process itself gets only PATH), so it cannot read the caller's environment variables (tokens, keys, ...).
Only the exception class of a failure is returned, never raw stderr, so nothing the program prints can
end up in the result files. stdout is discarded, at most STDERR_KEEP bytes of stderr are read back, and
each job runs in a transient systemd user scope (cgroup: MemoryMax, MemorySwapMax, TasksMax, CPUQuota for
the whole job) plus per-process rlimits (address space, heap, CPU time, file size, open files, processes)
and a wall-clock timeout that kills its whole process group. The root and /dev are read-only, /tmp and
/dev/shm are small size-capped tmpfs mounts, and nested user namespaces are disabled.

The per-job cgroup is REQUIRED: if `systemd-run --user --scope` is unavailable, run_program raises
SandboxUnavailable and executes nothing. HE_EXEC_ALLOW_NO_CGROUP=1 (or --allow-no-cgroup for the
self-test) is an UNSAFE opt-out that runs with per-process rlimits only, i.e. without any limit on the
whole job's memory. If the launcher (systemd-run, prlimit or bwrap) fails for a job, run_program raises
SandboxInfraError instead of reporting a pass or fail for the program.

This is best-effort isolation for a single-user workstation, not a security boundary: it still shares
the host kernel. Run the quality audit (which executes model-generated code) only in a disposable VM or
container. `python3 he_exec.py [--self-test] [--allow-no-cgroup]` runs a self-test."""
import os, re, subprocess, sys, tempfile, shutil, threading
PY = shutil.which("python3") or sys.executable
# The only variables the sandboxed interpreter sees.
SANDBOX_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
_EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning))\b")
class SandboxUnavailable(RuntimeError):
    """A required isolation layer (the per-job cgroup, prlimit) is missing; nothing was executed."""
class SandboxInfraError(RuntimeError):
    """The launcher (systemd-run, prlimit, bwrap) failed for this job; the program's result is unknown."""
def error_class(stderr):
    """Last traceback line reduced to its exception class name (e.g. 'AssertionError'), else 'error'."""
    for line in reversed((stderr or "").strip().splitlines()):
        m = _EXC.match(line.strip())
        if m: return m.group(1)[:80]
    return "error" if stderr else ""
def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text  # last block: the model may show tests first
def build_program(prompt, completion, test, entry_point):
    code = extract_code(completion)
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.M):
        body = code
        # keep the prompt's imports/helpers in scope (the model may drop them)
        body = prompt.split(f"def {entry_point}")[0] + "\n" + body
    else:
        body = prompt + code
    return body + "\n\n" + test + f"\n\ncheck({entry_point})\n"
# Per-job resource limits, applied (inherited through bwrap) to the sandboxed interpreter.
LIMITS = {
    "RLIMIT_AS": 2 << 30,        # 2 GiB address space
    "RLIMIT_DATA": 2 << 30,      # 2 GiB heap
    "RLIMIT_CPU": 20,            # CPU seconds (the wall-clock timeout below is the primary bound)
    "RLIMIT_FSIZE": 16 << 20,    # 16 MiB per written file (also caps the captured stderr)
    "RLIMIT_NOFILE": 64,         # open file descriptors
    "RLIMIT_CORE": 0,
}
# RLIMIT_NPROC is counted per user namespace, so it is set inside the sandbox (in the new namespace,
# before the program starts); set on the bwrap parent it would count all of the caller's processes.
NPROC_LIMIT = 64
# Markers written to stderr by the launcher stages before the untrusted program starts. A job whose
# stderr does not begin with them never reached the program: that is a launcher failure, not a result.
_SCOPE_MARK = b"he_exec:scope\n"   # written inside the systemd scope (the cgroup exists)
_BOOT_MARK = b"he_exec:boot\n"     # written inside bwrap after the NPROC limit is set
# Completion proof. A pass needs exit status 0 AND a per-job random nonce written back on a private pipe
# after prog.py (prompt + completion + the appended HumanEval tests, ending in check(...)) has run to its
# end. The boot code reads the nonce from an inherited pipe and closes it before the program starts, and
# writes it to a second (close-on-exec) pipe only once runpy returns normally. So a candidate that ends
# the process early -- sys.exit()/SystemExit(0)/exit()/quit() at module level or inside the function,
# os._exit(0), a signal -- or that closes or redirects stdout/stderr, cannot pass without the tests having
# completed. (Deliberately hostile code in the same interpreter could still dig the nonce out of memory;
# the sandbox, not the nonce, is the defence against that.)
def _boot_code():
    return ("import os, sys, resource, runpy\n"
            "def _he_exec_boot(rfd, wfd):\n"
            "    nonce = os.read(rfd, 64); os.close(rfd); os.set_inheritable(wfd, False)\n"
            "    resource.setrlimit(resource.RLIMIT_NPROC, (%d, %d))\n"
            "    sys.argv = ['prog.py']\n"
            "    os.write(2, %r)\n"
            "    runpy.run_path('prog.py', run_name='__main__')\n"
            "    os.write(wfd, nonce)\n"
            "_he_exec_boot(int(sys.argv[1]), int(sys.argv[2]))\n" % (NPROC_LIMIT, NPROC_LIMIT, _BOOT_MARK))
_BOOT = _boot_code()
EARLY_EXIT = "IncompleteRun"     # error class for exit status 0 without the completion nonce
TMPFS_BYTES = 32 << 20          # size of each writable tmpfs (/tmp, /dev/shm)
STDERR_KEEP = 512                # bytes of stderr read back (only the exception class is returned)
_PRLIMIT_OPT = {"RLIMIT_AS": "--as", "RLIMIT_DATA": "--data", "RLIMIT_CPU": "--cpu", "RLIMIT_FSIZE": "--fsize",
                "RLIMIT_NOFILE": "--nofile", "RLIMIT_CORE": "--core"}
# Per-job cgroup limits (memory, pids, CPU), enforced by a transient systemd user scope around each
# job, so they cover the whole job rather than each process. Needs a systemd user manager with the
# memory/pids/cpu controllers delegated (the default on current desktop distros). Without it nothing
# runs (SandboxUnavailable) unless HE_EXEC_ALLOW_NO_CGROUP=1, which is unsafe: per-process rlimits only.
# HE_EXEC_SYSTEMD_RUN overrides the systemd-run path (default: from PATH).
CGROUP = {"MemoryMax": "1G", "MemorySwapMax": "0", "TasksMax": "64", "CPUQuota": "100%"}
_probe = {}                      # systemd-run path -> scope creation works
_probe_lock = threading.Lock()
_warned = set()
def _allow_no_cgroup():
    return os.environ.get("HE_EXEC_ALLOW_NO_CGROUP") == "1"
def _systemd_run():
    exe = os.environ.get("HE_EXEC_SYSTEMD_RUN")
    return exe if exe is not None else (shutil.which("systemd-run") or "")
def _scope_base(exe):
    base = [exe, "--user", "--scope", "--quiet", "--collect"]
    for k, v in CGROUP.items(): base += ["-p", f"{k}={v}"]
    return base + ["--"]
def _cgroup_ok(exe):
    with _probe_lock:
        if exe not in _probe:
            try:
                ok = bool(exe) and subprocess.run(_scope_base(exe) + ["true"], stdin=subprocess.DEVNULL,
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                  env=_launcher_env(), timeout=60).returncode == 0
            except (OSError, subprocess.SubprocessError):
                ok = False
            _probe[exe] = ok
        return _probe[exe]
def _cgroup_prefix():
    """systemd-run scope prefix for one job; raises SandboxUnavailable when the scope cannot be created
    and the unsafe opt-out is not set. Returns [] only under HE_EXEC_ALLOW_NO_CGROUP=1."""
    exe = _systemd_run()
    if _cgroup_ok(exe):
        # the shell runs inside the new scope and marks that before exec'ing the rest of the job
        return _scope_base(exe) + ["/bin/sh", "-c", "printf '%s' \"$0\" >&2; exec \"$@\"", _SCOPE_MARK.decode()]
    msg = ("he_exec: no per-job cgroup (`systemd-run --user --scope` via %r is unavailable or failed)"
           % (exe or "systemd-run"))
    if not _allow_no_cgroup():
        raise SandboxUnavailable(msg + "; refusing to execute untrusted code without a whole-job memory limit. "
                                 "HE_EXEC_ALLOW_NO_CGROUP=1 overrides this (UNSAFE: per-process rlimits only).")
    if exe not in _warned:
        _warned.add(exe)
        print("WARNING: " + msg + "; HE_EXEC_ALLOW_NO_CGROUP=1 is set, running UNSAFELY with per-process "
              "rlimits only (no whole-job memory limit)", file=sys.stderr)
    return []
def _launcher_env():
    # systemd-run needs the user-bus address; bwrap --clearenv then hides it from the program.
    env = {"PATH": SANDBOX_ENV["PATH"]}
    for k in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        if k in os.environ: env[k] = os.environ[k]
    return env
def _limit_prefix():
    # prlimit(1) instead of preexec_fn: run_bench.py calls run_program from several threads, and
    # preexec_fn is not safe with threads.
    exe = shutil.which("prlimit")
    if not exe: raise SandboxUnavailable("prlimit (util-linux) is required to run untrusted code with resource limits")
    return [exe] + [f"{_PRLIMIT_OPT[k]}={v}" for k, v in LIMITS.items()] + ["--"]
def _sandbox_argv(d):
    """bwrap command line for a job whose prog.py is in d (mounted read-only at /work)."""
    # Network, IPC, PID, UTS, cgroup and user namespaces unshared; no nested user namespaces.
    # Every writable place is size-capped: / and /dev are remounted read-only (non-recursively),
    # /tmp and /dev/shm are TMPFS_BYTES tmpfs mounts; the rest of /dev is device nodes and devpts.
    cmd = ["bwrap", "--unshare-all", "--unshare-user", "--disable-userns",
           "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
           "--ro-bind", "/bin", "/bin", "--ro-bind", "/etc/alternatives", "/etc/alternatives",
           "--size", str(TMPFS_BYTES), "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev",
           "--size", str(TMPFS_BYTES), "--tmpfs", "/dev/shm", "--remount-ro", "/dev",
           "--ro-bind", d, "/work", "--remount-ro", "/", "--chdir", "/work", "--clearenv"]
    for k, v in SANDBOX_ENV.items(): cmd += ["--setenv", k, v]
    return cmd
def run_program(src, timeout=10.0):
    """Run untrusted code once. Returns (status, exception class) with status "pass", "fail" or
    "timeout"; "pass" requires exit status 0 AND the completion nonce (the program ran to its end; see
    _boot_code), exit 0 without it is ("fail", "IncompleteRun"). stdout is discarded and stderr is read
    back only up to STDERR_KEEP bytes. The whole process group is killed on timeout. Raises SandboxUnavailable (nothing executed) when the per-job
    cgroup is unavailable and HE_EXEC_ALLOW_NO_CGROUP=1 is not set, and SandboxInfraError when the
    launcher failed before the program started (never reported as a pass or fail)."""
    import signal, secrets
    scope = _cgroup_prefix()
    prefix = scope + _limit_prefix()
    marks = (_SCOPE_MARK if scope else b"") + _BOOT_MARK
    d = tempfile.mkdtemp(prefix="he_"); done_r = None
    try:
        p = os.path.join(d, "prog.py")
        with open(p, "w") as f: f.write(src)
        cmd = prefix + _sandbox_argv(d)
        nonce = secrets.token_hex(16).encode()
        nonce_r, nonce_w = os.pipe()           # parent -> boot: the nonce (read and closed before prog.py)
        done_r, done_w = os.pipe()             # boot -> parent: the nonce, only after prog.py completed
        os.write(nonce_w, nonce); os.close(nonce_w)
        cmd += [PY, "-I", "-c", _BOOT, str(nonce_r), str(done_w)]
        with tempfile.TemporaryFile() as err:
            def launcher_failed(what):
                # Only launcher output precedes the boot marker, never the program's.
                err.seek(0); head = err.read(400).decode("utf-8", "replace").strip()
                return SandboxInfraError(f"he_exec: {what}; launcher stderr: {head!r}")
            def started():
                err.seek(0); return err.read(len(marks)) == marks
            def completed():
                os.set_blocking(done_r, False)
                try: return os.read(done_r, 2 * len(nonce)) == nonce
                except BlockingIOError: return False
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err,
                                        env=_launcher_env(), start_new_session=True, pass_fds=(nonce_r, done_w))
            except OSError as e:
                raise SandboxInfraError(f"he_exec: cannot start the launcher: {e}") from e
            finally:
                os.close(nonce_r); os.close(done_w)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()
                if not started(): raise launcher_failed(f"launcher did not start the program within {timeout}s")
                return ("timeout", "")
            if not started(): raise launcher_failed(f"launcher exited with status {rc} before the program started")
            size = err.seek(0, os.SEEK_END)
            err.seek(max(len(marks), size - STDERR_KEEP))
            tail = err.read(STDERR_KEEP).decode("utf-8", "replace")
            if rc != 0: return ("fail", error_class(tail))
            return ("pass", "") if completed() else ("fail", EARLY_EXIT)
    finally:
        if done_r is not None: os.close(done_r)
        shutil.rmtree(d, ignore_errors=True)
if __name__ == "__main__":
    import argparse, contextlib
    ap = argparse.ArgumentParser(description="Self-test of the HumanEval sandbox.")
    ap.add_argument("--self-test", action="store_true", help="run the self-test (the default action)")
    ap.add_argument("--allow-no-cgroup", action="store_true",
                    help="UNSAFE: run without the per-job cgroup if it is unavailable (same as HE_EXEC_ALLOW_NO_CGROUP=1)")
    args = ap.parse_args()
    if args.allow_no_cgroup: os.environ["HE_EXEC_ALLOW_NO_CGROUP"] = "1"
    try:
        have_cgroup = bool(_cgroup_prefix())
    except SandboxUnavailable as e:
        print(f"REFUSED: {e}", file=sys.stderr); sys.exit(2)
    def outcome(src, timeout=10.0):
        try: return run_program(src, timeout)
        except (SandboxUnavailable, SandboxInfraError) as e: return ("raised", type(e).__name__)
    def expect(label, got, want):
        ok = got[0] == want[0] and (want[1] is None or got[1] == want[1])
        print(f"{'ok  ' if ok else 'FAIL'} {label}: {got}")
        return ok
    @contextlib.contextmanager
    def env(**kv):
        old = {k: os.environ.get(k) for k in kv}
        os.environ.update({k: v for k, v in kv.items() if v is not None})
        for k, v in kv.items():
            if v is None: os.environ.pop(k, None)
        try: yield
        finally:
            for k, v in old.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
    OK_SRC = "def f(x):\n    return x+1\n\ndef check(c):\n    assert c(1)==2\ncheck(f)\n"
    # every writable mount must be a size-capped tmpfs, a device node, proc or devpts; the rest read-only
    MOUNTS_SRC = ("cap = %d\nbad = []\nfor line in open('/proc/self/mountinfo'):\n"
                  "    a, b = line.split(' - ')\n    f = a.split(); fstype, _, sopts = b.split()[:3]\n"
                  "    mnt, opts = f[4], f[5].split(',')\n    if 'ro' in opts: continue\n"
                  "    if fstype in ('proc', 'devpts'): continue\n"
                  "    if mnt in ('/dev/null', '/dev/zero', '/dev/full', '/dev/random', '/dev/urandom', '/dev/tty'): continue\n"
                  "    size = [o[5:] for o in sopts.split(',') if o.startswith('size=')]\n"
                  "    if fstype == 'tmpfs' and size and size[0].endswith('k') and int(size[0][:-1]) * 1024 <= cap: continue\n"
                  "    bad.append(mnt)\nassert not bad, bad\nassert {'/tmp', '/dev/shm'} <= {l.split()[4] for l in open('/proc/self/mountinfo')}\n"
                  % TMPFS_BYTES)
    results = [
        expect("pass", outcome(OK_SRC), ("pass", "")),
        expect("no network", outcome("import socket\ns=socket.create_connection(('1.1.1.1',80),timeout=2)\n"), ("fail", "OSError")),
        expect("timeout", outcome("while True: pass\n", 3), ("timeout", "")),
        expect("memory cap (process)", outcome("x = bytearray(4 << 30)\n"), ("fail", "MemoryError")),
        expect("memory under job cap", outcome(
            "import os, time\nfor _ in range(4):\n    if os.fork() == 0:\n"
            "        x = bytearray(150 << 20); x[::4096] = b'1' * len(x[::4096]); time.sleep(1); os._exit(0)\n"
            "for _ in range(4):\n    _, st = os.wait()\n    if st: raise SystemExit(9)\n", 20), ("pass", "")),
    ]
    if have_cgroup:
        results.append(expect("memory cap (job, cgroup)", outcome(
            "import os, time\nfor _ in range(4):\n    if os.fork() == 0:\n"
            "        x = bytearray(700 << 20); x[::4096] = b'1' * len(x[::4096]); time.sleep(3); os._exit(0)\n"
            "for _ in range(4):\n    _, st = os.wait()\n    if st: raise SystemExit(9)\n", 20), ("fail", None)))
    else:
        print("SKIP memory cap (job, cgroup): no per-job cgroup (--allow-no-cgroup)")
    results += [
        expect("file-size cap", outcome("open('/tmp/big','wb').write(b'0' * (40 << 20))\n"), ("fail", None)),
        expect("/tmp size cap", outcome("for i in range(4):\n    open(f'/tmp/f{i}','wb').write(b'0' * (12 << 20))\n"), ("fail", "OSError")),
        expect("/dev/shm size cap", outcome("for i in range(4):\n    open(f'/dev/shm/f{i}','wb').write(b'0' * (12 << 20))\n"), ("fail", "OSError")),
        expect("read-only /dev", outcome("open('/dev/big','wb').write(b'0' * (8 << 20))\n"), ("fail", "OSError")),
        expect("writable mounts all size-capped", outcome(MOUNTS_SRC), ("pass", "")),
        expect("read-only root", outcome("open('/x','w').write('x')\n"), ("fail", "OSError")),
        expect("read-only /work", outcome("open('/work/x','w').write('x')\n"), ("fail", "OSError")),
        expect("no nested userns", outcome("import os\nos.unshare(os.CLONE_NEWUSER)\n"), ("fail", None)),
        expect("process cap", outcome("import os, time\npids = []\ntry:\n    for _ in range(200):\n"
                                      "        pid = os.fork()\n        if pid == 0:\n            time.sleep(2); os._exit(0)\n"
                                      "        pids.append(pid)\nexcept OSError:\n    raise SystemExit(3)\n"), ("fail", None)),
        expect("fd cap", outcome("fs=[open('/work/prog.py') for _ in range(500)]\n"), ("fail", "OSError")),
        expect("stdout discarded", outcome("print('x' * (50 << 20))\n"), ("pass", "")),
        expect("stderr capped", outcome("import sys\nsys.stderr.write('y' * (8 << 20))\nraise ValueError('z')\n"), ("fail", "ValueError")),
    ]
    # completion proof: exit status 0 is not enough, the program (and so the appended tests) must run to the end
    HE_PROMPT = "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n"
    HE_TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n"
    def he(completion): return outcome(build_program(HE_PROMPT, completion, HE_TEST, "add"))
    for label, src in (
            ("sys.exit(0) at module level", "import sys\nsys.exit(0)\n"),
            ("raise SystemExit(0)", "raise SystemExit(0)\n"),
            ("exit()", "exit()\n"), ("quit()", "quit()\n"),
            ("os._exit(0)", "import os\nos._exit(0)\n"),
            ("stdout/stderr closed, then sys.exit(0)", "import os, sys\nos.close(1); os.close(2)\nsys.exit(0)\n"),
            ("closes every fd, then completes", "import os\nos.closerange(3, 64)\n"),
            ("forges the completion pipe", "import os\nfor fd in range(3, 64):\n    try: os.write(fd, b'0' * 32)\n"
                                           "    except OSError: pass\nos._exit(0)\n")):
        results.append(expect(f"early exit: {label}", outcome(src), ("fail", None)))
    results += [
        expect("stdout redirected, completes", outcome("import io, sys\nsys.stdout = io.StringIO()\nprint('x')\n"), ("pass", "")),
        expect("humaneval: correct", he("```python\ndef add(a, b):\n    return a + b\n```"), ("pass", "")),
        expect("humaneval: wrong", he("```python\ndef add(a, b):\n    return a - b\n```"), ("fail", "AssertionError")),
        expect("humaneval: sys.exit(0) inside the function",
               he("```python\ndef add(a, b):\n    import sys\n    sys.exit(0)\n```"), ("fail", EARLY_EXIT)),
        expect("humaneval: os._exit(0) inside the function",
               he("```python\ndef add(a, b):\n    import os\n    os._exit(0)\n```"), ("fail", EARLY_EXIT)),
        expect("humaneval: sys.exit(0) at module level before the tests",
               he("```python\nimport sys\ndef add(a, b):\n    return a - b\nsys.exit(0)\n```"), ("fail", EARLY_EXIT)),
        expect("humaneval: stdout replaced, still correct",
               he("```python\nimport sys, io\nsys.stdout = io.StringIO()\ndef add(a, b):\n    return a + b\n```"), ("pass", "")),
    ]
    # the caller's environment is not visible inside the sandbox
    os.environ["HE_EXEC_SECRET_PROBE"] = "x"
    results.append(expect("clean env", outcome("import os,sys\nassert 'HE_EXEC_SECRET_PROBE' not in os.environ, 'leak'\nassert set(os.environ) <= %r\n" % (set(SANDBOX_ENV) | {'PWD'})), ("pass", "")))
    # cgroup unavailable (simulated): refused by default, runs only with the unsafe opt-out
    with env(HE_EXEC_SYSTEMD_RUN="/nonexistent/systemd-run", HE_EXEC_ALLOW_NO_CGROUP=None):
        results.append(expect("no cgroup: refused by default", outcome(OK_SRC), ("raised", "SandboxUnavailable")))
    with env(HE_EXEC_SYSTEMD_RUN="/nonexistent/systemd-run", HE_EXEC_ALLOW_NO_CGROUP="1"):
        results.append(expect("no cgroup: runs with HE_EXEC_ALLOW_NO_CGROUP=1", outcome(OK_SRC), ("pass", "")))
    # systemd-run failing at run time (probe passed; /bin/false stands in for a failing scope creation)
    with env(HE_EXEC_SYSTEMD_RUN="/bin/false", HE_EXEC_ALLOW_NO_CGROUP=None):
        _probe["/bin/false"] = True
        try: results.append(expect("systemd-run failure is an infra error", outcome(OK_SRC), ("raised", "SandboxInfraError")))
        finally: _probe.pop("/bin/false", None)
    print("per-job cgroup:", "yes" if have_cgroup else "NO (UNSAFE opt-out: per-process rlimits only)")
    sys.exit(0 if all(results) else 1)
