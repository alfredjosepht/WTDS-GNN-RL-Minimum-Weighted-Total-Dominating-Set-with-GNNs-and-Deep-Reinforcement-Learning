"""Run a RESUMABLE command, pausing it when free RAM gets low (Windows/Linux).

    python scripts/guarded_run.py --min-free-gb 1.0 -- python -m wtds.evaluate --config ...

If free physical memory drops below --min-free-gb, the child is terminated; once
free memory is back above min + --resume-margin-gb it is started again. Only use
this for commands that skip already-finished work (e.g. wtds.evaluate).
"""
import argparse
import ctypes
import subprocess
import sys
import time


def free_gb() -> float:
    if sys.platform == "win32":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2**30
    with open("/proc/meminfo") as f:
        info = dict(line.split(":") for line in f)
    return int(info["MemAvailable"].split()[0]) / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-gb", type=float, default=1.0)
    ap.add_argument("--resume-margin-gb", type=float, default=1.0)
    ap.add_argument("--low-priority", action="store_true",
                    help="run the child below normal CPU priority (yields CPU to training)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if cmd and cmd[0] in ("python", "python3"):
        cmd = [sys.executable] + cmd[1:]          # same interpreter / venv as this script
    while True:
        while free_gb() < a.min_free_gb + a.resume_margin_gb:
            time.sleep(20)
        print(f"[guard] start (free {free_gb():.2f} GB): {' '.join(cmd)}", flush=True)
        kw = {}
        if a.low_priority:
            if sys.platform == "win32":
                kw["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
            else:
                kw["preexec_fn"] = lambda: __import__("os").nice(10)
        p = subprocess.Popen(cmd, **kw)
        while p.poll() is None:
            time.sleep(5)
            if free_gb() < a.min_free_gb:
                print(f"[guard] free RAM {free_gb():.2f} GB < {a.min_free_gb}: pausing child", flush=True)
                p.terminate()
                p.wait()
                break
        else:
            print(f"[guard] child exited with {p.returncode}", flush=True)
            sys.exit(p.returncode)


if __name__ == "__main__":
    main()
