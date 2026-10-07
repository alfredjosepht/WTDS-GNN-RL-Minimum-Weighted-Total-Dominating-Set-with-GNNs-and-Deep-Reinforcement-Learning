"""SHA-256 of every file under a folder (read-only). Usage: python hash_tree.py <folder> <out.txt>"""
import hashlib
import os
import sys

root, out = sys.argv[1], sys.argv[2]
rows = []
for dp, dn, fn in os.walk(root):
    dn.sort()
    for f in sorted(fn):
        p = os.path.join(dp, f)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        rows.append(f"{h.hexdigest()}  {os.path.relpath(p, root).replace(os.sep, '/')}")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(rows) + "\n")
print(f"{len(rows)} files hashed -> {out}")
