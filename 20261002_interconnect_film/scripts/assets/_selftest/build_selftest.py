"""Self-test of the shared asset helpers: builds a tiny asset and previews it."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import common as C

OUT = C.argv_after_dashes()[0]
C.reset()
coll, root = C.new_asset("selftest", accuracy="C")
m = C.principled("MAT_selftest_gold", (0.8, 0.6, 0.2), metallic=1.0, rough=0.3)
b = C.cube("body", (C.mm(40), C.mm(20), C.mm(5)), mat=m)
C.add(b, coll, root)
c = C.cylinder("post", C.mm(3), C.mm(12), loc=(0, 0, C.mm(8.5)), mat=C.principled("MAT_selftest_gray", (0.3, 0.3, 0.35)))
C.add(c, coll, root)
C.hook("top", coll, root, loc=(0, 0, C.mm(15)))
C.finish("selftest", os.path.join(OUT, "selftest.blend"), coll, {"sources": [], "notes": "self-test"}, preview_dir=os.path.join(OUT, "previews"), views=("three_quarter",))
