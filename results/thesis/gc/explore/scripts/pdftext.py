import sys
sys.path.insert(0, "/tmp/claude-1000/-home-gellert-Developer-medical-dt/187d5d35-6c3c-4c5e-8d5a-f69463aa63f9/scratchpad/pylib")
import pymupdf
for p in sys.argv[1:]:
    doc = pymupdf.open(p)
    print("=" * 20, p, "pages", doc.page_count)
    for i, page in enumerate(doc):
        print(f"--- page {i+1}")
        print(page.get_text())
