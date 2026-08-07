# ============================================
# Blacklist keyword untuk Sistem Moderasi Gambar
# Portal SatuData - YOLOv11 + Tesseract + Keyword Filtering
# ============================================

# Keyword terkait obat aborsi / penggugur kandungan (iklan ilegal)
KEYWORDS_ABORSI = [
    "aborsi",
    "penggugur",
    "gugur kandungan",
    "pil aborsi",
    "obat aborsi",
    "obat penggugur",
    "jual obat aborsi",
    "telat datang bulan",
    "terlambat datang bulan",
    "pelancar haid",
    "pelancar mens",
    "obat telat bulan",
    "obat telat datang bulan",
    "penggugur janin",
    "penggugur kandungan",
    "aborsi aman",
    "aborsi tuntas",
    "aborsi secara alami",
    "cytotec",
    "misoprostol",
    "mifepristone",
    "gastrul",
    "prostadel",
    "aborsiva",
    "kuretase",
    "jual cytotec",
    "obat kuret",
    "penggugur kehamilan",
]

# Keyword terkait boraks / formalin pada makanan (pelanggaran BPOM)
KEYWORDS_BORAKS = [
    "boraks",
    "borax",
    "bleng",
    "bahan pengenyal",
    "pengenyal bakso",
    "boraks makanan",
    "jual boraks",
    "bubuk boraks",
    "kristal boraks",
    "formalin",
    "pengawet mayat",
]

# Keyword umum lain yang sering muncul di iklan spam ilegal
KEYWORDS_UMUM = [
    "100% original",
    "100% ampuh",
    "order langsung",
    "wa.me",
    "whatsapp",
    "cod",
    "dijamin tuntas",
    "ampuh tuntas",
    "rasa aman",
]

# Gabungan semua keyword (dipakai untuk pencocokan OCR)
KEYWORDS_ALL = list(set(KEYWORDS_ABORSI + KEYWORDS_BORAKS + KEYWORDS_UMUM))

# Alternatif: satu string besar untuk substring match cepat
KEYWORDS_TEXT = "|".join(KEYWORDS_ALL)

if __name__ == "__main__":
    print(f"Total keyword: {len(KEYWORDS_ALL)}")
    for k in sorted(KEYWORDS_ALL):
        print("  -", k)
