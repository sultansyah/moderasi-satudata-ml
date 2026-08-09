# ============================================
# Blacklist keyword untuk Sistem Moderasi Gambar
# YOLOv11 + Tesseract + Keyword Filtering
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

# Keyword judi online / slot (Indonesia, Vietnam, dan brand ASEAN).
# Istilah Vietnam ditulis TANPA diakritik: normalize() mengubah teks OCR
# beraksen (đánh bạc -> "danh bac") sebelum dicocokkan.
# Keyword panjang <= 3 karakter (bet, w88, ok9, ...) dicocokkan sebagai KATA UTUH,
# supaya tidak salah-match substring di kata lain (mis. "bet" di "alphabet").
KEYWORDS_JUDI = [
    # Indonesia
    "judi online", "situs judi", "bandar judi", "agen judi", "judi bola",
    "taruhan online", "judol", "judi slot", "slot gacor", "gacor", "maxwin",
    "link slot", "situs slot", "daftar slot", "slot online", "slot demo",
    "idn poker", "poker online", "domino qq", "baccarat", "roulette", "sicbo",
    # Vietnam (transliterasi ASCII)
    "danh bac", "danh bai", "ca cuoc", "ca do", "nha cai", "lo de", "xo so",
    "no hu", "quay hu", "ban ca", "doi thuong", "tai xiu", "xoc dia",
    "co bac", "bau cua", "da ga", "song bai", "game bai",
    # Brand situs judi (ASEAN)
    "w88", "m88", "fun88", "f88", "188bet", "1xbet", "bet365", "sbobet",
    "sbotop", "kubet", "jun88", "vn88", "shbet", "hi88", "new88", "ok9",
    "78win", "bk8", "debet", "fcb8", "8xbet", "bwing", "tf88", "v9bet",
    "dabet", "may88", "rikvip", "go88", "kingfun", "red88", "ee88", "zbet",
    "12bet", "letou", "win55", "fb88", "cmd368", "oppabet", "789bet",
    # Game slot populer (ZEUS, OLYMPUS, dll.)
    "bet", "zeus", "olympus", "sweet bonanza", "bonanza", "starlight princess",
    "mahjong ways", "sugar rush", "big bass", "wolf gold", "aztec gems",
    "book of dead", "wisdom of athena", "gates of hades", "fruit party",
    "pragmatic", "jili", "slot88", "cq9", "habanero", "pussy888", "mega888",
    "pg soft", "ante bet",
]

# Gabungan semua keyword (dipakai untuk pencocokan OCR)
KEYWORDS_ALL = list(set(KEYWORDS_ABORSI + KEYWORDS_BORAKS + KEYWORDS_UMUM + KEYWORDS_JUDI))

# Alternatif: satu string besar untuk substring match cepat
KEYWORDS_TEXT = "|".join(KEYWORDS_ALL)

if __name__ == "__main__":
    print(f"Total keyword: {len(KEYWORDS_ALL)}")
    for k in sorted(KEYWORDS_ALL):
        print("  -", k)
