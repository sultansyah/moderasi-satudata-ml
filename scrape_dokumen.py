import os
import re
import time
import urllib.parse
import urllib.request
import hashlib
import io

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

OUT = r"C:\Users\sulta\Desktop\dataset_dokumen"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
PROFILE = r"C:\Users\sulta\AppData\Local\Temp\opencode\chrome_profile"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

QUERIES = [
    "poster acara", "poster seminar", "poster edukasi", "poster promosi",
    "poster acara seminar", "poster kesehatan", "poster ilmiah",
    "sertifikat penghargaan", "sertifikat pelatihan", "sertifikat workshop",
    "sertifikat kursus", "sertifikat kehadiran",
    "banner promosi", "spanduk promosi", "brosur produk", "flyer promosi",
    "dokumen resmi", "surat resmi", "surat keterangan", "surat undangan",
    "ijazah", "transkrip nilai", "kartu identitas", "kartu anggota",
    "undangan pernikahan", "infografis pendidikan", "poster sekolah",
    "sertifikat kompetensi", "lembar pengumuman", "piagam penghargaan",
]

MIN_EDGE = 200
PER_QUERY = 12
TOTAL = 300


def make_driver():
    opts = Options()
    opts.binary_location = CHROME
    opts.add_argument(f"--user-data-dir={PROFILE}")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1600,1000")
    opts.add_argument("--lang=id-ID")
    opts.add_argument(f"--user-agent={UA}")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(service=Service(), options=opts)


def is_blocked(driver):
    try:
        body = driver.find_element("tag name", "body").text
    except Exception:
        return True
    return ("sorry/index" in driver.current_url
            or "unusual traffic" in body
            or "tidak wajar" in body
            or "lalu lintas" in body)


def collect_urls(driver):
    urls, seen = [], set()
    for el in driver.find_elements("css selector", "img[src*='encrypted-tbn0.gstatic.com/images']"):
        src = el.get_attribute("src") or ""
        if "faviconV2" in src or src in seen:
            continue
        seen.add(src)
        urls.append(src)
    for a in driver.find_elements("css selector", "a[href*='imgurl=']"):
        href = a.get_attribute("href") or ""
        m = re.search(r"[?&]imgurl=([^&]+)", href)
        if m:
            u = urllib.parse.unquote(m.group(1))
            if u.startswith("http") and u not in seen:
                seen.add(u)
                urls.append(u)
    return urls


def wait_for_captcha(driver, q, timeout=240):
    print(f"\n=== [ACTION NEEDED] Jendela Chrome terbuka. Kalau ada CAPTCHA, selesaikan. Query: {q}")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if not is_blocked(driver) and len(driver.find_elements("css selector", "img")) > 3:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def search_google(driver, q, max_urls=60):
    url = "https://www.google.com/search?q=" + urllib.parse.quote(q) + "&tbm=isch&hl=id&tbs=isz:l"
    driver.get(url)
    time.sleep(5)
    if is_blocked(driver):
        if not wait_for_captcha(driver, q):
            print(f"  [{q}] captcha tidak selesai, lewati")
            return []
    found = []
    last = -1
    for _ in range(10):
        found = collect_urls(driver)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        n = len(found)
        if n == last and n > 0:
            break
        last = n
        if n >= max_urls:
            break
    print(f"  {q}: collected {len(found)} urls")
    return found[:max_urls]


def _save_image(data, dst):
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:
        return False
    w, h = im.size
    if min(w, h) < MIN_EDGE:
        return False
    md5 = hashlib.md5(data).hexdigest()[:16]
    path = os.path.join(dst, f"{md5}.jpg")
    if os.path.isfile(path):
        return True
    im.convert("RGB").save(path, "JPEG", quality=88)
    return True


def download(url, dst):
    if not url.startswith("http"):
        return False
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "image/*,*/*;q=0.8"})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read()
    except Exception:
        return False
    if len(data) < 10 * 1024:
        return False
    return _save_image(data, dst)


def main():
    os.makedirs(OUT, exist_ok=True)
    done = len([f for f in os.listdir(OUT) if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))])
    print(f"Folder tujuan : {OUT}")
    print(f"Sudah ada     : {done} gambar")
    print(f"Target        : {TOTAL} ({PER_QUERY} per query)\n")
    if done >= TOTAL:
        print("Target sudah tercapai, selesai.")
        return

    driver = make_driver()
    try:
        for q in QUERIES:
            if done >= TOTAL:
                break
            try:
                urls = search_google(driver, q, 60)
            except Exception as e:
                print(f"  search error: {e}")
                continue
            got = 0
            for u in urls:
                if got >= PER_QUERY or done >= TOTAL:
                    break
                if download(u, OUT):
                    got += 1
                    done += 1
                time.sleep(0.15)
            print(f"  -> {got} baru (total {done})")
    finally:
        driver.quit()

    print(f"\nSELESAI. Total gambar di {OUT}: {done}")


if __name__ == "__main__":
    main()
