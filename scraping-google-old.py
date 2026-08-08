import os
import re
import time
import urllib.parse
import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

OUT = r"C:\Users\sulta\Desktop\dataset_obat_aborsi_google"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
PROFILE = r"C:\Users\sulta\AppData\Local\Temp\opencode\chrome_profile"

QUERIES = {
    "obat_aborsi_asli": ["obat aborsi asli", "obat aborsi original", "pil aborsi asli", "obat aborsi ampuh"],
    "obat_penggugur": ["obat penggugur kandungan", "obat penggugur kandungan asli", "pil penggugur kandungan"],
    "pil_aborsi": ["pil aborsi", "obat aborsi murah", "obat telat datang bulan", "pelancar haid"],
    "abortion_pills": ["abortion pills", "mifepristone misoprostol kit", "abortion pill packaging"],
    "misoprostol": ["misoprostol cytotec", "cytotec 200 mg", "misoprostol tablet"],
}


def make_driver():
    opts = Options()
    opts.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    opts.add_argument(f"--user-data-dir={PROFILE}")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1600,1000")
    opts.add_argument("--lang=en-US")
    opts.add_argument(f"--user-agent={UA}")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(service=Service(), options=opts)


def is_blocked(driver):
    return "sorry/index" in driver.current_url or "unusual traffic" in driver.find_element("tag name", "body").text


def collect_urls(driver):
    urls = set()
    for el in driver.find_elements("css selector", "img[src*='encrypted-tbn0.gstatic.com/images']"):
        src = el.get_attribute("src") or ""
        if "faviconV2" in src:
            continue
        urls.add(src)
    return urls


def wait_for_captcha(driver, q, timeout=240):
    print(f"=== [ACTION NEEDED] Chrome window opened. If a CAPTCHA appears, please solve it. Searching: {q}")
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
    url = "https://www.google.com/search?q=" + urllib.parse.quote(q) + "&tbm=isch&hl=en&tbs=isz:l"
    driver.get(url)
    time.sleep(5)
    if is_blocked(driver):
        ok = wait_for_captcha(driver, q)
        if not ok:
            print(f"  [{q}] captcha not solved, skipping")
            return []
    found = set()
    last = -1
    for _ in range(8):
        found |= collect_urls(driver)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        n = len(found)
        if n == last and n > 0:
            break
        last = n
        if n >= max_urls:
            break
    print(f"  {q}: collected {len(found)} urls")
    return list(found)[:max_urls]


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "image/*,*/*;q=0.8"})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read()
    except Exception:
        return False
    if len(data) < 500:
        return False
    with open(path, "wb") as f:
        f.write(data)
    return True


def main():
    os.makedirs(OUT, exist_ok=True)
    driver = make_driver()
    stats = {}
    try:
        for folder, queries in QUERIES.items():
            d = os.path.join(OUT, folder)
            os.makedirs(d, exist_ok=True)
            got = 0
            seen = set()
            for q in queries:
                if got >= 70:
                    break
                try:
                    urls = search_google(driver, q, 60)
                except Exception as e:
                    print(f"  search error: {e}")
                    continue
                for i, u in enumerate(urls):
                    if got >= 70:
                        break
                    if u in seen:
                        continue
                    seen.add(u)
                    path = os.path.join(d, f"{q.replace(' ', '_')}_{i:03d}.img")
                    if download(u, path):
                        got += 1
                        print(f"  + {os.path.basename(path)}")
                    time.sleep(0.15)
            stats[folder] = got
            print(f"[done] {folder}: {got}")
    finally:
        driver.quit()

    print("\n=== SUMMARY ===")
    total = 0
    for f, n in stats.items():
        print(f"  {f}: {n}")
        total += n
    print(f"  TOTAL: {total} -> {OUT}")


if __name__ == "__main__":
    main()
