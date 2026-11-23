"""Wikimedia Commons metadata lookup (generic User-Agent, 1.2 s between requests, exits on HTTP errors such as 429).
Usage: uv run --no-project python commons.py info "File:Name.jpg" ... | cat "Category:..." [limit]
"""
import json, sys, time, urllib.request, urllib.parse
UA = "course-media-fetcher/1.0 (educational project)"
API = "https://commons.wikimedia.org/w/api.php"
def get(params):
    params = dict(params, format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code); sys.exit(1)
    time.sleep(1.2)
    return data
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cat":
        d = get({"action": "query", "list": "categorymembers", "cmtitle": sys.argv[2], "cmtype": "file", "cmlimit": sys.argv[3] if len(sys.argv) > 3 else "40"})
        for m in d["query"]["categorymembers"]:
            print(m["title"])
    elif mode == "info":
        titles = sys.argv[2:]
        d = get({"action": "query", "titles": "|".join(titles), "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": "800"})
        for p in d["query"]["pages"]:
            ii = p.get("imageinfo", [{}])[0]; em = ii.get("extmetadata", {})
            g = lambda k: em.get(k, {}).get("value", "")
            print(json.dumps({"title": p["title"], "w": ii.get("width"), "h": ii.get("height"), "thumb": ii.get("thumburl"), "url": ii.get("url"), "desc": ii.get("descriptionurl"),
                "artist": g("Artist")[:600], "license": g("LicenseShortName"), "licurl": g("LicenseUrl"), "credit": g("Credit")[:150], "attr": g("AttributionRequired")}))
