"""CONTRACT
Input: free-text queries on argv (e.g. "beaker" "analytical balance").
Output: stdout, one line per hit: uid, face count, license slug, name, viewer URL.
  Only downloadable models are searched (downloadable=true); anonymous API, no token.
Consumer: a human/agent picks uids, then tools/fetch_objaverse.py downloads them
  through Objaverse 1.0 (same uid space as Sketchfab; not every uid is in Objaverse).
Assumed convention: Sketchfab uid == Objaverse 1.0 uid (verified: objaverse.load_objects
  returns the same model for a uid found here, see tools/fetch_objaverse.py output).
Reference output: none stored; results change as Sketchfab's index changes.
"""
import json, sys, urllib.parse, urllib.request

def search(q, count=24):
    url = "https://api.sketchfab.com/v3/search?" + urllib.parse.urlencode(
        {"type": "models", "q": q, "downloadable": "true", "count": count,
         "sort_by": "-likeCount"})
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)["results"]

for q in sys.argv[1:]:
    print(f"## {q}")
    for m in search(q):
        lic = (m.get("license") or {}).get("slug", "?")
        print(f"{m['uid']}  faces={m.get('faceCount')}  {lic:10s}  {m['name'][:60]}  {m['viewerUrl']}")
