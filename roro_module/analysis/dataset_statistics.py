from collections import defaultdict
from pathlib import Path
import re

class RoRoDatasetStatistics:
    __word_re = re.compile(r"\w+", flags=re.UNICODE)

    def __init__(self):
        pass

    def __text_len_words(self, s):
        if not isinstance(s, str):
            return 0, 0
        s = s.strip()
        return len(s), len(self.__word_re.findall(s))

    def __ancestors_with_gazeta(self, relative_parent, gazeta):
        yield "(root)"
        parts = list(relative_parent.parts)
        acc = []
        for p in parts:
            acc.append(p)
            yield "/".join(acc)
        if gazeta:
            yield "/".join(acc + [gazeta]) if acc else gazeta

    def run(self, entries, **kwargs):
        folder_stats = defaultdict(lambda: {"files": 0, "chars": 0, "words": 0})
        processed, skipped = 0, 0

        for entry in entries:
            try:
                content = entry.text
                chars, words = self.__text_len_words(content)

                rel_path = Path(entry.meta["rel_path"])
                fname = rel_path.name
                gazeta = fname.rsplit("_", 1)[0]
                rel_parent = rel_path.parent

                for cat in self.__ancestors_with_gazeta(rel_parent, gazeta):
                    folder_stats[cat]["files"] += 1
                    folder_stats[cat]["chars"] += chars
                    folder_stats[cat]["words"] += words
                processed += 1
            except Exception as e:
                print(f"[err] Failed to process {entry.meta.get('rel_path', '?')}: {e}")
                skipped += 1

        enriched_stats = {}
        for cat, vals in folder_stats.items():
            level = 0 if cat == "(root)" else cat.count("/") + 1
            enriched_stats[cat] = {"level": level, **vals}

        return {"stats": enriched_stats, "processed": processed, "skipped": skipped}
