import re
import csv 
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

class RoRoAnalyzer:
    def __init__(self, parser):
        self.parser = parser

        self.cache = {}

    def __flatten_dict(self, d):
        items = []
        for k, v in d.items():
            if isinstance(v, dict):
                items.extend(self.__flatten_dict(v))
            else:
                items.append(v)
        return items

    def run(self, func_name, query, uses_dict = True, **kwargs):
        """
        Runs a given function with the given query and arguments

        :param func_name: The name of the function to run
        :param query: The query to pass to the parser
        :param uses_dict: Whether the function expects a nested dict or a flat list of entries
        :param **kwargs: Additional arguments to pass to the function

        :return: The result of the function
        """
        func = getattr(self, func_name, None)

        if func is None:
            raise ValueError(f"Function {func_name} not found")

        entries = None 
        if query is not None:
            entries = self.parser.get(query)

            if uses_dict == False:
                entries = self.__flatten_dict(entries)
        elif uses_dict:
            entries = self.parser.get()
        else:
            entries = self.parser.get_flat()

        if entries is None:
            raise ValueError("No entries found")
        
        if not callable(func):
            raise ValueError(f"Function {func_name} is not callable")
        
        result = func(entries, **kwargs)

        self.cache = {
            "name": func_name,
            "query": query,
            "result": result
        }

        return result
    

    __word_re = re.compile(r"\w+", flags=re.UNICODE)
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
        
    
    def dataset_statistics(self, entries, **kwargs):
        """
        Computes statistics about the given dataset.

        This function takes a list of entries and returns a dict with the following keys:

        - `stats`: A nested dictionary with the following structure:
            - `(root)`: A dict with keys `files`, `chars`, and `words` containing the total counts of the above.
            - Each subfolder: A dict with keys `files`, `chars`, and `words` containing the total counts of the above for that subfolder.
        - `processed`: The number of entries successfully processed
        - `skipped`: The number of entries skipped due to errors

        The function will count the number of files, characters, and words in each subfolder and the total, and return the results.
        """
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
            enriched_stats[cat] = {
                "level": level, 
                **vals,
            }

        return {"stats": enriched_stats, "processed": processed, "skipped": skipped}
            
        
    def save_csv(self, out_csv="analysis.csv"):
        """
        Saves the current statistics to a CSV file.

        If the cache is empty or does not contain the `stats` key, this function will print an error message and return.

        The CSV file will have the following structure:

        - The first column is the key (folder name or "(root)")
        - The following columns are the values for each key, in the order specified by the `all_fields` set.
        - If a key is missing a particular field, the value will be an empty string.

        The filename is specified by the `out_csv` parameter, which defaults to "analysis.csv".
        """
        if not self.cache:
            print("[err] Nothing in cache")
            return

        result = self.cache["result"]
        if "stats" not in result:
            print("[err] Cache does not contain folder stats")
            return
        
        stats = result["stats"]
        
        if not isinstance(stats, dict) or not all(isinstance(v, dict) for v in stats.values()):
            print("[err] Statistics is not a dict-of-dicts; cannot save clean CSV")
            return

        # collect all field names across all inner dicts
        all_fields = set()
        for v in stats.values():
            all_fields.update(v.keys())
        all_fields = sorted(all_fields)

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["key"] + all_fields
            w.writerow(header)

            for k, v in stats.items():
                row = [k] + [v.get(field, "") for field in all_fields]
                w.writerow(row)

        print(f"[Analyzer] Saved CSV -> {out_csv}")