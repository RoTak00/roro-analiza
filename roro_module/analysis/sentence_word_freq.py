from collections import defaultdict, Counter
from pathlib import Path
import spacy
import gc

class RoRoSentenceWordFreq:

    def __init__(self, level = -1, spacy_model = "ro_core_news_sm", batch_size=512):
        self.level = level
        self.spacy_model = spacy_model
        self.batch_size = batch_size
        pass

    def _avg_words_per_sentence_from_docs(self, entries, level = -1):
        folder_sentence_lengths = defaultdict(list)
        total_processed = 0

        for entry in entries:
            doc = getattr(entry, "doc", None)
            if doc is None:
                continue

            rel_path = Path(entry.meta["rel_path"])
            parts = list(rel_path.parts)

            if level == 0:
                folder = parts[0]  # first (root-level)
            elif level == -1:
                folder = parts[-2] if len(parts) > 1 else parts[0]  # last folder, not file
            elif level > 0:
                if level < len(parts) - 1:
                    folder = parts[level]  # specific folder depth
                else:
                    folder = parts[-2] if len(parts) > 1 else parts[0]  # fallback to last folder
            else:
                folder = "(root)"

            for sent in doc.sents:
                words = [t for t in sent if t.is_alpha]
                if words:
                    folder_sentence_lengths[folder].append(len(words))
            
            total_processed += 1
            if total_processed % 100 == 0:
                print(f"[info] Processed {total_processed} entries")
            
        result = {
            folder: {"avg_words_per_sentence": sum(lengths) / len(lengths)} for folder, lengths in folder_sentence_lengths.items() if lengths
        }

        return {"stats": result, "processed": total_processed}
    
    def _chunks(self, seq, size):
        """Yield successive slices of size from seq."""
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    def _avg_words_per_sentence_from_text(self, entries, level = -1):
        
        nlp = spacy.load(self.spacy_model)

        folder_sentence_lengths = defaultdict(list)
        total_processed = 0

        for chunk in self._chunks(entries, self.batch_size):
            for entry, doc in zip(chunk, nlp.pipe((e.text for e in chunk), batch_size=self.batch_size, n_process=-1)):
                rel_path = Path(entry.meta["rel_path"])
                parts = list(rel_path.parts)

                if level == 0:
                    folder = parts[0]  # first (root-level)
                elif level == -1:
                    folder = parts[-2] if len(parts) > 1 else parts[0]  # last folder, not file
                elif level > 0:
                    if level < len(parts) - 1:
                        folder = parts[level]  # specific folder depth
                    else:
                        folder = parts[-2] if len(parts) > 1 else parts[0]  # fallback to last folder
                else:
                    folder = "(root)"

                for sent in doc.sents:
                    words = [t for t in sent if t.is_alpha]
                    if words:
                        folder_sentence_lengths[folder].append(len(words))
                
                total_processed += 1

                del doc
            
            print ("[info] Collecting garbage after batch")
            print (f"[info] Processed {total_processed} entries")
            gc.collect()

        result = {
            folder: {"avg_words_per_sentence": sum(lengths) / len(lengths)} for folder, lengths in folder_sentence_lengths.items() if lengths
        }

        return {"stats": result, "processed": total_processed}


    def run(self, entries, **kwargs):
        
       level = kwargs.get("level", self.level)
       
       if entries and getattr(entries[0], "doc", None) is not None:
           return self._avg_words_per_sentence_from_docs(entries, level=level)
       else:
           return self._avg_words_per_sentence_from_text(entries, level=level)

