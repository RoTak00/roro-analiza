from collections import defaultdict, Counter
from pathlib import Path
import spacy
import gc

class RoRoSentenceStats:

    def __init__(self, level = -1, spacy_model = "ro_core_news_sm", batch_size=512):
        self.level = level
        self.spacy_model = spacy_model
        self.batch_size = batch_size
        pass

    def _stats_from_doc(self, doc):
        """
        Given a spaCy document, returns five lists of statistics:
        1. sentence lengths (number of alphabetic tokens per sentence)
        2. number of punctuation tokens per sentence
        3. number of stop words per sentence
        4. number of pronouns per sentence
        5. unique word percentage (per document)

        :param doc: spaCy document
        :return: five lists of statistics
        """
        sent_lengths = []
        punct_counts = []
        stop_counts = []
        pron_counts = []

        for sent in doc.sents:
            words = [t for t in sent if t.is_alpha]
            puncts = [t for t in sent if t.is_punct]
            stops = [t for t in sent if t.is_stop]
            prons = [t for t in sent if t.pos_ == "PRON"]

            if words or puncts or stops or prons:
                sent_lengths.append(len(words))
                punct_counts.append(len(puncts))
                stop_counts.append(len(stops))
                pron_counts.append(len(prons))

        # unique word percentage (per doc)
        words_in_doc = [t.text.lower() for t in doc if t.is_alpha]
        unique_pct = (len(set(words_in_doc)) / len(words_in_doc)) if words_in_doc else 0.0

        return sent_lengths, punct_counts, stop_counts, pron_counts, unique_pct
    
    def _aggregate_results(self, folder_sentence_data, folder_unique_data):
        """
        Given two dictionaries, `folder_sentence_data` and `folder_unique_data`, where each value is a dictionary with
        the following keys:

        - `lengths`: list of sentence lengths
        - `puncts`: list of sentence punctuation counts
        - `stops`: list of sentence stop word counts
        - `prons`: list of sentence pronoun counts

        and `folder_unique_data` has the same structure but with unique word percentage values.

        Returns a dictionary with the same keys as `folder_sentence_data`, where each value is a dictionary with the
        following keys:

        - `avg_words_per_sentence`: average sentence length
        - `avg_punct_per_sentence`: average sentence punctuation count
        - `avg_stop_per_sentence`: average sentence stop word count
        - `avg_pronoun_per_sentence`: average sentence pronoun count
        - `avg_unique_word_pct`: average unique word percentage

        Each value is calculated as the average of the corresponding list of values in the input dictionaries.
        If a list is empty, the corresponding value is set to 0.0.
        """
        result = {}
        for folder, data in folder_sentence_data.items():
            # unpack arrays
            lengths, puncts, stops, prons = data["lengths"], data["puncts"], data["stops"], data["prons"]

            result[folder] = {
                "avg_words_per_sentence": sum(lengths) / len(lengths) if lengths else 0.0,
                "avg_punct_per_sentence": sum(puncts) / len(puncts) if puncts else 0.0,
                "avg_stop_per_sentence": sum(stops) / len(stops) if stops else 0.0,
                "avg_pronoun_per_sentence": sum(prons) / len(prons) if prons else 0.0,
                "avg_unique_word_pct": sum(folder_unique_data[folder]) / len(folder_unique_data[folder]) if folder_unique_data[folder] else 0.0,
            }
        return result
    
    def _from_docs(self, entries, level=-1):
        """
        Given an iterable of entries, each with a spaCy document, calculates and
        returns aggregated sentence statistics (average sentence length, average
        punctuation, stop words, and pronouns per sentence, and unique word
        percentage) for each folder in the hierarchy.

        :param entries: iterable of entries with a spaCy document
        :param level: -1 for last folder, 0 for root folder, 1 for first folder ...
        :return: a dictionary with two keys: "stats" and "processed". The value
            of "stats" is another dictionary with folder names as keys and a
            dictionary of aggregated statistics as values. The value of
            "processed" is the number of entries processed.
        """
        folder_sentence_data = defaultdict(lambda: {"lengths": [], "puncts": [], "stops": [], "prons": []})
        folder_unique_data = defaultdict(list)
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

            lengths, puncts, stops, prons, unique_pct = self._stats_from_doc(doc)
            folder_sentence_data[folder]["lengths"].extend(lengths)
            folder_sentence_data[folder]["puncts"].extend(puncts)
            folder_sentence_data[folder]["stops"].extend(stops)
            folder_sentence_data[folder]["prons"].extend(prons)
            folder_unique_data[folder].append(unique_pct)

            total_processed += 1
            if total_processed % 100 == 0:
                print(f"[info] Processed {total_processed} entries")

        result = self._aggregate_results(folder_sentence_data, folder_unique_data)
        return {"stats": result, "processed": total_processed}
    
    def _chunks(self, seq, size):
        """Yield successive slices of size from seq."""
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    def _from_text(self, entries, level=-1):
        nlp = spacy.load(self.spacy_model)
        folder_sentence_data = defaultdict(lambda: {"lengths": [], "puncts": [], "stops": [], "prons": []})
        folder_unique_data = defaultdict(list)
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

                lengths, puncts, stops, prons, unique_pct = self._stats_from_doc(doc)
                folder_sentence_data[folder]["lengths"].extend(lengths)
                folder_sentence_data[folder]["puncts"].extend(puncts)
                folder_sentence_data[folder]["stops"].extend(stops)
                folder_sentence_data[folder]["prons"].extend(prons)
                folder_unique_data[folder].append(unique_pct)

                total_processed += 1

                del doc

            print("[info] Collecting garbage after batch")
            print(f"[info] Processed {total_processed}/{len(entries)} entries ({total_processed/len(entries)*100}%)")
            gc.collect()

        result = self._aggregate_results(folder_sentence_data, folder_unique_data)
        return {"stats": result, "processed": total_processed}


    def run(self, entries, **kwargs):
        """
        Run sentence statistics analysis on the given entries.

        If the entries are pre-parsed with a Spacy model (i.e. they have a "doc"
        attribute), use the _from_docs method. Otherwise, use _from_text.

        :param entries: list of entries to process
        :param **kwargs: additional keyword arguments to pass to the analysis
                         module. Currently accepted:
                         - level: str, level of folder aggregation (default: "last")

        :return: dictionary with analysis results and the number of processed entries
        """
        level = kwargs.get("level", self.level)
        if entries and getattr(entries[0], "doc", None) is not None:
            return self._from_docs(entries, level=level)
        else:
            return self._from_text(entries, level=level)

