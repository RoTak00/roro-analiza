import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass 

@dataclass 
class RoRoEntry: 
    text: str
    meta: dict 
    doc: object | None = None

class RoRoParser:
    """
    Parser class for processing .json data files of the RoRo dataset

    Attributes
    ----------
    path : str
        The path of the directory containing the .json files
    use_spacy : bool
        Should the parser load also SpaCy docs of the text
    """
    def __init__(self, options = None):

        """
        Initialize the parser with a dictionary of options

        Parameters
        ----------
        options : dict
            A dictionary of options
            Example:
                {
                    'path': 'data', # The path of the directory containing the .json files
                    'use_spacy': False # Should the parser load also SpaCy docs of the text
                    'spacy_model_name': 'ro_core_news_sm' # The name of the SpaCy model
                    'content_key': 'content' # Which JSON key contains the text
                    'title_key': 'title' # Which JSON key contains the title
                    'verbose': False # Should the parser print verbose output
                    'force': False # Should the parser overwrite already parsed data
                    'limit': None # The maximum number of files to parse
                }
        """
        self.path = 'data'
        self.use_spacy = False
        self.spacy_model_name = 'ro_core_news_sm'
        self.content_key = 'content'
        self.title_key = 'title'
        self.verbose = False
        self.force = False
        self.limit = None

        self.__entries: list[RoRoEntry] = []
        self.__entries_dirs = defaultdict(dict)
        self.__spacy_model = None
        self.__errors = []

        if options:
            self.set(options)

        if self.verbose:
            print("[info] Initialized parser")
        pass

    def set(self, options):
        for key, value in options.items():
            setattr(self, key, value)

        return self
    
    def parse(self):
        if self.__entries != [] and not self.force:
            print("[err] Parser already parsed data")
            return
        
        self.__entries = []
        self.__entries_dirs = defaultdict(dict)

        for dirs, rel_path, fp in self.__iter_target_files():
            if self.limit and len(self.__entries) >= self.limit:
                break

            text, meta = self.__safe_load_json(fp)

            meta.update({"rel_path": rel_path, "dirs": dirs})

            entry = RoRoEntry(text=text, meta=meta)

            self.__entries.append(entry)

            # Creating nested dict
            d = self.__entries_dirs
            for dir in dirs[:-1]:
                if dir not in d:
                    d[dir] = {}
                d = d[dir]

            d.setdefault(dirs[-1], {})[rel_path] = entry

        if self.verbose:
            print("[info] Parsed data")

        if self.__errors:
            print(f"[err] Errors: {self.__errors}")
        elif self.verbose:
            print("[info] No errors")

        if self.use_spacy:
            self.create_spacy_docs()

        if self.verbose:
            print("[info] Finished parsing")
            print(f"[info] Found {self.count_files()} entries")

        return self
    
    def head(self, limit = 2):
        """
        Print the first 'limit' number of parsed texts and their corresponding metadata

        :param limit: The number of texts to print. Defaults to 5
        :type limit: int
        """
        print(self.__entries[:limit])

    def count_files(self):
        return len(self.__entries)
    
    def get_flat(self):
        return self.__entries
    
    def get(self, query=None):
        """
        Get the parsed entries (or subfolders) from the dataset.

        - If query is None → return the whole tree (__entries_dirs)
        - If query is a string → interpret as path (e.g. "politics/europe")
        - If query is a list of paths → build a nested dict structure

        :param query: subfolder(s) to get
        :type query: None | str | list[str]
        :return: dict[str, RoRoEntry] or nested dict of subfolders, or None if not found
        """

        if not self.__entries:
            print("[err] Call parse() first")
            return None

        # Case 1: no query → return entire entries tree
        if query is None:
            return self.__entries_dirs

        # Case 2: single path string
        if isinstance(query, str):
            return self.__get_dir(query)

        # Case 3: list of queries → build nested dict
        if isinstance(query, list):
            result = {}
            for q in query:
                parts = q.split("/")
                d = result
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                d[parts[-1]] = self.__get_dir(q)
            return result

        print(f"[err] Unsupported query type: {type(query)}")
        return None


    def __get_dir(self, path: str):
        """
        Traverse __entries_dirs following a '/'-separated path.
        Example: "politics/europe"
        """
        parts = path.split("/")
        node = self.__entries_dirs
        for p in parts:
            if p not in node:
                print(f"[err] Subfolder {p} not found in {path}")
                return None
            node = node[p]
        return node


    def __iter_target_files(self):
        """
        Iterate over the .json files in the root directory and its subdirectories
        
        Each item yielded is a tuple of (dirs, rel_path, fp) where
        
        - dirs is a list of strings representing the subdirectories in the path
        - rel_path is the relative path of the file from the root directory
        - fp is the Path object of the file
        """
        root = Path(self.path)

        for fp in root.rglob('*.json'):
            rel_path = fp.relative_to(root)
            dirs = list(rel_path.parts[:-1])
            yield dirs, rel_path.as_posix(), fp

    def __safe_load_json(self, fp):
        """
        Safely load a JSON file and extract the text, title and metadata from it
        
        Parameters
        ----------
        fp : Path
            The path of the JSON file
        
        Returns
        -------
        text : str
            The text of the JSON file
        meta : dict
            A dictionary containing the metadata of the JSON file
        """
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
            text = obj.get(self.content_key, "")
            title = obj.get(self.title_key, "")
            meta = obj.get("metadata", {})
            
            meta.update({"title": title})

            if isinstance(text, str):
                return text, meta
            else:
                raise Exception("JSON content or title is not a string")
        except Exception as e:
            if self.verbose:
                print(f"[skip] {fp}: {e}")
            self.__errors.append((fp, e))
            return "", {}   
        
    def __load_spacy(self):
        """
        Load a SpaCy model, and add the parser and sentencizer if they do not exist
        
        If the model could not be loaded, an error message is printed and the exception is stored in self.__errors
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        if self.verbose:
            print("[info] Loading SpaCy")
        import spacy

        try:
            self.__spacy_model = spacy.load(self.spacy_model_name)
        except Exception as e:
            print(f"[err] Failed to load SpaCy model {self.spacy_model_name}: {e}")
            self.__errors.append((self.spacy_model_name, e))
            return
        
        if "parser" not in self.__spacy_model.pipe_names and "sentencizer" not in self.__spacy_model.pipe_names:
            if self.verbose:
                print("[info] Adding parser and sentencizer")
            self.__spacy_model.add_pipe("sentencizer")
        
        if self.verbose:
            print("[info] Loaded SpaCy")

    def create_spacy_docs(self):

        self.use_spacy = True

        if self.verbose:
            print("[info] Creating SpaCy docs")

        if self.__entries == []:
            print("[err] Call parse() first")
            return
        
        if self.__entries[0].doc != None and not self.force:
            print("[err] SpaCy docs already created")
            return
        
        if not self.__spacy_model:
            self.__load_spacy()

        texts = (e.text for e in self.__entries)
        for entry, doc in zip(self.__entries, self.__spacy_model.pipe(texts, batch_size=100, n_process=-1)):
            entry.doc = doc

        
        if self.verbose:
            print("[info] Created SpaCy docs")


        

    
