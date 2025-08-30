from .parser import RoRoParser
from .analyzer import RoRoAnalyzer


if __name__ == "__main__":
    # use_spacy false ca sa se incarce cu batching.
    # daca folosim PC cu multa memorie si placa video, se poate marca use_spacy True si analizatoarele vor rula mai repede
    parser = RoRoParser({'path': 'data-cleaned', 'verbose': True, 'use_spacy': False, 'spacy_model_name': 'ro_core_news_sm'})

    parser.parse()

    analyzer = RoRoAnalyzer(parser)

    # Salvam un dataset statistic cu toate datele
    # Al doilea parametru este None deoarece nu facem un subquery
    analyzer.run('dataset_statistics', None) 

    analyzer.save_csv('dataset_statistics.csv')

    # Analiza pe Judete si Raioane
    # Vrem ca rezultatul sa includa regiunile, deci level = 1
    # Al treilea parametru este fals deoarece analizatorul nu trebuie sa primeasca un dict, ci un array flat
    # Level este un parametru **kwargs utilizat de sentence_stats pentru a decide la ce nivel sa faca analiza  
    analyzer.run('sentence_stats', ['judete', 'raioane'], False, level=1)

    analyzer.save_csv('RoMD_statistici_text.csv')

    analyzer.plot('RoMD_statistici_text')



