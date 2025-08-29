from .parser import RoRoParser
from .analyzer import RoRoAnalyzer


if __name__ == "__main__":
    parser = RoRoParser({'path': 'data-cleaned', 'verbose': True, 'use_spacy': False, 'spacy_model_name': 'ro_core_news_sm'})

    parser.parse()

    analyzer = RoRoAnalyzer(parser)

    # Analiza pe toate datele
    result = analyzer.run('dataset_statistics', None, False)

    analyzer.save_csv('All.csv')

    # Analiza doar pe judete si raioane
    result = analyzer.run('dataset_statistics', ['judete', 'raioane'], False)

    analyzer.save_csv('RO_MD.csv')


    # Analiza doar pe Ardeal
    result = analyzer.run('dataset_statistics', 'judete/Ardeal_cleaned', False)

    analyzer.save_csv('Ardeal.csv')
