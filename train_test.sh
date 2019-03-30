python main_ontonotes.py
python main.py --train
python main.py --test
cd gap_coreference
python gap_scorer.py --gold_tsv gap-development.tsv --system_tsv ../results/gap-pred-scorer-development.tsv