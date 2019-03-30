https://www.kaggle.com/c/gendered-pronoun-resolution

# Data - Ontonotes
- https://catalog.ldc.upenn.edu/LDC2013T19

```
tar xf ontonotes-release-5.0_LDC2013T19.tar
```

- https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO
```
git clone https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO
cd OntoNotes-5.0-NER-BIO
./conll-formatted-ontonotes-5.0/scripts/skeleton2conll.sh -D ../ontonotes-release-5.0/data/files/data ./conll-formatted-ontonotes-5.0
```

```
git clone https://github.com/jsalt18-sentence-repl/jiant.git
python extract_ontonotes_all.py --ontonotes ../gendered_pronoun_resolution/conll-formatted-ontonotes-5.0 \
  --tasks coref \
  --splits train development test conll-2012-test \
  -o ontonotes_jiant
```

# Commands
conda env update -f=environment_gpu.yml

conda activate coref

python main_ontonotes.py

python main.py --train

python main.py --test

cd gap_coreference

python gap_scorer.py --gold_tsv gap-development.tsv --system_tsv ../results/gap-pred-scorer-development.tsv