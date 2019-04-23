https://www.kaggle.com/c/gendered-pronoun-resolution

Code to finish 31/838

# Commands

```
conda env update -f=environment_gpu.yml
conda activate coref
python ensemble_train.py
python ensemble_eval.py
cd gap_coreference
python gap_scorer.py --gold_tsv gap-development.tsv --system_tsv ../results/gap-pred-scorer-development.tsv
```