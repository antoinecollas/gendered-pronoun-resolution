https://www.kaggle.com/c/gendered-pronoun-resolution

conda env update -f=environment_gpu.yml
conda activate coref

python train.py
python test.py
cd gap_coreference
python gap_scorer.py --gold_tsv gap-test.tsv --system_tsv ../results/gap-pred-scorer-test.tsv