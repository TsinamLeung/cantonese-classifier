# mg-classifier

A Classical Chinese - Modern Chinese Language Classifier

## Prepare

```sh
wget https://dumps.wikimedia.org/zhwikisource/20220401/zhwikisource-20220401-pages-articles-multistream.xml.bz2
python -m wikiextractor.WikiExtractor zhwikisource-20220401-pages-articles-multistream.xml.bz2
```

## Train

```sh
python train.py
```
