# Chinese Data Process

## Get raw data

```shell
curl -L -o part-0000.jsonl https://huggingface.co/datasets/CASIA-LM/ChineseWebText/resolve/main/cleaner_subset/2023-23/part-0000.jsonl
```

## Filter the Data

Use `filter.py` to get a subset of the raw data and convert them into `filtered_data.txt`

```shell
python filter.py
```

You can also change the target chars and min_score.

## Prepare the font

```shell
curl -L -O https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansSC.zip
unzip SourceHanSansSC.zip
rm SourceHanSansSC.zip
```

## Process to get the data

```shell
python process.py
```