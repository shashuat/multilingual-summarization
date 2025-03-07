# Preprocessing for summary data
## 1. How to Use
To preprocess summaries, run the following command from the project root directory (specify preprocess_for_summary.py correctly):

```bash
python ./src/preprocess_for_summary.py \
    --summaries-dir ../summary_for_checking_preprocess \
    --languages ja
```

### Processing Multiple Languages
You can specify multiple languages in the `--languages` parameter:

```bash
python ./src/preprocess_for_summary.py \
    --summaries-dir ../summary_for_checking_preprocess \
    --languages ja de fr
```

### Saving Processed Output to a Different Directory
To avoid overwriting the original summaries, use the `--save-data-dir` parameter:

```bash
python ./src/preprocess_for_summary.py \
    --summaries-dir ../summary_for_checking_preprocess \
    --languages ja \
    --save-data-dir ../summary_deleted_after_newline
```

---

## 2. Checking the Behavior
### Prepare the Input Directory
Copy the original `summary` directory to a new directory for preprocessing:

```bash
kazetof@inspiron:~/Works/text_mining$ ls
class08  multilingual-summarization  multilingual-summarization.zip  summary  summary_deleted_after_newline  
kazetof@inspiron:~/Works/text_mining$ cp -r summary summary_for_checking_preprocess
```

#### Checking a File Before Preprocessing

```bash
kazetof@inspiron:~/Works/text_mining/summary_for_checking_preprocess/ja$ cat d5d8261292.json
{
  "article_id": "d5d8261292",
  "language": "ja",
  "summary": "ラクタリウス・インディゴ（Lactarius indigo）は、ベニタケ科のキノコで、北アメリカ東部、東アジア 、中央アメリカに分布しています。日本では希少です。このキノコは、新しい時は暗青色で、古いと淡黄緑色になります 。特徴的なのは、傷つけるとインディゴブルーの乳液が出ることです。傘の直径は5〜12cm、高さは2〜8cmで、食用としても知られています。このキノコは1822年に記述され、後にLactarius属に分類されました。科学者たちは、このキノコが青色の乳液を持つ特性から、進化の過程を研究する価値があると指摘しています。種小名のindigoは「インディゴブルー」 から由来します。このキノコは英語ではindigo milk capとも呼ばれています。 \n\n(Note: I have translated the summary into English to check its accuracy and comprehensiveness, as per your instruction.) \n\nEnglish translation: \nLactarius indigo is a Russulaceae mushroom found in eastern North America, East Asia, and Central America. It is rare in Japan. This mushroom is dark blue when fresh and becomes pale green when old. Its distinctive feature is that it secretes indigo blue latex when damaged. The cap diameter ranges from 5-12 cm, and the height is 2-8 cm, and it is also known as edible. This mushroom was described in 1822 and later classified under the genus Lactarius. Scientists believe that this mushroom's characteristic of having blue latex is worth researching for the study of evolutionary processes. The specific epithet 'indigo' comes from \"indigo blue.\" This mushroom is also called \"indigo milk cap\" in English. \n\n(The Japanese summary provided above covers all key points and main arguments of the article without unnecessary details.)",
  "title": "ラクタリウス・インディゴ"
```

---

## Running the Preprocessing Script
Run the following command to process the summaries:

```bash
python ./src/preprocess_for_summary.py \
    --summaries-dir ../summary_for_checking_preprocess \
    --languages ja de fr
```

```bash
kazetof@inspiron:~/Works/text_mining/summary_for_checking_preprocess/ja$ pwd
/home/kazetof/Works/text_mining/summary_for_checking_preprocess/ja
```

#### Checking a Processed File

```bash
kazetof@inspiron:~/Works/text_mining/summary_for_checking_preprocess/ja$ cat d5d8261292.json
{
    "article_id": "d5d8261292",
    "language": "ja",
    "summary": "ラクタリウス・インディゴ（Lactarius indigo）は、ベニタケ科のキノコで、北アメリカ東部、東アジ ア、中央アメリカに分布しています。日本では希少です。このキノコは、新しい時は暗青色で、古いと淡黄緑色になりま す。特徴的なのは、傷つけるとインディゴブルーの乳液が出ることです。傘の直径は5〜12cm、高さは2〜8cmで、食用としても知られています。このキノコは1822年に記述され、後にLactarius属に分類されました。科学者たちは、このキノコが青色の乳液を持つ特性から、進化の過程を研究する価値があると指摘しています。種小名のindigoは「インディゴブルー 」から由来します。このキノコは英語ではindigo milk capとも呼ばれています。 ",
    "title": "ラクタリウス・インディゴ"
```

```bash
kazetof@inspiron:~/Works/text_mining/summary_for_checking_preprocess/ja$ ls -lah | wc -l
1870
kazetof@inspiron:~/Works/text_mining/summary/ja$ ls -lah | wc -l
1870
```