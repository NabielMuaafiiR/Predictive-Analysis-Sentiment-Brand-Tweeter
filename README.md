# Laporan Proyek Machine Learning - Nabiel Muaafii Rahman

## Domain Proyek

Dalam era digital saat ini, media sosial telah menjadi salah satu platform utama bagi konsumen untuk mengekspresikan opini dan perasaan mereka terhadap suatu merek. Twitter, dengan format pesannya yang singkat dan cepat, memungkinkan penyebaran opini secara luas dalam waktu singkat. Hal ini menjadikan Twitter sebagai sumber data yang kaya untuk melakukan analisis sentimen terhadap merek.

Analisis sentimen di Twitter memberikan wawasan yang bernilai bagi perusahaan dalam memahami persepsi publik, mengidentifikasi potensi masalah reputasi, serta merespons umpan balik pelanggan secara lebih proaktif. Studi oleh Hu et al. (2017) menunjukkan bahwa analisis terhadap 330 juta tweet dapat mengungkap sentimen pengguna terhadap berbagai industri dan merek, yang pada akhirnya dapat membantu perusahaan dalam menyusun strategi pemasaran dan pengembangan produk yang lebih tepat sasaran.

Proyek ini bertujuan untuk mengevaluasi sejauh mana model prediktif Naive Bayes mampu melakukan klasifikasi sentimen Positive, Negative, dan Neutral, terhadap tweet yang berkaitan dengan merek (brand) yang sedang banyak diperbincangkan di media sosial, khususnya Twitter. Dengan adanya model ini, diharapkan sistem dapat secara otomatis mengidentifikasi apakah suatu tweet mengandung sentimen positif, negatif, atau netral terhadap suatu merek. Hasil klasifikasi ini dapat dimanfaatkan oleh perusahaan untuk memantau persepsi publik, meningkatkan kualitas layanan, serta merancang strategi komunikasi dan pemasaran yang lebih efektif dan responsif.

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Seberapa akurat model Naive Bayes dapat melakukan klasifikasi sentimen?
- Bagaimana performa model Naive Bayes dibandingkan pada matriks evaluasi accuracy dan F1-score?
- Apakah model Naive Bayes layak untuk digunakan analisis klasifikasi sentimen?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membangun model klasifikasi sentimen dengan Naive Bayes.
- Mengevaluasi performa model Naive Bayes menggunakan matriks evaluasi accuracy dan F1-score.
- Memberikan gambaran dan penjelasan terhadap kelayakan model dalam melakukan klasifikasi sentimen.

## Data Understanding
Dataset yang diambil pada projek ini berasal dari kaggle https://www.kaggle.com/datasets/tusharpaul2001/brand-sentiment-analysis-dataset . Data terdiri dari 8589 baris dan 3 kolom: tweet_text (text tweet), emotion_in_tweet_is_directed_at (brand apa yang dibahas), dan is_there_an_emotion_directed_at_a_brand_or_product (label sentimen tweet). Dataset semua bertipe string.

### Tipe Data
**(gambar tipe data)**

### Deskripsi Variabel
Variabel  |	Keterangan
-----------|------------
tweet_text  |	Text berisi tweet
emotion_in_tweet_is_directed_at  |	Brand yang dibahas pada tweet 
is_there_an_emotion_directed_at_a_brand_or_product | Label sentimen tweet

### Visualisasi Data EDA

**(gambar sebaran sentimen)**

Interpretasi:
Data yang digunakan dalam proyek ini memiliki tingkat ketidakseimbangan (imbalanced class distribution) yang cukup tinggi. Sentimen positif mendominasi secara signifikan dibandingkan sentimen negatif, sementara jumlah data dengan sentimen netral sangat sedikit. Ketidakseimbangan ini berpotensi memengaruhi kinerja model dalam mengenali kelas minoritas, sehingga perlu diperhatikan dalam proses pelatihan dan evaluasi model.

**(gambar sebaran brand)**

Interpretasi:
Sebaran brand yang dibahas dalam data sentimen analisis tweeter ini merujuk pada brand teknologi ternama, seperti apple dan google. Sebaran paling banyak terdapat pada iPad yang merupakan salah satu produk apple. Hampir secara keseluruhan brand yang dibahas adalah seputar teknologi apple.

**(gambar sebaran sentimen tiap brand)**

Interpretasi:
Sebaran sentimen tiap brand merujuk kepada positif semua. Hal ini dikarenakan terdapat banyak missing value yang tertera pada data asli, sehingga membuat sentimen lain yang kosong menjadi terhapus.

**(gambar wordcloud)**

Interpretasi:
Kata yang paling sering muncul adalah sxsw, yang sepertinya kalau dilihat dari datanya itu merupakan sebuah tagar suatu acara. Disisi lain produk seperti ipad, iphone, dan google sering muncul dalam data tweeter ini.

## Data Preparation
### Menangani Missing Value dan Duplikasi Data
Pada tahap ini, dataset diperiksa untuk memastikan tidak ada nilai yang hilang (missing values) Berdasarkan analisis awal:

**(gambar missing value)**

Terdapat banyak sekali missing value pada Kolom emotion_in_tweet_is_directed_at (brand yang dibahas), hal ini dilakukan yang namanya penghapusan missing value sekaligus menghapus duplikasi data, sehingga data menjadi seperti ini:

**(gambar setelah dihapus)**

### Dilakukan pre-processing text meliputi berbagai tahapan berikut:
```
def cleaning(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove mention first before punctuation
    text = re.sub(r"@[\w_]+", "", text) # Remove mention

    # 3. Remove numbers
    text = re.sub(r"\b\d+\b", "", text)

    # 4. Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)

    # 5. Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    
    # 6. Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    # 7. Remove newlines and some extra characters
    text = re.sub('\n', '', text)
    text = re.sub('[’“”…]', '', text)
    
    
    # 8. Remove RT and mention literal (optional)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r'\bmention\b', '', text)
    
    # 9. Remove emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)


    # 10. Fix contractions
    text = contractions.fix(text)

    # 11. Remove stopwords
    stop_list = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop_list])

    # 12. Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

df_clean['tweet_text'] = df_clean['tweet_text'].apply(cleaning)
```

**(gambar hasil cleaning)**

Pre-processing data pada text perlu dilakukan guna membantu model dalam memahami tiap karakter yang ada
1. Cleansing adalah tahap dimana karakter dan tanda baca yang tidak diperlukan dihapus dari teks. Bekerja
untuk mengurangi noise pada dataset (Saputra I, _et al._ 2021). Pada tahap ini dilakukan penghapusan mention, url, angka, html tag, tanda baca, newline, dan tanda baca yang tidak diperlukan, menghapus RT (retweet), dan terakhir emoji.
2. Case folding, atau merubah huruf menjadi tidak kapital semua. Hal ini bertujuan untuk membuat semua kata kapital dan tidak kapital menjadi sama.
3. Contractions adalah memperjelas kata, seperti "whats" menjadi "what is".
4. Mengapus stopwords. Alasan menghapus kata yang terkait dengan penambangan teks adalah karena penggunaannya yang.
terlalu umum, sehingga pengguna dapat fokus pada kata-kata lain yang jauh lebih penting.
5. Lemmatization atau merubah kata kerja menjadi kata dasar seperti "playing" menjadi "play".

### Melakukan encoding label:
```
df_clean['is_there_an_emotion_directed_at_a_brand_or_product'].replace({"Positive emotion": 1,
                                                                        "Negative emotion": 0,
                                                                        "No emotion toward brand or product": 2,
                                                                        "I can't tell": 2}, inplace=True)
df_clean.head()
```
**(gambar setelah di encode)**

Hal ini dilakukan karna model hanya dapat mengenai angka sepagai representasi kelas.

### Membuang kolom 'emotion_in_tweet_is_directed_at'(brand)
```
df_clean.drop('emotion_in_tweet_is_directed_at', axis=1, inplace=True)
```
Kolom tersebut hanya sebuah label brand apa yang dibahas diluar itu tidak ada pengaruh terhadap model.

### Embedding Text/Pembobotan Kata
```
tfidf = TfidfVectorizer()
text_count = tfidf.fit_transform(df_clean['tweet_text'])
```
Pembobotan kata adalah suatu mekanisme untuk memberikan skor terhadap frekuensi kemunculan sebuah kata dalam dokumen teks. pada projek ini digunakan metode TfidfVerctorizer dengan parameter default.

### Splitting Data
```
X=text_count
y=df_clean['is_there_an_emotion_directed_at_a_brand_or_product']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42, stratify=y)
```
Sebelum data dimasukkan kedalam model, data dibagi menjadi dua bagian, yaitu data latih dan data uji, dengan proporsi 80% untuk data latih dan 20% untuk data uji. Pembagian ini bertujuan agar model dapat melakukan analisis dan pembelajaran tanpa mengalami kebocoran data (data leakage), sehingga hasil evaluasi model menjadi lebih objektif dan dapat dipercaya.

## Modeling
Model yang digunakan hanya Naive Bayes. Menurut Annur 2018 Metode Bayes merupakan pendekatan statistic untuk melakukan inferensi induksi pada persoalan klasifikasi. Teorema bayes memiliki bentuk umum sebagai berikut:

**(gambar rumus bayes)**

Keterangan :
X = Data dengan class yang belum diketahui
H = Hipotesis data X merupakan suatu class spesifik
P(H|X) = Probabilitas hipotesis H berdasarkan kondisi x (posteriori prob.)
P(H) = Probabilitas hipotesis H (prior prob.)
P(X|H) = Probabilitas X berdasarkan kondisi tersebut
P(X) = Probabilitas dari X

Pada percobaan pertama dilakukan fitting data dengan parameter default
```
cnb = ComplementNB()
cnb.fit(X_train, y_train)
```

Pada percobaan kedua dilakukan GridSearchCV dengan parameter sebagai berikut:
```
param_dist = {
    'alpha': np.arange(0,1,0.1),
    'norm': [True, False]
}
```
```
scorer = make_scorer(f1_score, average='macro')

cnb = ComplementNB()

grid_search = GridSearchCV(
    estimator=cnb,
    param_grid=param_dist,
    scoring=scorer,
    cv=5,
    verbose=1,
    n_jobs=-1
)
```
```
grid_search.fit(X_train, y_train)
```

## Evaluation

### Penjelasan Matriks
Matriks evaluasi yang digunakan antara lain **akurasi, precision, recall, dan F1-score**, tetapi pada projek ini lebih difokuskan pada akurasi dan F1-score.

#### 1. Akurasi
   
Akurasi adalah metrik evaluasi yang mengukur proporsi prediksi yang benar terhadap seluruh jumlah data.

Rumus:

**(gambar rumus akurasi)**

Keterangan:
TP: True Positive (positif yang diprediksi benar)

TN: True Negative (negatif yang diprediksi benar)

FP: False Positive (negatif yang diprediksi positif)

FN: False Negative (positif yang diprediksi negatif)

Kelemahan:
Akurasi tidak cocok untuk dataset yang tidak seimbang, karena bisa memberikan skor tinggi hanya dengan memprediksi mayoritas kelas.

#### 2. F1-Score
   
F1-score adalah metrik harmonis antara precision dan recall. F1 digunakan untuk mengevaluasi model terutama ketika data tidak seimbang, karena mempertimbangkan kesalahan pada prediksi kelas minoritas.

Rumus:
F1-score

**(gambar rumus f1)**
 
Precision:

 **(gambar rumus precision)**
 
Mengukur berapa banyak prediksi positif yang benar.

Recall:

 **(gambar rumus recall)**
 
Mengukur berapa banyak dari total data positif yang berhasil dikenali dengan benar.

Keunggulan:
- F1-score tidak terpengaruh oleh distribusi kelas.

Sangat cocok jika:

- False positive dan false negative sama-sama penting.

- Kelas target (misal: positif) jumlahnya jauh lebih sedikit daripada kelas lain.

### Hasil Evaluasi

#### 1. Parameter Default
Pada pengujian pertama dengan parameter default didapatkan akurasi sebesar 80.21% dan F1-score sebesar 0.4845679012345679 dengan detail confusion matriks sebagai berikut:

ComplementNB model accuracy is 80.21%

Confusion Matrix:
|     |  0  |  1  |  2  |
|-----|-----|-----|-----|
|  0  | 44  | 52  |  8  |
|  1  | 28  | 481 | 24  |
|  2  |  4  | 14  |  2  |

Keterangan: 0=Negative, 1=Positif, 2=Netral

Classification Report:
```
              precision    recall  f1-score   support

           0       0.58      0.42      0.49       104
           1       0.88      0.90      0.89       533
           2       0.06      0.10      0.07        20

    accuracy                           0.80       657
   macro avg       0.51      0.48      0.48       657
weighted avg       0.81      0.80      0.80       657
```
Data test digunakan sebanyak 657, dimana 533 diantaranya adalah sentimen positif atau sebesar 78.9%. Model dapat dengan mudah mencapai nilai akurasi 78.9% hanya dengan menebak semua tweet adalah positif.


#### 2. GridSearchCV
Setelah dilakukan GridSearchCV hasil evaluasi model mengalami penurunan dimana akurasi menjadi 73.97%, dan F1-score sebesar 0.4817721924914634, untuk detail cpnfusion matriks sebagai berikut:

ComplementNB model accuracy is 73.97%
Confusion Matrix:
|     |  0  |  1  |  2  |
|-----|-----|-----|-----|
|  0  | 58  | 32  |  14  |
|  1  | 45  | 425 | 63  |
|  2  |  6  | 11  |  3  |

Keterangan: 0=Negative, 1=Positif, 2=Netral

Classification Report:
```
              precision    recall  f1-score   support

           0       0.53      0.56      0.54       104
           1       0.91      0.80      0.85       533
           2       0.04      0.15      0.06        20

    accuracy                           0.74       657
   macro avg       0.49      0.50      0.48       657
weighted avg       0.82      0.74      0.78       657
```
## Inference
Model diuji menggunakan teks yang dibuat secara acak, di mana masing-masing teks secara eksplisit merepresentasikan sentimen yang berbeda—teks pertama bersentimen positif, dan teks kedua bersentimen negatif. Selain itu, disiapkan sebuah fungsi inferensi untuk memprediksi sentimen secara langsung berdasarkan input teks tersebut.
```
text1 = "Apple has a good technology"
text2 = "Ipad is very useless"
```
```
def inference(text):
  text = cleaning(text)
  text = tfidf.transform([text])
  prediction = cnb.predict(text)

  if prediction == 1:
    return 'Positive emotion'
  elif prediction == 0:
    return 'Negative emotion'
  else:
    return 'No emotion toward brand or product'
```

Hasil Prediksi:
```
Text 1: Positive emotion
Text 2: Negative emotion
```
Model dapat memprediksi sentimen sesuai yang dicantumkan secara eksplisit, walaupun sederhana tetapi model berhasil mengklasifikasikannya.


## Kesimpulan
Model Naive Bayes yang digunakan untuk melakukan klasifikasi sentimen terhadap tweet suatu brand berhasil mencapai akurasi sebesar 80,21%, dan setelah dilakukan inference model berhasil mengklasifikasikan sesuai dengan sentimen yang dicantumkan secara eksplisit. Namun, angka ini belum mencerminkan performa model yang baik secara menyeluruh, karena terdapat ketidakseimbangan kelas dalam data. Pada data uji, terdapat 657 tweet, di mana 78,9% di antaranya merupakan tweet dengan sentimen positif. Hal ini menunjukkan bahwa dengan hanya memprediksi semua tweet sebagai positif, model sudah dapat mencapai akurasi sebesar 78,9%.

Oleh karena itu, dilakukan pengukuran tambahan menggunakan F1-score untuk mendapatkan gambaran yang lebih seimbang terhadap performa model, khususnya dalam menangani kelas minoritas. Hasil evaluasi menunjukkan bahwa F1-score model hanya sebesar 0.485, yang berarti kemampuan model dalam mengklasifikasi sentimen masih cukup rendah. Bahkan, setelah dilakukan optimasi menggunakan GridSearchCV, performa model mengalami penurunan, yang mengindikasikan bahwa model ini belum optimal untuk menangani data yang tidak seimbang.

## Saran
1. Melakukan oversampling pada kelas minoritas, seperti menggunakan teknik SMOTE atau metode oversampling lainnya, untuk mengatasi ketidakseimbangan data dan meningkatkan kemampuan model dalam mengenali semua kelas secara adil.
2. Menggunakan model yang lebih kompleks, seperti model berbasis deep learning, transfer learning (contohnya BERT), atau algoritma lain yang lebih mampu menangani data teks dan ketidakseimbangan kelas dengan lebih baik.
3. Menerapkan teknik evaluasi yang lebih beragam, seperti confusion matrix per kelas, macro/micro averaging, dan visualisasi performa model, untuk memberikan pemahaman lebih menyeluruh terhadap hasil klasifikasi.

## Refrensi
Hu, G., Bhargava, P., Fuhrmann, S., Ellinger, S., & Spasojevic, N. (2017, November). Analyzing users’ sentiment towards popular consumer industries and brands on twitter. In 2017 IEEE International conference on Data mining workshops (ICDMW) (pp. 381-388). IEEE. https://ieeexplore.ieee.org/abstract/document/8215687/

Saputra, I., Darono, H. E., Amsury, F., Fahdia, M. R., Ramadhan, B., & Ardiansyah, A. (2021). Analisis Sentimen Pengguna Marketplace Bukalapak dan Tokopedia di Twitter Menggunakan Machine Learning. Faktor Exacta, 13(4), 200. https://www.academia.edu/download/100636852/3723.pdf

Hadna, N. M. S., Santosa, P. I., & Winarno, W. W. (2016). Studi literatur tentang perbandingan metode untuk proses analisis sentimen di Twitter. Semin. Nas. Teknol. Inf. dan Komun, 2016, 57-64. https://www.researchgate.net/profile/Nurrun-Muchammad-Hadna/publication/292831965_Studi_Literatur_Tentang_Perbandingan_Metode_Untuk_Proses_Analisis_Sentimen_di_Twitter/links/56b182ec08ae5ec4ed4895b1/Studi-Literatur-Tentang-Perbandingan-Metode-Untuk-Proses-Analisis-Sentimen-di-Twitter.pdf

Dicoding. (2024). Machine Learning Terapan. Diakses pada 25 Mei 2025 dari https://www.dicoding.com/academies/319-machine-learning-terapan.

Annur, H. (2018). Klasifikasi Masyarakat miskin menggunakan metode naïve bayes. ILKOM Jurnal Ilmiah, 10(2), 160-165. https://jurnal.fikom.umi.ac.id/index.php/ILKOM/article/view/303
