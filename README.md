# Laporan Proyek Machine Learning - Nabiel Muaafii Rahman

## Domain Proyek

Dalam era digital saat ini, media sosial telah menjadi salah satu platform utama bagi konsumen untuk mengekspresikan opini dan perasaan mereka terhadap suatu merek. Twitter, dengan format pesannya yang singkat dan cepat, memungkinkan penyebaran opini secara luas dalam waktu singkat. Hal ini menjadikan Twitter sebagai sumber data yang kaya untuk melakukan analisis sentimen terhadap merek.

Analisis sentimen di Twitter memberikan wawasan yang bernilai bagi perusahaan dalam memahami persepsi publik, mengidentifikasi potensi masalah reputasi, serta merespons umpan balik pelanggan secara lebih proaktif. Studi oleh Hu et al. (2017) menunjukkan bahwa analisis terhadap 330 juta tweet dapat mengungkap sentimen pengguna terhadap berbagai industri dan merek, yang pada akhirnya dapat membantu perusahaan dalam menyusun strategi pemasaran dan pengembangan produk yang lebih tepat sasaran.

Proyek ini bertujuan untuk mengevaluasi sejauh mana model prediktif Naive Bayes mampu melakukan klasifikasi sentimen Positive, Negative, dan Neutral, terhadap tweet yang berkaitan dengan merek (brand) yang sedang banyak diperbincangkan di media sosial, khususnya Twitter. Dengan adanya model ini, diharapkan sistem dapat secara otomatis mengidentifikasi apakah suatu tweet mengandung sentimen positif, negatif, atau netral terhadap suatu merek. Hasil klasifikasi ini dapat dimanfaatkan oleh perusahaan untuk memantau persepsi publik, meningkatkan kualitas layanan, serta merancang strategi komunikasi dan pemasaran yang lebih efektif dan responsif.

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Seberapa akurat model Naive Bayes dapat melakukan klasifikasi sentimen?
- Bagaimana performa model Naive Bayes dibandingkan pada matriks evaluasi accuracy dan F1-score?
- Apakah model Naive Bayes layak untuk digunakan analisis klasifikasi sentimen?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membangun model klasifikasi sentimen dengan Naive Bayes
- Mengevaluasi performa model Naive Bayes menggunakan matriks evaluasi accuracy dan F1-score
- Memberikan gambaran dan penjelasan terhadap kelayakan model dalam melakukan klasifikasi sentimen

## Data Understanding
Dataset yang diambil pada projek ini berasal dari kaggle https://www.kaggle.com/datasets/tusharpaul2001/brand-sentiment-analysis-dataset . Data terdiri dari 8589 baris dan 3 kolom: tweet_text (text tweet), emotion_in_tweet_is_directed_at (brand apa yang dibahas), dan is_there_an_emotion_directed_at_a_brand_or_product (label sentimen tweet). Dataset semua bertipe string.

**Tipe Data**

**Deskripsi Variabel**
Variabel  |	Keterangan
-----------|------------
tweet_text  |	Text berisi tweet
emotion_in_tweet_is_directed_at  |	Brand yang dibahas pada tweet 
is_there_an_emotion_directed_at_a_brand_or_product | Label sentimen tweet

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

## Refrensi
Hu, G., Bhargava, P., Fuhrmann, S., Ellinger, S., & Spasojevic, N. (2017, November). Analyzing users’ sentiment towards popular consumer industries and brands on twitter. In 2017 IEEE International conference on Data mining workshops (ICDMW) (pp. 381-388). IEEE.
