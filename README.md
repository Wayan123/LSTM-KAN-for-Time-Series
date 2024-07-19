## LSTM-KAN for Time Series pada Prediksi Konsumsi Energi

Kode berikut dimodifikasi dari repository [https://github.com/iamirmasoud/energy_consumption_prediction](https://github.com/iamirmasoud/energy_consumption_prediction). Kode awal hanya menggunakan GRU dan LSTM, sedangkan kode yang telah saya modifikasi ditambahkan KAN (Kolmogorov Arnold Networks) sebagai alternatif MLP menjadi GRU-KAN dan LSTM-KAN.

**Note:** Kode dijalankan pada Kaggle. Jika ingin menggunakan Google Colaboratory, beberapa baris kode untuk dataset mungkin perlu dimodifikasi.

![Results](images/1.png)

### Analisis Hasil Akhir

Gambar di atas menunjukkan hasil prediksi konsumsi energi dengan menggunakan empat metode berbeda: GRU, GRU-KAN, LSTM, dan LSTM-KAN. Grafik dibagi menjadi empat subgrafik, masing-masing untuk dataset energi yang berbeda: DOM, EKPC, DUQ, dan DAYTON.

### Hasil dan Perbandingan

Eksperimen dilakukan dengan menguji kombinasi GRU-MLP, GRU-KAN, LSTM-MLP, dan LSTM-KAN pada prediksi time series untuk dataset konsumsi energi. Setelah melakukan pelatihan masing-masing kombinasi dengan 20 epoch, hasil yang didapatkan adalah sebagai berikut:

- **GRU-MLP**: sMAPE = 0.23%
- **GRU-KAN**: sMAPE = 0.237%
- **LSTM-MLP**: sMAPE = 0.251%
- **LSTM-KAN**: sMAPE = 0.281%

Dari hasil tersebut, pemenangnya adalah **GRU-MLP** dengan nilai sMAPE terendah. Alternatif KAN cukup baik, namun masih kalah dengan kombinasi MLP. 

### Kesimpulan

- **GRU-MLP** memberikan hasil terbaik dengan sMAPE terendah (0.23%).
- **KAN** sebagai alternatif MLP menunjukkan performa yang cukup baik, namun belum mengungguli kombinasi dengan MLP.
- Semua model menunjukkan tren yang mirip dengan data aktual, menunjukkan kemampuan mereka dalam menangkap pola konsumsi energi.

Dengan hasil ini, GRU-MLP menjadi pilihan terbaik untuk prediksi konsumsi energi pada dataset yang diuji. Kombinasi KAN masih dapat dipertimbangkan sebagai alternatif, namun perlu pengembangan lebih lanjut untuk meningkatkan akurasi.
