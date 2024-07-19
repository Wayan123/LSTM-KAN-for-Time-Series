## LSTM-KAN for Time Series in Energy Consumption Prediction

The following code is modified from the repository [https://github.com/iamirmasoud/energy_consumption_prediction](https://github.com/iamirmasoud/energy_consumption_prediction). The original code only uses GRU and LSTM. In contrast, the modified code adds KAN (Kolmogorov Arnold Networks) as an alternative to MLP, resulting in GRU-KAN and LSTM-KAN.

**Note:** The code was run on Kaggle. If you want to use Google Colaboratory, some lines of code for the dataset may need to be modified.

![Results](images/1.png)

### Final Results Analysis

The image above shows the energy consumption prediction results using four different methods: GRU, GRU-KAN, LSTM, and LSTM-KAN. The chart is divided into four sub-charts, each for a different energy dataset: DOM, EKPC, DUQ, and DAYTON.

### Results and Comparison

Experiments were conducted by testing the combinations of GRU-MLP, GRU-KAN, LSTM-MLP, and LSTM-KAN for time series prediction on the energy consumption dataset. After training each combination with 20 epochs, the results obtained are as follows:

- **GRU-MLP**: sMAPE = 0.23%
- **GRU-KAN**: sMAPE = 0.237%
- **LSTM-MLP**: sMAPE = 0.251%
- **LSTM-KAN**: sMAPE = 0.281%

From these results, the winner is **GRU-MLP** with the lowest sMAPE. The KAN alternative performed well but was still outperformed by the MLP combination.

### Conclusion

- **GRU-MLP** provided the best results with the lowest sMAPE (0.23%).
- **KAN** as an MLP alternative performed well but did not surpass the MLP combination.
- All models exhibited trends similar to the actual data, demonstrating their capability to capture energy consumption patterns.

With these results, GRU-MLP becomes the best choice for predicting energy consumption on the tested dataset. The KAN combination can still be considered an alternative, but it requires further development to improve accuracy.

Note: For different results, consider conducting more in-depth and measured experiments.
