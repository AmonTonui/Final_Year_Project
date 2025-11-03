# Final_Year_Project
# Predicting UK Elections Using Self-Organizing Maps (SOM)

<img width="452" height="400" alt="image" src="https://github.com/user-attachments/assets/83776c80-7bc1-4bef-924d-d015bbd029bd" />


## 🧭 Overview

This project explores **unsupervised learning** through the application of **Self-Organizing Maps (SOMs)** to uncover voting behaviour patterns in UK constituencies using **2021 Census demographic data** and **2024 General Election results**. By mapping multi-dimensional demographic data (age, education level, and overcrowding) into a 2D topological space, the SOM reveals underlying structures and clusters that align with political preferences.

The research demonstrates how SOMs can be used as a transparent, interpretable alternative to traditional black-box models (e.g., Random Forests, SVMs, or Neural Networks) for **election forecasting**, offering a data-driven way to understand why people vote as they do.

## 🎯 Objective

To determine whether demographic patterns such as age, education, and overcrowding can **predict voting behaviour** in UK elections and to uncover clusters of constituencies that share similar socio-political characteristics.

## ⚙️ Methodology

** CRISP-DM Data Mining Pipeline Was Followed:

<img width="452" height="406" alt="Picture 1" src="https://github.com/user-attachments/assets/aa31b70a-d878-4b6c-a211-cd7e40dfbe88" />

1. **Business Understanding**

    *Understanding the UK political landscape and the main demographics that influence elections.
2. **Data Understanding:**

   * 2021 UK Census demographic data.
   * 2024 General Election constituency results.
3. **Data Preparation:**

   * Normalised each constituency’s demographic proportions (rows = constituencies, columns = demographic features).
   * Filtered out irrelevant categories (e.g., under-18s) and non-major parties.
3. **Modelling:**

   * Implemented using the `MiniSom` library.
   * Hyperparameters tuned via **Bayesian Optimization** (learning rate, sigma, iterations).
   * Final model trained on full dataset (unsupervised learning → no need for data splitting).
4. **Visualization and Evaluation:**

   * Combined **U-Matrix** (distance) and **P-Matrix** (density) to create the **U*-Matrix** for enhanced interpretability.
   * Overlaid 2024 election results on the SOM to analyse **party dominance per cluster**.

---

## 🧩 Key Features

* **Transparent cluster discovery**: interpretable blue valleys (homogeneous demographics) and red ridges (boundaries).
* **Cluster purity analysis**: measured % of dominant party per cluster.
* **Feature profiling**: extracted mean feature values for neurons in homogeneous regions.
* **Predictive insight**: formulated heuristic rules for new constituencies based on BMU mapping.

---

## 📊 Results Summary

| Dataset          | Key Finding                                               | Cluster Purity | Dominant Party |
| ---------------- | --------------------------------------------------------- | -------------- | -------------- |
| Age Demographics | Younger populations (16–24) → Labour strongholds          | 100%           | Labour         |
| Education Level  | Low-to-mid qualifications (Levels 1–3) → Labour leaning   | 76%            | Labour         |
| People per Room  | Low overcrowding areas → Labour-leaning                   | 76.5%          | Labour         |
| Combined Dataset | High overcrowding + low education → clear Labour clusters | 100%           | Labour         |

---

## 🧠 Interpretation

The SOM successfully revealed **coherent demographic clusters** strongly aligned with party voting behaviour. For instance, valleys with younger or less-educated demographics corresponded with Labour-dominant regions. The U*-Matrix made these relationships visually interpretable, supporting the project hypothesis that demographic patterns can predict voting outcomes.

---

## 🔍 Example Visualization

<img width="452" height="400" alt="image" src="https://github.com/user-attachments/assets/d83e9648-f4db-443e-be66-261a7f2b8db3" />



> Blue = homogeneous demographic valleys, Red = cluster boundaries. Circles represent neurons coloured by dominant party.

---

## 💡 Technologies Used

* **Python**: Data preprocessing, SOM training, and visualization.
* **Libraries**: Pandas, NumPy, Matplotlib, MiniSom, SciPy.
* **Visualization**: Custom U*-Matrix implementation inspired by Ultsch & Mörchen (2005).

---

## 🚀 Future Work

* Integrate **real-time sentiment data** (e.g., Twitter via BERT/GPT-4) for hybrid models.
* Extend analysis to **2019 Election** once corresponding demographic data are matched.
* Develop a **web-based dashboard** for interactive exploration of SOM clusters.

---

## 📘 Citation

If you reference this work in academic writing:

> Tonui, A. K. (2025). *Predicting Elections Using Unsupervised Learning: A Self-Organizing Map Analysis of UK Demographics and Voting Behaviour.* Aston University.

---

## 🧑‍💻 Author

**Amon Kiprono Tonui**
BSc (Hons) Computer Science, Aston University
Supervised by Dr. James Borg
Email: [amontoe04@gmail.com](mailto:amontoe04@gmail.comm)

---

### 📎 Example Code Snippet

```python
# Assign Best Matching Units (BMUs)
winning_neurons = [som.winner(x) for x in normalized_features]
normalized_df['Neuron Row'] = [wn[0] for wn in winning_neurons]
normalized_df['Neuron Col'] = [wn[1] for wn in winning_neurons]

# Compute U*-Matrix
u_matrix = som.distance_map()
density_matrix = np.zeros((n_neurons, m_neurons))
for r, c in zip(normalized_df['Neuron Row'], normalized_df['Neuron Col']):
    density_matrix[r, c] += 1
density_matrix = median_filter(density_matrix, size=3)
scale_factor = (density_matrix - np.min(density_matrix)) / (np.mean(density_matrix) - np.min(density_matrix))
u_star_matrix = u_matrix * scale_factor
```

---

📈 *This repository illustrates how demographic-based Self-Organizing Maps can make political data interpretable, visual, and predictive — bridging machine learning and social science.*
