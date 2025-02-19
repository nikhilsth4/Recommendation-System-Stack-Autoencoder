# Stacked Autoencoder Recommendation System

This repository contains a Jupyter Notebook that implements a movie recommendation system using a **Stacked Autoencoder (SAE)** built with **PyTorch**. The system leverages collaborative filtering by learning latent user preferences to predict unseen movie ratings and recommend movies accordingly.

## Project Overview

The project uses a deep learning approach to model user–movie interactions. By training a Stacked Autoencoder, the system learns to reconstruct user ratings and infer the ratings for movies not yet rated by a user.

### Key Features

- **Collaborative Filtering:** Learns user preferences based on historical ratings.
- **Stacked Autoencoder Architecture:** Uses multiple hidden layers to capture latent features.
- **PyTorch Implementation:** Utilizes PyTorch for building, training, and evaluating the model.
- **MovieLens Dataset:** Uses the MovieLens 1M and 100K datasets for training and testing.

## Dataset

The system uses the MovieLens datasets:
- **MovieLens 1M:** For loading movie, user, and rating data.
- **MovieLens 100K:** For training and testing the recommendation model.

The data is preprocessed to form a user–movie rating matrix where each row corresponds to a user and each column to a movie. Missing ratings are set to zero.

## Model Architecture

The Stacked Autoencoder (SAE) model consists of:
- **Input Layer:** Size equals the number of movies.
- **Encoding Layers:** 
  - First hidden layer with 20 neurons.
  - Second hidden layer with 10 neurons.
- **Decoding Layers:**
  - First decoding layer with 20 neurons.
  - Output layer with a number of neurons equal to the number of movies.
- **Activation Function:** Sigmoid is used between layers.
- **Loss Function:** Mean Squared Error (MSE) is used for reconstruction loss.
- **Optimizer:** RMSprop with weight decay is used for optimization.

## Notebook Walkthrough

The Jupyter Notebook is structured as follows:

1. **Importing Libraries:**  
   Import necessary libraries including `numpy`, `pandas`, and PyTorch modules.

2. **Data Import and Preprocessing:**  
   Load the MovieLens datasets, convert the raw data into a user–movie matrix, and split it into training and testing sets.

3. **Converting Data into Tensors:**  
   Transform the data matrices into PyTorch tensors for model compatibility.

4. **Defining the SAE Model:**  
   Create the autoencoder architecture using fully connected layers with sigmoid activation.

5. **Training the Model:**  
   Train the autoencoder over multiple epochs. The notebook prints out the training loss for each epoch.

6. **Testing the Model:**  
   Evaluate the autoencoder on the test set and print the test loss.

7. **Generating Recommendations:**  
   For a selected user, predict the ratings for all movies, mask the movies that have already been rated, and output the top 10 recommended movie indices.

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/)
- numpy
- pandas
- Jupyter Notebook (or Google Colab)

Install the required Python packages using:

```bash
pip install numpy pandas torch jupyter
```

## How to Run

### Clone the Repository:

```bash
git clone https://github.com/nikhilsth4/stacked-autoencoder-recommendation.git
cd stacked-autoencoder-recommendation
```

###Launch the Jupyter Notebook:
```bash
jupyter notebook Stacked_Autoencoder_Recommendation.ipynb
```
### Run the Notebook Cells:

Execute the cells sequentially to:

- Import data  
- Train the model  
- Test the model  
- Generate movie recommendations  

### Results  

- **Training:** The notebook logs the training loss per epoch.  
- **Testing:** The test loss is computed and printed after model evaluation.  
- **Recommendations:** The notebook prints out the indices of the top 10 movie recommendations for a selected user.  

### Acknowledgements  

- **MovieLens:** Thanks to MovieLens for providing the dataset.  
- **PyTorch:** The deep learning framework used to build and train the model.  




