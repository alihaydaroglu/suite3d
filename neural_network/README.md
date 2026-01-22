## Neural network code


This folder contains some of the neural network-related code we experimented with. 
We ended up opting for PCA for the dimensionality reduction part of the curation pipeline, so this code was not used in the final method. We include it here in case anyone wants to experiment with nonlinear dimensionality reduction - this may give better curation results but will take a bit of work to get things up and running!

The idea would be to drop in a neural network's latent representations in place of the PCA embeddings currently used. 
There are a couple of different architectures and training functions provided (autoencoder, multi-channel CNN, contrastive learning, augmentation module, etc.).
