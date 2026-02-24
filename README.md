# PyTorch RNN: From-Scratch Recurrent Architectures 

This repo was developed as part of my undergraduate degree in Data Science and Business Analytics at the UAS St Pölten.

### Overview: "From Scratch" Implementation
This project demonstrates a deep understanding of sequential modeling by bypassing high-level abstractions like `nn.GRU` or `nn.LSTM`. Instead, the models are built using **complex layers from scratch** to reveal the underlying mechanics of recurrence.

* **Custom RNN Cells:** Manual implementation of the hidden state transition: 
    $$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$
* **State Management:** Explicitly handling hidden state initialization and propagation across time steps.
* **Framework:** Built with **PyTorch**, utilizing Tensors and Autograd for backpropagation through time (BPTT).



### Technical Highlights
* **Layer Architecture:** Designed custom linear transformations for input-to-hidden and hidden-to-hidden transitions.
* **Mathematics:** Practical application of activation functions and weight matrices in a recursive loop.
* **Optimization:** Focused on gradient flow and training stability in basic recurrent structures.
