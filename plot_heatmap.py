import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import data
from transformer import TransformerLM

STUDENT={'name': 'Eldar Hacohen',
         'ID': '311587661'}

# Load pre-trained model and tokenizer
data_path = "data/"
tokenizer, tokenized_data = data.load_data(data_path)
seq_len = 128
batch_size = 64
data_path = "data/"
n_layers = 6
n_heads = 6
embed_size = 192
mlp_hidden_size = embed_size * 4


MODEL_PATH = "model_shakespeare.pth"
learning_rate = 5e-4
gradient_clipping = 1.0

num_batches_to_train = 5000

# NOTE: are data items are longer by one than the sequence length,
# They will be shortened by 1 when converted to training examples.
data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

model: torch.nn.Module = TransformerLM(
    n_layers,
    n_heads,
    embed_size,
    seq_len,
    tokenizer.vocab_size(),
    mlp_hidden_size,
    with_residuals=True,
)

model.load(MODEL_PATH)
def plot_attention_heatmap(attention_matrices):
    num_layers = len(attention_matrices)
    num_heads = len(attention_matrices[0])

    plt.figure(figsize=(10, 8))

    for layer_num, layer_attention in enumerate(attention_matrices):
        for head_num, head_attention in enumerate(layer_attention):
            ax = plt.subplot(num_layers, num_heads, layer_num * num_heads + head_num + 1)
            sns.heatmap(head_attention.squeeze().cpu().detach().numpy(), cmap='inferno', xticklabels=False, yticklabels=False, ax=ax)
            ax.set_title(f'Layer {layer_num + 1} Head {head_num + 1}')

    plt.tight_layout()
    plt.show()


# Example usage:

def get_attention_matrices(model, samples):
    attention_matrices = []
    tensor_samples = torch.tensor(samples).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        model(tensor_samples)
        for layer in model.layers:
            layer_matrices = []
            attention_weights = layer.causal_attention.attention_weights
            for head in attention_weights:
                layer_matrices.append(head)
            attention_matrices.append(layer_matrices)
    return attention_matrices



# Function to generate attention heatmap
if __name__ == '__main__':
    text = "To be or not to be, that is the question"
    samples = tokenizer.tokenize(text)
    attention_matrices = get_attention_matrices(model, samples)

    # Plot the attention matrices for each sample
    plot_attention_heatmap(attention_matrices)
