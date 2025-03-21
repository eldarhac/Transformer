from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

STUDENT={'name': 'Eldar Hacohen',
         'ID': '311587661'}

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals
        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.2)

    def forward(self, inputs):
        if self.with_residuals:
            x = inputs
            x = self.layer_norm_1(x)
            x = self.causal_attention(x)
            x_res = x + inputs
            x_res = self.dropout_1(x_res)
            x = self.layer_norm_2(x_res)
            x = self.mlp(x)
            x = x + x_res
            x = self.dropout_2(x)
            return x
        else:
            x = inputs
            x = self.layer_norm_1(x)
            x = self.causal_attention(x)
            x = self.dropout_1(x)
            x = self.layer_norm_2(x)
            x = self.mlp(x)
            x = self.dropout_2(x)
            return x


class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x):
        # x has the shape (b x n) where b is batch dimension and n is sequence length.
        # each item is an int, indicating a vocabulary item.
        # The output should be of shape (b x n x d), where d is the embedding dimension.
        device = x.device
        tok_embeddings = self.token_embeddings(x)
        pos_embeddings = self.position_embeddings(torch.arange(x.size(1)).to(device))
        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len
        self.gelu = nn.GELU()
        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        logits = self.gelu(logits)
        return logits

    def init_weights(self):
        # initialize weights
        self.embed.token_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.embed.position_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.word_prediction.weight.data.uniform_(-0.1, 0.1)
        self.word_prediction.bias.data.zero_()
        # The code break down the parameters by type (layer-norm, linear, embedding),
        # but can also condition on individual names, for example by checking pn.endswith(...).
        for pn, p in self.named_parameters():
            if isinstance(p, nn.LayerNorm):
                torch.nn.init.zeros_(p.bias)
                torch.nn.init.ones_(p.weight)
            elif isinstance(p, nn.Linear):
                # You can look at initializers in torch.nn.init4
                torch.nn.init.zeros_(p.bias)
                torch.nn.init.xavier_uniform_(p.weight)
            elif isinstance(p, nn.Embedding):
                # You can look at initializers in torch.nn.init
                torch.nn.init.zeros_(p.bias)
                torch.nn.init.xavier_uniform_(p.weight)

    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        device = next(self.parameters()).device  # Get the device of the model parameters
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32).to(device))  # Move tensor to the correct device
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                # get index of sampled token
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1).item()
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        device = next(self.parameters()).device  # Get the device of the model parameters
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32).to(device))  # Move tensor to the correct device
                topk_logits, topk_indices = torch.topk(logits[0][-1], topK)
                topk_logits = topk_logits / temperature
                distribution_for_last_token = F.softmax(topk_logits, dim=-1)
                # get index of sampled token
                sampled_token = topk_indices[torch.multinomial(distribution_for_last_token, num_samples=1).item()]
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
