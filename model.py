import torch
import torch.nn as nn
import math

# 2:40
# d_model is the size of the vector that you transform one single word into
class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    # In the paper it sepcified that the weight is exactly the square root of the d_model, hence
    def forward(self, x):
        self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    # Of cocurse you need the seq_len here because the concept "position" is regard to the word inside that sentence with lenght of seq_len
    def __init__(self, d_model:int, seq_len:int, dropout:float)-> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout =nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        # Why seq_len in the row and d_model in the columns, lets just put it aside first and use the graph as reference now
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1), unsqueeze(1) because I want this to be a column vector
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()* (-math.log(10000.0)/d_model))
        # Apply the sin to even position
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        # pe now have the dimensiom of (seq_len, d_model) but we will have a batch of sentences, so we add a new dimension to it
        pe = pe.unsqueeze(0) #(1, seq_len, d_model)

        # Save the tensor in the model when the model is saved (not as a learnable parameter)
        self.register_buffer('pe', pe)

        def forward(self, x):
            # Here we only specify the seq_len which is different in every input
            x = x + (self.pe[:, :x.shape[1], :]).require_grad_(False) # don't change so no need to update
            return self.dropout(x)

# 13:30 
# Each sentence is an item, for each item calculate its mean and variance
# Independent from the item of the same batch
# Then calculate the new values using its own mean and variance
# Why do we need eps? cause if std becomes zero or very close to zero, new x will be too large
class LayerNormalization(nn.Module):

    def __init__(self, eps:float=10**-6)-> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied, use nn.Parameter makes it learnable
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        # The mean function cancls the dimension to which it is applied, but we want to keep it
        # For an input tensor x of shape (batch_size, seq_len, d_model), dim=-1 corresponds to the d_model (the last dimension)
        # The mean is calculated independently for each token across its features, resulting in a tensor of shape (batch_size, seq_len, 1)
        # LayerNorm normalizes along the "d_model (feature)" dimension to keep each token’s representation well-scaled. 
        # This prevents the vector space from exploding or collapsing as the model goes deeper.
        # We don't normalize along the seq_len dimension because we don't want each words information to be mixed up
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean)/(std+self.eps)+ self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff: int, dropout: float)-> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # Include W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # Include W2 and B2

    def forward(self, x):
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_ff) --> (Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    # h is how many head are there
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h ==0, "d_model is not divisible by h"

        self.d_k = d_model //h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    # To make this method able to be called directly on the class itself.
    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        # I think what d_k means is how many dimension can each head take care of
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        # key.transpose(-2, -1) is the same with key.transpose(-1, -2) according to chatGPT
        attention_scores = (query @ key.transpose(-2, -1)) /math.sqrt(d_k)
        if mask is not None:
        # Remember the mask here is a torch tensor!
        # The reason we shouldn't be using "if mask" is because python raises an error if mask is a Tensor with more than one element
        # So rule of thumb, if mask and the dropout below is more then one element then keep with "if mask is not None"
            # Write a very low value (indicating -inf) to the position where ths mask ==0
            # Why can't we just sign 0 to the future words and do it without softmax???
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention_scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        key = self.w_k(k) # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        value = self.w_v(v) # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attentione
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h* self.d_k)
        # contiguous(): Returns a version of the tensor with its data in a continuous memory block. 
        # This is often needed before using view() to reshape a tensor reliably.
        # Chatgpt said it is more of a safeguard if we use (x.shape[0], -1, self.h* self.d_k) instead of (x.shape[0], x.shape[1], self.h* self.d_k) in here
        # Careful you cannot direectly do x = x.view(x.shape[0], -1, self.h*self.d_k), the 2 axis you want to merge have to be adjacent!!! They call this "flattening".
        
        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        # Accrording to chatGPT, this is called Pre-LN (pre-layer normalization), which is different from the original paper
        # In the original paper, the layer normalization is applied after the residual connection which is more like LayerNorm(x + sublayer(x))

        
class EncoderBlock(nn.Module):
# This is just one block of the encoder
# This "lets the caller decide the architecture",
# Enabling more flexible hyperparameter tuning and dynamic networks (e.g., different blocks in different layers).
# So don't hard code stuff in here, for example: self.self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    
    def __init__(self, features: int, self_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        # Remember, the ResidualConnection here is calling the __init__ method not the forward method!!
        # Always wrap in brackets [...] when using nn.ModuleList (or nn.Sequential)!!!

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # lambda x: ... wraps the attention block in a callable function, delaying execution
        # Without it, you're immediately executing the attention block and passing a tensor where a function is expected 
        # → this breaks the residual wrapper
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
# This part is needed because there will be more then 1 encoderblock stacking together!

    def __init__(self, features: int, layers: nn.ModuleList) ->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        # This applies LayerNorm to the final output of the encoder stack which is often called "final LayerNorm" or "output LayerNorm".
        # Another deviation from the original paper.
        # It’s a modern, stability-driven enhancement.


class DecoderBlock(nn.Module):
    def __init__(self, feature: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float)-> None:
        super().__init__()
        self.self_attention_block  = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(feature, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_msk, tgt_msk):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_msk))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_msk))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList)-> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_msk, tgt_msk):
        for layer in self.layers:
            x = layer(x, encoder_output, src_msk, tgt_msk)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
        # the log_softmax is required if you are using nn.NLLLoss as the loss function
        # If you are using nn.CrossEntropyLoss, you don't need to apply log_softmax here, as it combines softmax and negative log likelihood loss in one single class.
        # You can just use self.proj(x) directly and pass the output to nn.CrossEntropyLoss.
        # The reason for using log_softmax instead of softmax is to prevent numerical instability when calculating the loss.
        # The log_softmax function is more numerically stable than applying softmax followed by log, especially for large values.
        # This is because log(softmax(x)) can lead to very small values, which can cause numerical underflow.
        # By using log_softmax directly, you avoid this issue and ensure that the gradients are computed correctly during backpropagation.

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed:InputEmbeddings, tgt_embed: InputEmbeddings, src_pos:PositionalEncoding, tgt_pos:PositionalEncoding, projection_layer:ProjectionLayer)->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output:torch.Tensor, src_mask:torch.Tensor, tgt: torch.Tensor, tgt_mask:torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


# If we look closely, dropout is applied somewhere in every function, except for the InputEmbeddings, LayerNormalization and ProjectionLayer
# LayerNormalization only applied at residual connection and the output of Encoder and Decoder 