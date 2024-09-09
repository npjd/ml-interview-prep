import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(InputEmbedding,self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # Register the buffer so that it is not considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
       #  must not be trainable
       x = x + (self.pe[:, :x.size(1)]).requires_grad_(False)
       return self.dropout(x)
    

class LayerNorm(nn.Module):

    def __init__(self, eps = 10**-6) -> None:
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # each feature in the emebdding is normalized separately (w.r.t that entire embeddings mean + std)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FFN(nn.Module):

    def __init__(self, d_model, d_ff):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        return self.ffn(x)

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, h, dropout) -> None:
        super(MultiHeadAttention, self).__init__()

        # since each head will have d_k dimensions and we have h heads, d_model must be divisible by h
        assert d_model % h == 0

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask = None):
        Q = self.W_Q(q)
        K = self.W_K(k)
        V = self.W_V(v)

        Q = Q.view(Q.size(0), Q.size(1), self.h, self.d_k).transpose(1, 2) # review operation like this
        K = K.view(K.size(0), K.size(1), self.h, self.d_k).transpose(1, 2)
        V = V.view(V.size(0), V.size(1), self.h, self.d_k).transpose(1, 2)

        # Q, K, V -> (batch_size, h, seq_len, d_k)

        attention_scores = Q @ K.transpose(-1, -2) / math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        # attention_scores -> (batch_size, h, seq_len, seq_len)

        output = (attention_scores @ V).transpose(1, 2).contiguous().view(Q.size(0), -1, self.d_model) # -1 in view autofills it up

        return self.W_O(output)


class ResidualConnection(nn.Module):

    def __init__(self, dropout) -> None:
        super(ResidualConnection, self).__init__()

        self.norm = LayerNorm()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):

    def __init__(self, self_attn: MultiHeadAttention, ffn:FFN, dropout) -> None:
        super(EncoderBlock, self).__init__()

        self.self_attn = self_attn
        self.ffn = ffn
        self.residuals = nn.ModuleList(
            [ResidualConnection(dropout), ResidualConnection( dropout)]
        )
    
    def forward(self, x, mask):
        
        x = self.residuals[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.residuals[1](x, lambda x: self.ffn(x))

        return x

class Encoder(nn.Module):

    def __init__(self, layer: EncoderBlock, N) -> None:
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([layer for _ in range(N)])
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
    
class DecoderBlock(nn.Module):

    def __init__(self, self_attn: MultiHeadAttention, src_attn: MultiHeadAttention, ffn: FFN, dropout) -> None:
        super(DecoderBlock, self).__init__()

        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ffn = ffn

        self.residuals = nn.ModuleList(
            [ResidualConnection(dropout), ResidualConnection(dropout), ResidualConnection(dropout)]
        )
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residuals[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.residuals[1](x, lambda x: self.src_attn(x, encoder_output, encoder_output, src_mask))
        x = self.residuals[2](x, lambda x: self.ffn(x))

class Decoder(nn.Module):

    def __init__(self, layer: DecoderBlock, N) -> None:
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([layer for _ in range(N)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size) 
        return torch.softmax(self.linear(x), dim=-1)


class Transformer(nn.Module):

    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed:InputEmbedding, tgt_embed:InputEmbedding, src_pos:PositionalEncoding, tgt_pos:PositionalEncoding, generator:ProjectionLayer) -> None:
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        return self.decoder(self.tgt_pos(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.generator(decoder_output)
    
def generate_model(
        d_model,
        src_vocab_size,
        tgt_vocab_size,
        src_max_seq_len,
        tgt_max_seq_len,
        h,
        N,
        d_ff,
        dropout
):
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_max_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_max_seq_len, dropout)
    generator = ProjectionLayer(d_model, tgt_vocab_size)
    self_attn = MultiHeadAttention(d_model, h, dropout)
    src_attn = MultiHeadAttention(d_model, h, dropout)
    ffn = FFN(d_model, d_ff)
    
    encoder = Encoder(EncoderBlock(self_attn, ffn, dropout), N)
    decoder = Decoder(DecoderBlock(self_attn, src_attn, ffn, dropout), N)

    model = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model


