import torch
import torch.nn as nn
import math
from torch.functional import F

class DiffusionUNet(nn.Module):
    """
    Base class for Diffusion U-Net model.
    """
    requires_alpha_hat_timestep = False

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by number of heads"

        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, seq_length, embed_size = x.shape

        # Split the embedding into `num_heads` pieces
        K = self.keys(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        Q = self.queries(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = self.values(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Compute attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K]) / (self.embed_size ** (1 / 4))
        attention = torch.softmax(energy, dim=-1)

        # Apply attention to values and concatenate heads
        out = torch.einsum("nhql,nlhd->nqhd", [attention, V]).reshape(batch_size, seq_length, embed_size)
        out = self.fc_out(out)

        return out


class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SimpleSelfAttention, self).__init__()
        self.embed_size = embed_size

        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.values = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_size)
        K = self.keys(x)
        Q = self.queries(x)
        V = self.values(x)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attention, V)
        return out

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

# class SelfAttention(nn.Module):
#     """
#     Implements self-attention mechanism.

#     Args:
#         in_dim (int): Number of input channels.
#     """
#     def __init__(self, in_dim):
#         super(SelfAttention, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
        
#         batch_size, C, width, height = x.size()
#         query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
#         key = self.key_conv(x).view(batch_size, -1, width * height)
#         value = self.value_conv(x).view(batch_size, -1, width * height)

#         attention = self.softmax(torch.bmm(query, key))
#         out = torch.bmm(value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, C, width, height)

#         return out + x
    
    

class ResidualBlock(nn.Module):
    """
    Implements a residual block with optional residual connections.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int, optional): Number of middle channels. Defaults to None.
        residual (bool, optional): Whether to include a residual connection. Defaults to False.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False) -> None:
        super(ResidualBlock, self).__init__()
        self.residual = residual

        if not mid_channels:
            mid_channels = out_channels

        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels), #TODO Try other normalization (group)
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
 

    def forward(self, x):
        if self.residual:
            res = self.res(x)
        else:
            res = x

        x = self.double_conv(x)
        x += res
        return x
    
    
class DownBlock(nn.Module):
    """
    Implements a down-sampling block consisting of multiple residual blocks and an average pooling layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        block_depth (int): Number of residual blocks.
        emb_dim (int, optional): Embedding dimension size. Defaults to 256.
    """
    def __init__(self, in_channels, out_channels, block_depth, emb_dim=256) -> None:
        super(DownBlock, self).__init__()

        layers = nn.ModuleList()
        for i in range(block_depth):
            if i == 0:
                layers.append(ResidualBlock(in_channels, out_channels, residual=True))
            else:
                layers.append(ResidualBlock(out_channels, out_channels, residual=False))

        self.residual_blocks = layers
        self.downsample = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        skip_outputs = []
        for residual_block in self.residual_blocks:
            x = residual_block(x)
            skip_outputs.append(x)
        x = self.downsample(x)
        return x, skip_outputs
    

class UpBlock(nn.Module):
    """
    Implements an up-sampling block consisting of multiple residual blocks and an up-sample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        skip_channels (int): Number of skip connection channels.
        block_depth (int): Number of residual blocks.
        emb_dim (int, optional): Embedding dimension size. Defaults to 256.
    """
    def __init__(self, in_channels, out_channels, skip_channels, block_depth, emb_dim=256) -> None:
        super(UpBlock, self).__init__()
    
        layers = nn.ModuleList()
        for i in range(block_depth):
            if i == 0:
                layers.append(ResidualBlock(in_channels + skip_channels, out_channels, residual=True))
            else:
                layers.append(ResidualBlock(out_channels + skip_channels, out_channels, residual=True))

        self.residual_blocks = layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
   
    def forward(self, x, skip_inputs):
        x = self.upsample(x)
        for residual_block in self.residual_blocks:
            x = torch.cat([x, skip_inputs.pop()], dim=1)
            x = residual_block(x)
        return x
    
# features=[64, 128, 256, 512, 1024]
class UNet(DiffusionUNet):
    """
    Implements the UNet architecture for diffusion models.

    Args:
        c_in (int, optional): Number of input channels. Defaults to 3.
        c_out (int, optional): Number of output channels. Defaults to 3.
        image_size (int, optional): Size of the image. Defaults to 64.
        conv_dim (int, optional): Base number of convolution dimensions. Defaults to 64.
        block_depth (int, optional): Depth of blocks in the model. Defaults to 3.
        time_emb_dim (int, optional): Time embedding dimension size. Defaults to 256.
    """
    def __init__(self, c_in=3, c_out=3, image_size=128, conv_dim=64, block_depth=3, time_emb_dim=256) -> None:
        super(UNet, self).__init__()
        self.requires_alpha_hat_timestep = True


        channel_sizes = [32, 64, 96, 128, 256, 512]
        self.pre_conv = nn.Conv2d(c_in, 32, kernel_size=3, padding=1, bias=False)
        self.embedding_upsample = nn.Upsample(size=(image_size, image_size), mode='nearest')

        #self.attn_down1 = SelfAttention(64)
        self.down1 = DownBlock(64, 32, block_depth)
        self.down2 = DownBlock(32, 64, block_depth)

        self.attn_down3 = SelfAttention(64, 32)
        self.down3 = DownBlock(64, 96, block_depth)

        self.attn_down4 = SelfAttention(96, 16)
        self.down4 = DownBlock(96, 128, block_depth)

        self.bottleneck1 = ResidualBlock(128, 256, residual=True)
        self.attn_bottleneck = SelfAttention(256, 8)
        self.bottleneck2 = ResidualBlock(256, 256, residual=True) # TODO: Deeper, wider net, for detail
        
        self.up1 = UpBlock(256, 128, 128, block_depth)
        self.attn_up1 = SelfAttention(128, 16)

        self.up2 = UpBlock(128, 96, 96, block_depth)
        self.attn_up2 = SelfAttention(96, 32)

        self.up3 = UpBlock(96, 64, 64, block_depth)
        self.up4 = UpBlock(64, 32, 32, block_depth)

        self.output = nn.Conv2d(32, c_out, kernel_size=3, padding=1, bias=False)

        # DownBlock:  3, 32     ------>                     # UpBlock: 64, 32, + 32
            # DownBlock: 32, 64     ------>             # UpBlock: 96, 64, + 64
                # DownBlock: 64, 96     ------>     # UpBlock: 128, 96, + 96

                                    # Bottleneck: 96, 128
                                    # Bottleneck: 128, 128

    def sinusoidal_embedding(self, x):
        """
        Generates sinusoidal embeddings for time steps.

        Args:
            x (Tensor): Tensor of time steps.

        Returns:
            Tensor: Sinusoidal embeddings.
        """
        embedding_min_frequency = 1.0
        embedding_max_frequency = 1000.0
        embedding_dims = 32
        frequencies = torch.exp(
            torch.linspace(
                math.log(embedding_min_frequency),
                math.log(embedding_max_frequency),
                embedding_dims // 2,
            )
        ).to(x.device)
        angular_speeds = 2.0 * math.pi * frequencies
        sin_part = torch.sin(angular_speeds * x)
        cos_part = torch.cos(angular_speeds * x)
        embeddings = torch.cat([sin_part, cos_part], dim=3).permute(0, 3, 1, 2)
        return embeddings

    def forward(self, x, t):
        #print("Input shape:", x.shape)
        x = self.pre_conv(x)
        #print("After pre_conv shape:", x.shape)
        t = self.sinusoidal_embedding(t)
        #print("After sinusoidal_embedding shape:", t.shape)
        t = self.embedding_upsample(t)
        #print("After embedding_upsample shape:", t.shape)
        x = torch.cat([x, t], dim=1)
        #print("After concatenation shape:", x.shape)

        # Downward path
        
        x, skip1 = self.down1(x)
        #print("After down1 shape:", x.shape)

        x, skip2 = self.down2(x)
        #print("After down2 shape:", x.shape)
        x = self.attn_down3(x)  
        x, skip3 = self.down3(x)
        #print("After down3 shape:", x.shape)

        x = self.attn_down4(x)  
        x, skip4 = self.down4(x)
        #print("After down4 shape:", x.shape)

        # Bottleneck
        x = self.bottleneck1(x)
        #print("After bottleneck1 shape:", x.shape)
        x = self.attn_bottleneck(x)
        x = self.bottleneck2(x)
        #print("After bottleneck2 shape:", x.shape)

        # Upward path
        x = self.up1(x, skip4)
        x = self.attn_up1(x)  
        #print("After up1 shape:", x.shape)
        x = self.up2(x, skip3)
        x = self.attn_up2(x)  
        #print("After up2 shape:", x.shape)
        
        x = self.up3(x, skip2)
        
        #print("After up3 shape:", x.shape)

        x = self.up4(x, skip1)
        #print("After up4 shape:", x.shape)
       

        output = self.output(x)
        #print("Output shape:", output.shape)

        return output




if __name__ == '__main__':
    net = UNet(block_depth=2)
    net = net.to('cpu')
    ##print(sum([p.numel() for p in net.parameters()]))
    ##print(net)
    x = torch.randn(2, 3, 128, 128)
    t = torch.tensor([[[[0.2860]]],[[[0.2860]]]])
    x = x.to('cpu')
    t = t.to('cpu')
    pred = net(x, t) 
    #print(pred.shape)










