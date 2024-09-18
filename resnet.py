from torch import nn

class ResBlock(nn.Module):
    def __init__(self, dim, act="relu"):
        super().__init__()
        if act == "relu":
            activation = nn.ReLU()
        elif act == "elu":
            activation = nn.ELU()
        self.block = nn.Sequential(
            activation,
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            activation,
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)
class EncoderVqResnet28(nn.Module):
    def __init__(
                self,
                dim_z:  int = 64,
                num_rb: int = 2,
                flg_bn: bool = True
    ):
        super().__init__()
        # Convolution layers
        layers_conv = []
        layers_conv.append(nn.Conv2d(1, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 2))
        layers_conv.append(nn.ReLU(True))
        layers_conv.append(nn.Conv2d(dim_z // 2, dim_z, 4, stride=2, padding=1))
        self.conv = nn.Sequential(*layers_conv)
        # Resblocks
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)

    def forward(self, x):
        out_conv = self.conv(x)
        out_res = self.res(out_conv)
        mu = self.res_m(out_res)
        return mu

class DecoderVqResnet28(nn.Module):
    def __init__(self, dim_z, num_rb, flg_bn=True):
        super().__init__()
        # Resblocks
        layers_resblocks = []
        for i in range(num_rb):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        # Convolution layers
        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU(True))
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, 1, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        self.convt = nn.Sequential(*layers_convt)

    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)
        return out

class Auto_Encoder(nn.Module):
    def __init__(self, dim_z, num_rb, flg_bn):
        super().__init__()
        self.encoder = EncoderVqResnet28(dim_z=dim_z, num_rb=num_rb, flg_bn=flg_bn)
        self.decoder = DecoderVqResnet28(dim_z=dim_z, num_rb=num_rb, flg_bn=flg_bn)

    def encode(self,x):
        return self.encoder(x)

    def decode_code(self, code):
        return self.decoder(code)

    def forward(self, x):
        code = self.encoder(x)
        reco = self.decoder(code)
        return reco