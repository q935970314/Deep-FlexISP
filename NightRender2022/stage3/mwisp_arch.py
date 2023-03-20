import torch
import networks as N
import torch.nn as nn
import math
import torch.optim as optim

class MWRCAN(nn.Module):
    def __init__(self):
        super(MWRCAN, self).__init__()
        c1 = 64
        c2 = 96
        c3 = 128
        n_b = 20

        self.hh = nn.PixelUnshuffle(2)


        self.head = N.DWTForward()

        self.down1 = N.seq(
            nn.Conv2d(64, c1, 3, 1, 1),
            nn.PReLU(),
            N.RCAGroup(in_channels=c1, out_channels=c1, nb=n_b)
        )

        self.down2 = N.seq(
            N.DWTForward(),
            nn.Conv2d(c1 * 4, c2, 3, 1, 1),
            nn.PReLU(),
              N.RCAGroup(in_channels=c2, out_channels=c2, nb=n_b)
        )

        self.down3 = N.seq(
            N.DWTForward(),
            nn.Conv2d(c2 * 4, c3, 3, 1, 1),
            nn.PReLU()
        )

        self.middle = N.seq(
            N.RCAGroup(in_channels=c3, out_channels=c3, nb=n_b),
            N.RCAGroup(in_channels=c3, out_channels=c3, nb=n_b)
        )
        
        self.up1 = N.seq(
            nn.Conv2d(c3, c2 * 4, 3, 1, 1),
            nn.PReLU(),
            N.DWTInverse()
        )

        self.up2 = N.seq(
            N.RCAGroup(in_channels=c2, out_channels=c2, nb=n_b),
            nn.Conv2d(c2, c1 * 4, 3, 1, 1),
            nn.PReLU(),
            N.DWTInverse()
        )

        self.up3 = N.seq(
            N.RCAGroup(in_channels=c1, out_channels=c1, nb=n_b),
            nn.Conv2d(c1, 64, 3, 1, 1)
        )

        self.tail = N.seq(
            N.DWTInverse(),
            nn.Conv2d(16, 3, 3, 1, 1),
            # nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x, c=None):
        c0 = self.hh(x)
        c1 = self.head(c0)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        m = self.middle(c4)
        c5 = self.up1(m) + c3
        c6 = self.up2(c5) + c2
        c7 = self.up3(c6) + c1
        out = self.tail(c7)

        return out
