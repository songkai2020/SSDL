'''
The code of stage2_BNN is modified from https://github.com/COMP6248-Reproducability-Challenge/selfsupervised-denoising/blob/master-with-report/ssdn/ssdn/models/noise_network.py
'''
import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange



def rotate(x, angle):
    """Rotate images by 90 degrees clockwise. Can handle any 2D data format.
    Args:
        x (Tensor): Image or batch of images.
        angle (int): Clockwise rotation angle in multiples of 90.
        data_format (str, optional): Format of input image data, e.g. BCHW,
            HWC. Defaults to BCHW.
    Returns:
        Tensor: Copy of tensor with rotation applied.
    """
    h_dim, w_dim = 2, 3

    if angle == 0:
        return x
    elif angle == 90:
        return x.flip(w_dim).transpose(h_dim, w_dim)
    elif angle == 180:
        return x.flip(w_dim).flip(h_dim)
    elif angle == 270:
        return x.flip(h_dim).transpose(h_dim, w_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")



class Crop2d(nn.Module):
    """Crop input using slicing. Assumes BCHW data.

    Args:
        crop (Tuple[int, int, int, int]): Amounts to crop from each side of the image.
            Tuple is treated as [left, right, top, bottom]/
    """

    def __init__(self, crop: Tuple[int, int, int, int]):
        super().__init__()
        self.crop = crop
        assert len(crop) == 4

    def forward(self, x):
        (left, right, top, bottom) = self.crop
        x0, x1 = left, x.shape[-1] - right
        y0, y1 = top, x.shape[-2] - bottom
        return x[:, :, y0:y1, x0:x1]


class Shift2d(nn.Module):
    """Shift an image in either or both of the vertical and horizontal axis by first
    zero padding on the opposite side that the image is shifting towards before
    cropping the side being shifted towards.

    Args:
        shift (Tuple[int, int]): Tuple of vertical and horizontal shift. Positive values
            shift towards right and bottom, negative values shift towards left and top.
    """

    def __init__(self, shift: Tuple[int, int]):
        super().__init__()
        self.shift = shift
        vert, horz = self.shift
        y_a, y_b = abs(vert), 0
        x_a, x_b = abs(horz), 0
        if vert < 0:
            y_a, y_b = y_b, y_a
        if horz < 0:
            x_a, x_b = x_b, x_a
        # Order : Left, Right, Top Bottom
        self.pad = nn.ZeroPad2d((x_a, x_b, y_a, y_b))
        self.crop = Crop2d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x):
        return self.shift_block(x)


class ShiftConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """Custom convolution layer as defined by Laine et al. for restricting the
        receptive field of a convolution layer to only be upwards. For a h × w kernel,
        a downwards offset of k = [h/2] pixels is used. This is applied as a k sized pad
        to the top of the input before applying the convolution. The bottom k rows are
        cropped out for output.
        """
        super().__init__(*args, **kwargs)
        self.shift_size = (self.kernel_size[0] // 2, 0)
        # Use individual layers of shift for wrapping conv with shift
        shift = Shift2d(self.shift_size)
        self.pad = shift.pad
        self.crop = shift.crop

    def forward(self, x):
        x = self.pad(x)
        x = super().forward(x)
        x = self.crop(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CTB(nn.Module):
    def __init__(self, channel, head=1):
        super(CTB, self).__init__()
        self.qkv = nn.Conv2d(channel, 3 * channel, 1, padding=0, bias=True)
        self.norm = nn.LayerNorm(64)
        self.head = head

    def forward(self, x):
        b, c, h, w = x.shape
        x_qkv = self.qkv(x)
        q, k, v = x_qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.head)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.head, h=h, w=w)
        return out


class SUB_BNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, dim=48):
        super(SUB_BNN, self).__init__()
        in_channels = in_ch

        ####################################
        # Encode Blocks
        ####################################

        # Layers: enc_conv0, enc_conv1, pool1
        self.encode_block_1 = nn.Sequential(
            ShiftConv2d(in_channels, dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ShiftConv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Shift2d((1, 0)),
            nn.MaxPool2d(2)
        )

        # Layers: enc_conv(i), pool(i); i=2..5
        def _encode_block_2_3_4_5() -> nn.Module:
            return nn.Sequential(
                ShiftConv2d(dim, dim, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                Shift2d((1, 0)),
                nn.MaxPool2d(2)
            )

        # Separate instances of same encode module definition created
        self.encode_block_2 = _encode_block_2_3_4_5()
        self.encode_block_3 = _encode_block_2_3_4_5()
        self.encode_block_4 = _encode_block_2_3_4_5()
        self.encode_block_5 = _encode_block_2_3_4_5()

        # Layers: enc_conv6
        self.encode_block_6 = nn.Sequential(
            ShiftConv2d(dim, dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Decode Blocks
        ####################################
        # Layers: upsample5
        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self.decode_block_5 = nn.Sequential(
            ShiftConv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ShiftConv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        def _decode_block_4_3_2() -> nn.Module:
            return nn.Sequential(
                ShiftConv2d(3 * dim, 2 * dim, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ShiftConv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        # Separate instances of same decode module definition created
        self.decode_block_4 = _decode_block_4_3_2()
        self.decode_block_3 = _decode_block_4_3_2()
        self.decode_block_2 = _decode_block_4_3_2()

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self.decode_block_1 = nn.Sequential(
            ShiftConv2d(2 * dim + in_channels, 2 * dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ShiftConv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.init_weights()

    def init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).

        Only convolution layers have learnable weights. All convolutions use a leaky
        relu activation function (negative_slope = 0.1) except the last which is just
        a linear output.
        """
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
        # Initialise last output layer

    def forward(self, x):

        # Encoder
        pool1 = self.encode_block_1(x)
        pool2 = self.encode_block_2(pool1)
        pool3 = self.encode_block_3(pool2)
        pool4 = self.encode_block_4(pool3)
        pool5 = self.encode_block_5(pool4)
        encoded = self.encode_block_6(pool5)

        # Decoder
        upsample5 = self.decode_block_6(encoded)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode_block_5(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode_block_4(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode_block_3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode_block_2(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        x = self.decode_block_1(concat1)

        return x

    @staticmethod
    def input_wh_mul() -> int:
        """Multiple that both the width and height dimensions of an input must be to be
        processed by the network. This is devised from the number of pooling layers that
        reduce the input size.

        Returns:
            int: Dimension multiplier
        """
        max_pool_layers = 5
        return 2 ** max_pool_layers


class MS_BNN(nn.Module):
    def __init__(self, blindspot1, blindspot2, blindspot3, in_ch=1, out_ch=1, dim=48):
        super(MS_BNN, self).__init__()
        out_channels = out_ch

        self.blindspot1 = blindspot1
        self.blindspot2 = blindspot2
        self.blindspot3 = blindspot3

        self.BNN = SUB_BNN(in_ch, out_ch, dim)

        # Shift blindspot pixel down
        self.shift1 = Shift2d(((self.blindspot1 + 1) // 2, 0))
        self.shift2 = Shift2d(((self.blindspot2 + 1) // 2, 0))
        self.shift3 = Shift2d(((self.blindspot3 + 1) // 2, 0))

        self.output_conv = ShiftConv2d(2 * dim, out_channels, 1)
        self.output_block = nn.Sequential(
            ShiftConv2d(3 * 8 * dim, 8 * dim, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ShiftConv2d(8 * dim, 2 * dim, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.output_conv,
        )
        nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def forward(self, x, shift=None):

        if shift is not None:
            self.shift1 = Shift2d((shift, 0))
            self.shift2 = Shift2d((shift, 0))
            self.shift3 = Shift2d((shift, 0))
        else:
            self.shift1 = Shift2d(((self.blindspot1 + 1) // 2, 0))
            self.shift2 = Shift2d(((self.blindspot2 + 1) // 2, 0))
            self.shift3 = Shift2d(((self.blindspot3 + 1) // 2, 0))

        rotated = [rotate(x, rot) for rot in (0, 90, 180, 270)]
        x = torch.cat((rotated), dim=0)

        x = self.BNN(x)

        # Apply shift
        shifted1 = self.shift1(x)
        shifted2 = self.shift2(x)
        shifted3 = self.shift3(x)

        # Unstack, rotate and combine
        rotated_batch1 = torch.chunk(shifted1, 4, dim=0)
        rotated_batch2 = torch.chunk(shifted2, 4, dim=0)
        rotated_batch3 = torch.chunk(shifted3, 4, dim=0)

        aligned1 = [
            rotate(rotated, rot)
            for rotated, rot in zip(rotated_batch1, (0, 270, 180, 90))
        ]
        aligned2 = [
            rotate(rotated, rot)
            for rotated, rot in zip(rotated_batch2, (0, 270, 180, 90))
        ]
        aligned3 = [
            rotate(rotated, rot)
            for rotated, rot in zip(rotated_batch3, (0, 270, 180, 90))
        ]
        x = torch.cat((aligned1 + aligned2 + aligned3), dim=1)
        x = self.output_block(x)

        return x


if __name__ == "__main__":
    a = torch.Tensor([1, 2, 3])
    b = torch.Tensor([4, 5, 6])
    print(torch.concat((a, b), dim=0))


