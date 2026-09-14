import torch
from torch import nn


class SubjectLayer(nn.Module):
    """Apply an identity-plus-learned channel correction for each subject."""

    def __init__(self, n_subjects, channels, bias=False):
        super().__init__()

        self.delta = nn.Parameter(
            torch.zeros(n_subjects, channels, channels)
        )

        self.register_buffer(
            "identity",
            torch.eye(channels),
            persistent=False,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(n_subjects, channels))
        else:
            self.bias = None

    def forward(self, x, subject_id):
        """
        Args:
            x: EEG tensor with shape [batch, channels, time].
            subject_id: Integer subject identifier for each batch item [batch].

        Returns:
            EEG transformed by each subject's own channel-alignment matrix.
        """

        W = self.identity.unsqueeze(0) + self.delta[subject_id]
        x = torch.bmm(W, x)

        if self.bias is not None:
            b = self.bias[subject_id]
            x = x + b.unsqueeze(-1)

        return x


class ResidualConv1d(nn.Module):
    """Track temporal patterns with a residual dilated convolution."""

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()

        padding = (kernel_size // 2) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.net(x)


class GLUConv1d(nn.Module):
    """Control feature flow with a convolutional gated linear unit."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=2 * channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GLU(dim=1),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Combine two dilated residual convolutions with a final GLU gate."""

    def __init__(self, channels=320, kernel_size=3, dilation1=1, dilation2=2):
        super().__init__()

        self.res1 = ResidualConv1d(
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation1,
        )

        self.res2 = ResidualConv1d(
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation2,
        )

        self.glu = GLUConv1d(
            channels=channels,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.glu(x)

        return x


class EEGEncoder(nn.Module):
    """Encode EEG using an architecture modeled after Défossez et al."""

    def __init__(
        self,
        n_subjects=20,
        in_channels=125,
        hidden_channels=320,
        out_channels=768,
        embedding_dim=512,
        use_subject_layer=True,
        use_projection_head=True,
    ):
        super().__init__()

        self.use_subject_layer = use_subject_layer
        self.use_projection_head = use_projection_head

        if use_subject_layer:
            self.subject_layer = SubjectLayer(
                n_subjects=n_subjects,
                channels=in_channels,
                bias=False,
            )
        else:
            self.subject_layer = None

        self.input_projection = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
        )

        self.blocks = nn.ModuleList()

        for k in range(5):
            dilation1 = 2 ** ((2 * k) % 5)
            dilation2 = 2 ** ((2 * k + 1) % 5)

            self.blocks.append(
                ResidualBlock(
                    channels=hidden_channels,
                    kernel_size=3,
                    dilation1=dilation1,
                    dilation2=dilation2,
                )
            )

        self.output_projection = nn.Sequential(
            nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(2 * hidden_channels, out_channels, kernel_size=1),
        )

        if use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(out_channels, embedding_dim, kernel_size=1),
            )
        else:
            self.projection_head = nn.Identity()

    def forward(self, x, subject_id):
        if self.use_subject_layer:
            x = self.subject_layer(x, subject_id)

        x = self.input_projection(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_projection(x)
        x = self.projection_head(x)
        return x
