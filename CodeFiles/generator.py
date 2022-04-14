import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.res(x) # check this.

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        num_residual_blocks = 9        
        self.generator_model = nn.Sequential(
            #c7s1-64
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            #d128
            *self.convolution_layer( in_channels = 64, 
                                    out_channels = 128, 
                                    kernel_size = 3, 
                                    instance_norm = 128, 
                                    stride = 2, 
                                    padding = 1
                                ),
            
            #d256
            *self.convolution_layer( in_channels = 128, 
                                    out_channels = 256, 
                                    kernel_size = 3, 
                                    instance_norm = 256, 
                                    stride = 2, 
                                    padding = 1
                                ),
            
            *[ResidualBlock(256) for _ in range(num_residual_blocks)],
            
            #u128
            *self.transposed_convolution_layer(  in_channels = 256, 
                                                out_channels = 128, 
                                                kernel_size = 3, 
                                                instance_norm = 128, 
                                                stride = 2, 
                                                padding = 1, 
                                                output_padding = 1
                                            ),
            
            #u64
            *self.transposed_convolution_layer(  in_channels = 128, 
                                                out_channels = 64, 
                                                kernel_size = 3, 
                                                instance_norm = 64, 
                                                stride = 2, 
                                                padding = 1, 
                                                output_padding = 1
                                            ),
            
            #c7s1-3
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.InstanceNorm2d(3),
            nn.Sigmoid()
        )
    
    @staticmethod
    def convolution_layer(in_channels, out_channels, kernel_size, instance_norm, stride = 1, padding = 0):
        return(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(instance_norm),
            nn.ReLU(inplace = True),
        )

    @staticmethod
    def transposed_convolution_layer(in_channels, out_channels, kernel_size, instance_norm, stride = 1, padding = 0, output_padding = 0):
        return(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.InstanceNorm2d(instance_norm),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.generator_model(x)
        