import argparse
import os
import sys
import pickle

from generator import Generator
from discriminator import Discriminator

import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import DataLoader

import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as img_utils

torch.manual_seed(1337)

class GAN:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.define_arguments(self.parser)

    @staticmethod
    def terminate(error):
        """
        Function to display an error message and exit the program
        """
        print("Error: " + error)
        sys.exit(-1)

    @staticmethod
    def define_arguments(parser):
        """
        Define command line arguments
        """
        parser.add_argument(
            "--data",
            dest = "data",
            type = str,
            required = True,
            help = "Specify path to dataset"
        )

        parser.add_argument(
            "--out",
            dest = "output_path",
            type = str,
            required = True,
            help = "Specify the path to store output"
        )

        parser.add_argument(
            "--resume",
            dest = "resume",
            action='store_true',
            help = "Resume training from a previous checkpoint"
        )

        parser.add_argument(
            "--genpath",
            dest = "generator_path",
            type = str,
            help = "Specify path to generator checkpoint"
        )

        parser.add_argument(
            "--discpath",
            dest = "discriminator_path",
            type = str,
            help = "Specify path to discriminator checkpoint"
        )

        parser.add_argument(
            "--ngpu",
            dest = "ngpu",
            type = int,
            required = True,
            help = "Specify number of GPUs to use"
        )

        parser.add_argument(
            "--dim",
            dest = "image_dimension",
            type = int,
            required = True,
            help = "Specify the dimensions of the input image"
        )

    def validate_input(self, args):
        """
        Validates use input
        """
        if args.resume and not (args.genpath and args.destpath):
            self.terminate("Need location of generator and discriminator data to resume training")

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
    
    @staticmethod
    def load_data(path, dimensions, batch_size = 64, num_workers = 2):
        transform = transforms.Compose([
            transforms.Resize(dimensions),
            transforms.CenterCrop(dimensions),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        training_data = dataset.ImageFolder(
            root = path,    
            transform = transform
        )
        return DataLoader(  training_data,
                            batch_size = batch_size, 
                            shuffle = True, 
                            num_workers = num_workers)
    @staticmethod
    def train_model(device, generator, discriminator, batch_size, max_epochs, lr, beta1, \
                    gen_weights_dir, disc_weights_dir):
        loss_function = nn.MSELoss()
        generator_optimizer = optimizer.Adam(   generator.parameters(),
                                            lr = lr,
                                            betas = (beta1, 0.999)
                                        )
        discriminator_optimizer = optimizer.Adam(discriminator.parameters(),
                                            lr = lr,
                                            betas = (beta1, 0.999)
                                        )
        start = 0
        real_label = 1
        fake_label = 0
        iterations = 0
        image_list = []
        generator_loss_values = []
        discriminator_loss_values = []
        fixed_noise = torch.randn(batch_size, 3, 256, 256, device = device)
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        image_grid = img_utils.make_grid(fake, padding = 2, normalize = True)
        image_list.append(image_grid)
        img_utils.save_image(image_grid, os.path.join(gen_weights_dir, "img_epoch_0.png"))
        for epoch in range(start + 1, max_epochs):
            # For each batch in the data_loader
            print("Current Epoch: ", epoch)
            for i, data in enumerate(data_loader):       
            
                real_img = data[0].to(device)
                b_size = real_img.size(0)
                real_label = torch.full((b_size, 1, 30, 30), 1, device = device, dtype = torch.float32)
                fake_label = torch.full((b_size, 1, 30, 30), 0, device = device, dtype = torch.float32)
                
                generator_optimizer.zero_grad()
                # Generate batch of latent vectors
                noise = torch.randn(b_size, 3, 256, 256, device = device)
                # Generate fake image batch with G
                fake_img = generator(noise)
                fake_g_output = discriminator(fake_img)
                generator_loss = loss_function(fake_g_output,real_label)
                generator_loss.backward()
                generator_optimizer.step()
                
                discriminator_optimizer.zero_grad()
                real_d_output = discriminator(real_img)
                discriminator_loss_real = loss_function(real_d_output, real_label)
                noise_d = torch.randn(b_size, 3, 256, 256, device = device)
                fake_d_output = discriminator(generator(noise_d).detach())
                discriminator_loss_fake = loss_function(fake_d_output, fake_label)
                
                discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 2
                discriminator_loss.backward()
                discriminator_optimizer.step()
        
                # Print training statistics
                if i % 10 == 0:
                    print(  '[%d/%d][%d/%d] \
                            \tDiscrminator Loss: %.4f \
                            \tGenerator Loss: %.4f\t'
                          % (epoch, max_epochs, i, len(data_loader),
                             discriminator_loss.item(), generator_loss.item())
                        )
        
                # Save Losses for plotting 
                generator_loss_values.append(generator_loss.item())
                discriminator_loss_values.append(discriminator_loss.item())
        
                iterations += 1
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            image_grid = img_utils.make_grid(fake, padding = 2, normalize = True)
            image_list.append(image_grid)
            # Save image grid for this epoch
            img_utils.save_image(image_grid, os.path.join(gen_weights_dir, "img_epoch_" + str(epoch + 1) + ".png"))
            # Save generator weights for this epoch
            torch.save(generator.state_dict(), os.path.join(gen_weights_dir, "gen_epoch_"+ str(epoch + 1) + ".pth"))
            # Save disciminator weights for this epoch
            torch.save(discriminator.state_dict(), os.path.join(disc_weights_dir, "disc_epoch_" + str(epoch + 1) + ".pth"))
        return image_list

    @staticmethod
    def check_output_path(output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        generator_weights_dir = "weights/generator/"
        discriminator_weights_dir = "weights/discrimnator/"
        for dir in [generator_weights_dir, discriminator_weights_dir]:
            if not os.path.exists(output_path + dir):
                os.makedirs(output_path + dir)
        return([output_path + dir for dir in [generator_weights_dir, discriminator_weights_dir]])


if __name__ == "__main__":
    gan = GAN()
    args = gan.parser.parse_args()
    gan.validate_input(args)

    gen_weights_dir, disc_weights_dir = gan.check_output_path(args.output_path)

    dimensions = args.image_dimension
    dataset_path = args.data
    n_gpus = args.ngpu

    data_loader = gan.load_data(dataset_path, dimensions)

    if (torch.cuda.is_available() and n_gpus > 0):
        device = torch.device("cuda:0")  
    else:
        device = ("cpu")

    # Construct models
    generator = Generator().to(device)
    if (device == 'cuda:0' and n_gpus > 1): 
        generator = nn.DataParallel(generator, list(range(n_gpus)))

    discriminator = Discriminator().to(device)
    if (device == 'cuda:0' and n_gpus > 1):
        discriminator = nn.DataParallel(discriminator, list(range(n_gpus)))

    if args.resume:
        generator_checkpoint = args.generator_path
        discriminator_checkpoint = args.discriminator_path
        generator.load_state_dict(torch.load(generator_checkpoint))
        discriminator.load_state_dict(torch.load(discriminator_checkpoint))

    else:   
        generator.apply(gan.initialize_weights)
        discriminator.apply(gan.initialize_weights)

    image_list = gan.train_model(   device = device,
                                    generator = generator,
                                    discriminator = discriminator,
                                    batch_size = 64,
                                    max_epochs = 2,
                                    lr = 0.0002,
                                    beta1 = 0.5,
                                    gen_weights_dir = gen_weights_dir,
                                    disc_weights_dir = disc_weights_dir
                                )

    torch.save(generator.state_dict(), gen_weights_dir + "gen_final.pth")
    torch.save(discriminator.state_dict(), disc_weights_dir + "disc_epoch_final.pth")
        
    with open('./gan.pkl', 'wb') as f:
        pickle.dump(image_list, f)
    