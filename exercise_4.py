import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)  # Flatten the image
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)

# QUESTION 4

# Answer to question 1:
# Changing the batch_size from 32 to 64 triggers a structural bug because the labels and latent space samples were being
# created with a fixed batch size. When the batch size is dynamically altered, this causes mismatches during training.
# The fix is to use real_samples.size(0) to dynamically match the batch size in both the labels and latent space samples.

# Answer to question 2:
# The cosmetic bug is related to how the output is displayed during training. The clear_output(wait=False) method would
# immediately clear the output, causing flickering or an undesirable refresh behavior. By changing to clear_output(wait=True),
# the output will only clear when new content is available, improving the user experience and avoiding unnecessary flickering.

def train_gan(batch_size: int = 32, num_epochs: int = 100, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    real_samples, mnist_labels = next(iter(train_loader))

    fig = plt.figure()
    for i in range(16):
        sub = fig.add_subplot(4, 4, 1 + i)
        sub.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
        sub.axis('off')

    fig.tight_layout()
    fig.suptitle("Real images")
    display(fig)
    time.sleep(5)

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    lr = 0.0001
    loss_function = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for n, (real_samples, mnist_labels) in enumerate(train_loader):
            real_samples = real_samples.to(device=device)

            # Structural bug fix: Dynamically use real_samples.size(0) to match batch size
            real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device=device) 
            latent_space_samples = torch.randn((real_samples.size(0), 100)).to(device=device)

            #real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
            #latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((generated_samples.size(0), 1)).to(device=device)

            # Training the discriminator
            optimizer_discriminator.zero_grad()
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Training the generator
            optimizer_generator.zero_grad()
            latent_space_samples = torch.randn((real_samples.size(0), 100)).to(device=device)
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

            if n == batch_size - 1:
                name = f"Generate images\n Epoch: {epoch} Loss D.: {loss_discriminator:.2f} Loss G.: {loss_generator:.2f}"
                generated_samples = generated_samples.detach().cpu().numpy()
                fig = plt.figure()
                for i in range(16):
                    sub = fig.add_subplot(4, 4, 1 + i)
                    sub.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
                    sub.axis('off')
                fig.suptitle(name)
                fig.tight_layout()

                # By changing the wait to True, we only clear the output when a new one is received instead of immediately, without waiting for the new output.
                clear_output(wait=True) 
                
                #clear_output(wait=False)
                display(fig)

train_gan(batch_size=32, num_epochs=100)
