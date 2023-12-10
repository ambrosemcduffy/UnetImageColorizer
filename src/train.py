import torch
import torchvision.utils as vutils

from torchvision import datasets, transforms
from model import AutoEncoder, CustomDataset
from torchvision.transforms import Grayscale
from torch.utils.tensorboard import SummaryWriter
from unet import Unet
torch.cuda.empty_cache()

data_dir = '/mnt/e/autoEncoder/data/denoise_and_color'
input_transforms = transforms.Compose([
    Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

target_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

batchsize = 16
dataset = CustomDataset(root_dir=data_dir, input_transform=input_transforms, target_transform=target_transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=24)

writer = SummaryWriter("runs/colorAutoEncoder_1")

print(dataloader)

# Putting model on GPU if detected
model = Unet()
if torch.cuda.is_available():
    print("GPU Found!::", torch.cuda.get_device_name())
    print("TOTAL GPU Memory: ", torch.cuda.max_memory_allocated())
    model = model.cuda()

# Add the model graph. We only need to do this once.
dummy_input = torch.zeros(1, 3, 288, 512).cuda()  # Replace with actual input shape your model uses
writer.add_graph(model, dummy_input)

# for param in model.resnet.parameters():
#    param.requires_grad = False
    

criterion = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
n_epochs = 100

for epoch in range(n_epochs):
    train_loss = 0.0
    old_loss = 0.0
    for i, (data, target) in enumerate(dataloader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        output = model(data)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        writer.add_scalar("training loss", loss.item(), epoch)
        
        # if epoch == 9:
        #    print("Un Freezing Model..")
        #    for param in model.resnet.parameters():
        #        param.requires_grad = True
        
        # Add images to TensorBoard
        if i % 100 == 0:  # Adjust this value to control how often you want to log images
            img_grid_input = vutils.make_grid(data[:4], normalize=True)
            img_grid_output = vutils.make_grid(output[:4], normalize=True)
            img_grid_target = vutils.make_grid(target[:4], normalize=True)
            writer.add_image('input_images', img_grid_input, global_step=epoch)
            writer.add_image('output_images', img_grid_output, global_step=epoch)
            writer.add_image('target_images', img_grid_target, global_step=epoch)
            torch.save(model.state_dict(), "weights/modelUnet_v2.pth")


    print("Epoch {} loss {}".format(epoch+1, train_loss/len(dataloader)))

writer.close()
torch.save(model.state_dict(), "weights/model{}Unet2.pth".format(epoch))
