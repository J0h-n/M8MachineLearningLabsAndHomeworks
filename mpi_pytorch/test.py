import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16
from mpi4py import MPI

classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale',
'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
'bottles', 'bowls', 'cans', 'cups', 'plates',
'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
'furniture	bed', 'chair', 'couch', 'table', 'wardrobe',
'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
'bear', 'leopard', 'lion', 'tiger', 'wolf',
'bridge', 'castle', 'house', 'road', 'skyscraper',
'cloud', 'forest', 'mountain', 'plain', 'sea',
'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
'crab', 'lobster', 'snail', 'spider', 'worm',
'baby', 'boy', 'girl', 'man', 'woman',
'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
'maple', 'oak', 'palm', 'pine', 'willow',
'bicycle', 'bus', 'motorcycle', 'pickup', 'truck', 'train',
'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor')

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    p = comm.Get_size()

    img_size = (300, 400)

    transform = transforms.Compose(
      [transforms.Resize(img_size),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if my_rank == 0:
        #итерация по тестовому датасету
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
        batch_size = 1
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
        it = 0
        for i, data in enumerate(testloader, 0):
            if it == 5:
                break
            inputs, labels = data
            #отправить моделям на предсказание
            for i in range(1, p):
                comm.send(inputs, i)

            preds = []
            #получить предсказания и сделать ансамбль голосов
            for procid in range(1, p):
                preds.append(comm.recv(source = procid))
            #среднее арифметическое
            buf = torch.zeros_like(preds[0])
            for p1 in preds:
                for idx in range(len(p1)):
                    buf[idx] += p1[idx]
                    
            for idx in range(len(buf)):
                buf[idx] /= len(preds)
            it += 1
            print('Predicted: ', ' '.join(f'{classes[torch.argmax(buf)]}'))
        for i in range(1, p):
            comm.send("end", i)
            
    else:
        #загрузить свою модель
        PATH = f'./net_{my_rank}.pth'
        net = vgg16(pretrained=False)
        num_ftrs = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_ftrs, 10)
        net.load_state_dict(torch.load(PATH))

        while True:
            #получить предсказание по одной картинке
            image = comm.recv(source = 0)
            if image == "end":
                break
            outputs = net(image)
            comm.send(outputs, dest = 0)
    MPI.Finalize
