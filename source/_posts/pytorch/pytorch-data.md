---
title: 基于pytorch的代码完整流程
date: 2021-05-20 10:37:31
categories: Deep Learning
tags:
- Pytorch
- Training Methods
author: Fanrencli
---
## 基于Pytorch深度学习完整训练过程展示

过去文章只涉及相关的主干特征提取网络的介绍以及一些经典算法的关键代码复现，并没有涉及到整个训练过程的代码展示，而本文就之前给出的特征提取网络，选取其中一种特征提取网络进行经典的网络分类。

### 数据处理

首先定义我们的数据类`MydataLoader`,注意这个类必须要继承`torch.utils.data.Dataset`类，然后必须要实现三个方法：`__init__(self)`、`__len__(self)`、`__getitem__(self,idx)`：
- 第一个方式用于类的初始化，数据加载和处理过程都在这个方法中实现
- 第二个方法用于获取数据的数量
- 第三个方法用于根据索引获取数据

由于做深度学习时候的数据量通常很大，我们不能一次性加载到内存中，所以我们需要数据的时候就获取数据的路径和类别，然后在第三个方法中实现根据路径和类别来读取图片并进行一系列的操作，通过这样实现数据的输入。

本文的数据放在`UECFOOD256`文件夹下，里面包含256类的数据，每一类数据在一个文件夹下，每个文件夹的名称就是数据的类别。按照这个标准准备好数据，之后再初始化方法中对根目录的文件夹中内容进行读取，获取所有文件夹下的图片路径和类别（就是文件夹的名称），写入txt文件中。这样所有的数据相关的路径和类别就存入了txt文件中，当我们需要读取数据时，就读入txt文件中的内容，每一行就是一个图片路径，通过索引将数据输入到网络中。
```python
class MydataLoader(Dataset):
    # resize用于将数据统一到相同的尺寸大小，mode用于区别数据是用于训练or验证or测试，root用于表示根目录
    def __init__(self, root='UECFOOD256\\', resize=(224,224), mode="train"):
        super(MydataLoader,self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root,name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        self.images,self.labels = self.loadCSV()
        print(len(self.images))
        if mode == 'Training':
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        if mode == 'val':
            self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        if mode == 'test':
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]
    #第一次训练时会将数据保存到txt，如果txt存在则世界读取数据
    def loadtxt(self):
        if not os.path.exists('train.txt'):
            img = []
            for name in self.name2label.keys():
                img += glob.glob(os.path.join(self.root, name, "*.jpg" ))
            random.shuffle(img)
            with open('train.txt','a') as f:
                for i in img:
                    name = i.split(os.sep)[-2]
                    label = self.name2label[name]
                    f.write(i+","+str(label)+'\n')
                print('save the data into train.txt')
            images ,labels= [],[]
            with open('train.txt','r') as f:
                while True:
                    data = f.readline()
                    if data=='':
                        break
                    [img,label] = data.split(',')
                    images.append(img)
                    labels.append(int(label))
        images ,labels= [],[]
        with open('train.txt','r') as f:
            while True:
                data = f.readline()
                if data=='':
                    break
                img,label = data.split(',')
                images.append(img)
                labels.append(int(label))
        return images,labels
    def __len__(self):
        return len(self.images)
    # 根据素银获取图片的路径，并对图片进行才做包括调整尺寸，旋转，中心剪切，归一化等等
    def __getitem__(self,idx):
        img ,label = self.images[idx],self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(img),
            transforms.Resize((int(self.resize[0]*1.25),int(self.resize[1]*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img,label

```

### 模型训练

模型训练的过程通常按照固有的流程，首先我们要设置一些参数：batch_size（每批次训练数据大小，根据GPU的等级设计），学习率lr（这个是深度学习的概念，通常设置为1e-3），迭代次数epochs等等。

然后就是固有的流程：初始化模型、优化器选择、loss函数设计、开始训练。在训练的过程中，我们通过引入`tqdm`库来实时跟踪模型的训练进度。关于这个方式的使用请读者自行百度。其中每一个迭代我们都进行计算一下精度，评估验证集的数据准确性，评估的过程我们不需要计算梯度。

```python
def evaluate(model,db):
    correct = 0
    total = len(db.dataset)
    for x,y in db:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred,y).sum().float().item()
    return correct/total

# 设置训练的相关参数
batch_size = 1
lr = 1e-3
epochs = 100
device = torch.device('cuda')

# 加载三种数据
train_db = MydataLoader()
val_db = MydataLoader(mode = 'val')
test_db = MydataLoader(mode = 'test')
train_loader = DataLoader(train_db,batch_size = batch_size,shuffle = True)
val_loader = DataLoader(val_db,batch_size = batch_size,shuffle = True)
test_loader = DataLoader(test_db,batch_size = batch_size,shuffle = True)

# 模型初始化以及设置训练的次数和相关参数
best_acc = 0
model = ResNet101(256).to(device)
optimizer = optim.Adam(model.parameters(),lr=lr)
criteon = nn.CrossEntropyLoss()
with open("train.txt","r") as f:
    train_lines = f.readlines()
epoch_size = len(train_lines)//batch_size

# 开始训练
for epoch in range(epochs):
    with tqdm(total = epoch_size, desc = f'Epoch {epoch + 1}/{epochs}',mininterval=0.3 ) as pbar:
        for step,(x,y) in enumerate(train_loader):
            if step>epoch_size:
                break
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criteon(logits,y)

            loss.backward()
            optimizer.step()
            pbar.update(1)
        val_acc = evaluate(model,val_loader)
        if val_acc>best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),'best_model.h5')

print("best_acc:",best_acc)
model.load_state_dict(torch.load('best_model.h5'))
print('load the model')
test_acc = evaluate(model,test_loader)
```