import os

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.utils as vutils
import matplotlib.ticker as ticker

from torchinfo import summary
from torch import optim
from Custom_dataset import CustomDataset
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from resnet import Auto_Encoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Resnet_Trainer(object):
    def __init__(self, args):
        self.args = args

        if args.visualize:
            self.samples = {}

        self.train_data, self.val_data, self.test_data = self.get_data(args.data, args.sample_rate)
        self.model = self.get_model('autoencoder', args.params)
        self.iter = 0
        self.epoch = 0
        self.loss = torch.nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = self.get_optimizer()
        self.logger = SummaryWriter(log_dir=args.log_dir)

    def get_optimizer(self, lr=5e-4):
        return optim.Adam(self.model.parameters(), lr=lr)

    def save_mixed_dataset(self, loader, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for batch_idx, (images, labels) in enumerate(loader):
            for i in range(images.size(0)):
                img_tensor = images[i]
                label = labels[i].item()

                # 将Tensor转换为PIL图像
                img = transforms.ToPILImage()(img_tensor)

                # 创建标签子目录
                label_dir = os.path.join(save_dir, str(label))
                os.makedirs(label_dir, exist_ok=True)

                # 定义图像文件名
                img_filename = f'image_{batch_idx * loader.batch_size + i}.png'
                img_path = os.path.join(label_dir, img_filename)

                # 保存图像
                img.save(img_path)

    def get_combine_data(self, data_a, data_b, samples_rate):
        train_a, val_a, test_a = data_a[0], data_a[1], data_a[2]
        train_b, val_b, test_b = data_b[0], data_b[1], data_b[2]

        # compute samples
        samples_train = int(len(train_b) * samples_rate / 10)
        samples_test = int(len(test_b) * samples_rate / 10)
        samples_val = int(len(val_b) * samples_rate / 10)

        # get correspond subset
        test_a = self.get_subset(test_a, samples_test)
        train_a = self.get_subset(train_a, samples_train)
        val_a = self.get_subset(val_a, samples_val)

        if self.args.visualize and self.args.data == 'combine':
            self.samples['train'] = [train_a, train_b]
            self.samples['test'] = [test_a, test_b]
            self.samples['val'] = [val_a, val_b]

        # combine data
        combined_train = torch.utils.data.ConcatDataset([train_a, train_b])
        combined_test = torch.utils.data.ConcatDataset([test_a, test_b])
        combined_val = torch.utils.data.ConcatDataset([val_a, val_b])
        return combined_train, combined_test, combined_val

    def get_data(self, data, samples_rate=0.0):
        print('begin getting data')
        print('data type:{} samples rate:{}'.format(data, samples_rate))
        if data == 'combine':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

            # download mnist
            mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            mnist_train, mnist_val = random_split(mnist_train, [0.8, 0.2])

            # transform mnist
            mnist_train = CustomDataset(subset=mnist_train, offset=0)
            mnist_val = CustomDataset(subset=mnist_val, offset=0)

            # down fashion-mnist
            fashion_mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
            fashion_mnist_train, fashion_mnist_val = random_split(fashion_mnist_train, [0.8, 0.2])

            # transform fashion-mnist
            fashion_mnist_train = CustomDataset(subset=fashion_mnist_train, offset=10)
            fashion_mnist_val = CustomDataset(subset=fashion_mnist_val, offset=10)
            fashion_mnist_test.targets += 10

            if samples_rate != 0.0:
                if samples_rate > 0.0:
                    combined_train, combined_test, combined_val = self.get_combine_data(
                                            data_a=[fashion_mnist_train, fashion_mnist_val, fashion_mnist_test],
                                            data_b=[mnist_train, mnist_val, mnist_test],
                                            samples_rate=samples_rate
                    )
                else:
                    combined_train, combined_test, combined_val = self.get_combine_data(
                                            data_a=[mnist_train, mnist_val, mnist_test],
                                            data_b=[fashion_mnist_train, fashion_mnist_val, fashion_mnist_test],
                                            samples_rate=-samples_rate
                    )
            else:  # concat data
                combined_train = torch.utils.data.ConcatDataset([mnist_train, fashion_mnist_train])
                combined_test = torch.utils.data.ConcatDataset([mnist_test, fashion_mnist_test])
                combined_val = torch.utils.data.ConcatDataset([mnist_val, fashion_mnist_val])

            train_loader = torch.utils.data.DataLoader(combined_train, batch_size=64, shuffle=True)
            test_loader = torch.utils.data.DataLoader(combined_test, batch_size=64, shuffle=False)
            val_loader = torch.utils.data.DataLoader(combined_val, batch_size=64, shuffle=False)
            if self.args.save_data:
                root_path = './mix_data_{:.4f}'.format(samples_rate)
                train = os.path.join(root_path, 'train')
                test = os.path.join(root_path, 'test')
                val = os.path.join(root_path, 'val')
                self.save_mixed_dataset(train_loader, train)
                self.save_mixed_dataset(test_loader, test)
                self.save_mixed_dataset(val_loader, val)

        elif data == 'Fashion':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            fashion_mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
            fashion_mnist_train, fashion_mnist_val = random_split(fashion_mnist_train, [0.8, 0.2])

            fashion_mnist_train = CustomDataset(subset=fashion_mnist_train, offset=10)
            fashion_mnist_val = CustomDataset(subset=fashion_mnist_val, offset=10)
            fashion_mnist_test.targets += 10

            # adjust label of fashion-mnist
            train_loader = torch.utils.data.DataLoader(fashion_mnist_train, batch_size=64, shuffle=True)
            test_loader = torch.utils.data.DataLoader(fashion_mnist_test, batch_size=64, shuffle=False)
            val_loader = torch.utils.data.DataLoader(fashion_mnist_val, batch_size=64, shuffle=False)
        elif data == 'MNIST':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            mnist_train, mnist_val = random_split(mnist_train, [0.8, 0.2])

            mnist_train = CustomDataset(subset=mnist_train, offset=0)
            mnist_val = CustomDataset(subset=mnist_val, offset=0)

            train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
            test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)
            val_loader = torch.utils.data.DataLoader(mnist_val, batch_size=64, shuffle=False)
        else:
            raise NotImplemented

        return train_loader, val_loader, test_loader

    def get_subset(self, data, samples=100):
        selected_indices = []
        _, label_test = next(iter(data))
        if label_test >= 10:
            upper = 20
            lower = 10
        else:
            upper = 10
            lower = 0

        label_count = {label: 0 for label in range(lower, upper)}

        for idx in range(len(data)):
            _, label = data[idx]

            if lower <= label < upper and label_count[label] < samples:
                selected_indices.append(idx)
                label_count[label] += 1

                if all(count >= samples for count in label_count.values()):
                    break

        return Subset(data, selected_indices)

    def get_model(self, model, params):
        if model == 'autoencoder':
            model = Auto_Encoder(**params)
            if self.args.visualize:
                path = os.path.join(self.args.save_path, 'current.pth')
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            raise NotImplemented
        return model

    def log_add_img(self, names, img, iter):
        """ Add an image in tensorboard"""
        if self.logger is None:
            return
        self.logger.add_image(tag=names, img_tensor=img, global_step=iter)

    def log_add_scalar(self, names, scalar, iter):
        """ Add scalar value in tensorboard """
        if self.logger is None:
            return
        if isinstance(scalar, dict):
            self.logger.add_scalars(main_tag=names, tag_scalar_dict=scalar, global_step=iter)
        else:
            self.logger.add_scalar(tag=names, scalar_value=scalar, global_step=iter)

    def save_network(self, model, optimizer, path):
        torch.save({'iter': self.iter,
                    'global_epoch': self.epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   path)

    def get_encoded_data(self, loader):
        encoded_data = []
        with torch.no_grad():
            for data, label in loader:
                data = data.to(self.device)
                encoded = self.model.encode(data)

                encoded = encoded.permute(0, 2, 3, 1).contiguous()  # [batch_size, height, width, channels]

                # encoded = encoded.view(-1, encoded.size(3))
                # 

                encoded_data.append(encoded.cpu())
        return torch.cat(encoded_data)

    def get_ori_data(self, loader):
        code_data = []
        for data, label in loader:
            code_data.append(data)
        return torch.cat(code_data)


    def get_visualize_data(self, samples_loader, intact_loader, encode):
        if encode:
            samples_encoded = self.get_encoded_data(samples_loader)
            intact_encoded = self.get_encoded_data(intact_loader)
        else:
            samples_encoded = self.get_ori_data(samples_loader)
            intact_encoded = self.get_ori_data(intact_loader)
        return samples_encoded, intact_encoded

    def visualize_pca(self, data_samples, data_intact, name, encode):
        samples_loader = torch.utils.data.DataLoader(data_samples, batch_size=64, shuffle=True)
        intact_loader = torch.utils.data.DataLoader(data_intact, batch_size=64, shuffle=False)
        samples_encoded, intact_encoded = self.get_visualize_data(samples_loader, intact_loader, encode)
        combined = torch.cat((samples_encoded, intact_encoded))

        scaler = StandardScaler()
        standard_combined = scaler.fit_transform(combined)

        pca = PCA(n_components=1)

        pca_results = pca.fit_transform(standard_combined)

        samples_rate = len(samples_encoded) / len(combined)
        print(samples_rate)
        intact_rate = len(intact_encoded) / len(combined)

        samples_pca = pca_results[:len(samples_encoded)]
        intact_pca = pca_results[len(samples_encoded):]

        sns.kdeplot(samples_pca.squeeze(), color='blue', label='samples')
        sns.kdeplot(intact_pca.squeeze(), color='red', label='intact')
        sns.kdeplot(pca_results.squeeze(), color='green', label='total')

        for line in plt.gca().get_lines()[:2]:
            line.set_ydata(line.get_ydata() * (samples_rate if line.get_label() == 'samples' else intact_rate))

        name = name if encode else name + '_ori'
        plt.legend()
        plt.title('PCA of MNIST and Fashion-MNIST Encoded Data (' + name + ')')
        plt.xlabel('PCA Feature')
        plt.ylabel('PCA Density')
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.025))

        path = './pca_distribution/{}_{:.4f}/'.format(self.args.data, self.args.sample_rate)
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + name + '.png')
        plt.show()

    def visualize_distribution(self):
        self.model.eval()
        self.model.to(self.device)
        for key, value in self.samples.items():
            self.visualize_pca(value[0], value[1], key, True)
            self.visualize_pca(value[0], value[1], key, False)

    def train_one_epoch(self, model, optimizer, loader):
        model.train()
        loss_value = 0.0
        for batch_idx, (inputs, _) in enumerate(loader):
            inputs = inputs.to(self.device)
            model = model.to(inputs.device)

            outputs = model(inputs)

            loss = self.loss(outputs, inputs)

            with torch.no_grad():
                loss_value += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.iter += 1

        self.log_add_scalar('Train/loss', loss_value / len(self.train_data), self.epoch)
        print('Epoch {}, Iter {}, Loss {}'.format(self.epoch, self.iter, loss_value / len(self.train_data)))
        save_path = os.path.join(self.args.save_path, f"current.pth")
        self.save_network(self.model, self.optimizer, save_path)

    def test(self, model, loader, type='Val'):
        loss = 0
        minor_ori_images = []
        minor_reco_images = []
        with torch.no_grad():
            for data, labels in loader:
                data = data.to(self.device)
                output = model(data)
                if loss == 0:
                    original_images = vutils.make_grid(data, normalize=True)
                    recon_images = vutils.make_grid(output, normalize=True)

                    self.log_add_img(type + '/ori', original_images, self.epoch)
                    self.log_add_img(type + '/reco', recon_images, self.epoch)
                mask = labels >= 10
                ori_selected = data[mask]
                reco_selected = output[mask]
                if len(ori_selected) > 0 and sum(len(img) for img in minor_reco_images ) < 32:
                    minor_ori_images.append(ori_selected)
                    minor_reco_images.append(reco_selected)

                loss += self.loss(data, output)

            minor_reco_images = vutils.make_grid(torch.cat(minor_reco_images[:32]))
            minor_ori_images = vutils.make_grid(torch.cat(minor_ori_images[:32]))
            self.log_add_img(type + '/ori_minor', minor_ori_images, self.epoch)
            self.log_add_img(type + '/reco_minor', minor_reco_images, self.epoch)

            loss = loss.item() / len(loader)
            self.log_add_scalar(type + '/loss', loss, self.epoch)

    def fit(self):
        os.makedirs(self.args.save_path, exist_ok=True)
        summary(self.model, input_size=(64, 1, 28, 28), col_names=["input_size", "output_size", "num_params", "params_percent"])
        for epoch in range(self.args.epochs):
            self.train_one_epoch(self.model, self.optimizer, self.train_data)
            if epoch % 10 == 0:
                save_path = os.path.join(self.args.save_path, f"epoch_{epoch:03d}.pth")
                self.save_network(self.model, self.optimizer,  save_path)

            if epoch % 2 == 0:
                self.test(self.model, self.val_data, type='Val')
            self.epoch += 1
        self.test(self.model, self.test_data, type='Test')
