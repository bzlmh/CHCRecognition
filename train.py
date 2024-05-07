import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
import numpy as np
from tensorboardX import SummaryWriter
from torch import device
from torchvision import transforms
from torchsummary import summary
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from dataloader import dataloader
from network.efficientv2_msa import efficientnetv2_s

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
result_folder = './confidencs'

def calculate_metrics(labels, predicted):
    precision = precision_score(labels, predicted, average='weighted')
    recall = recall_score(labels, predicted, average='weighted')
    f1 = f1_score(labels, predicted, average='weighted')
    matrix = confusion_matrix(labels, predicted)
    return precision, recall, f1, matrix

def save_results_to_csv(epoch, all_labels, all_predicted_top1, all_confidences_list_top1, correct_top1):
    csv_file_path = os.path.join(result_folder, f'epoch_{epoch}.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['True_Label', 'Predicted_Label', 'Confidence', 'Correct'])
        correct_top1_int = [int(correct) for correct in correct_top1]
        for label, predicted, confidence, correct in zip(all_labels, all_predicted_top1, all_confidences_list_top1, correct_top1_int):
            writer.writerow([label, predicted, confidence, correct])

def valid(epoch, net, test_loader, writer):
    print("epoch %d Start verifying..." % epoch)

    with torch.no_grad():
        correct_top1 = 0
        correct_top2 = 0
        correct_top3 = 0
        correct_top4 = 0
        correct_top5 = 0
        total = 0
        all_confidences_list_top1 = []
        all_labels = []
        all_predicted_top1 = []
        all_probabilities_top1 = []

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences_top1, predicted_top1 = torch.max(probabilities, 1)
            _, predicted_topk = torch.topk(probabilities, 5, dim=1)
            total += labels.size(0)
            correct_top1 += (predicted_top1 == labels).sum().item()
            correct_top2 += torch.sum(predicted_topk[:, :2] == labels.view(-1, 1)).item()
            correct_top3 += torch.sum(predicted_topk[:, :3] == labels.view(-1, 1)).item()
            correct_top4 += torch.sum(predicted_topk[:, :4] == labels.view(-1, 1)).item()
            correct_top5 += torch.sum(predicted_topk[:, :5] == labels.view(-1, 1)).item()

            all_labels.extend(labels.cpu().numpy())
            all_confidences_list_top1.extend(confidences_top1.cpu().numpy())
            all_predicted_top1.extend(predicted_top1.cpu().numpy())
            all_probabilities_top1.extend(probabilities[:, 0].cpu().numpy())

        acc_1 = 100 * correct_top1 / total
        acc_2 = 100 * correct_top2 / total
        acc_3 = 100 * correct_top3 / total
        acc_4 = 100 * correct_top4 / total
        acc_5 = 100 * correct_top5 / total

        print('The identification accuracy of the %d epoch (TOP-1) is：%.3f' % (epoch, acc_1))
        print('The identification accuracy of the %d epoch (TOP-2) is：%.3f' % (epoch, acc_2))
        print('The identification accuracy of the %d epoch (TOP-3) is：%.3f' % (epoch, acc_3))
        print('The identification accuracy of the %d epoch (TOP-4) is：%.3f' % (epoch, acc_4))
        print('The identification accuracy of the %d epoch (TOP-5) is：%.3f' % (epoch, acc_5))

        writer.add_scalar('valid_acc_1', acc_1, global_step=epoch)
        writer.add_scalar('valid_acc_2', acc_2, global_step=epoch)
        writer.add_scalar('valid_acc_3', acc_3, global_step=epoch)
        writer.add_scalar('valid_acc_4', acc_4, global_step=epoch)
        writer.add_scalar('valid_acc_5', acc_5, global_step=epoch)

        for i, (confidence, label, predicted) in enumerate(
                zip(all_confidences_list_top1, all_labels, all_predicted_top1)):
            writer.add_scalar(f'confidence/sample_{i}_top1', confidence, global_step=epoch)
            writer.add_scalar(f'label/sample_{i}_top1', label, global_step=epoch)
            writer.add_scalar(f'predicted/sample_{i}_top1', predicted, global_step=epoch)

        csv_filename = f'top1_probabilities_epoch_{epoch}.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Sample', 'TOP-1 Probability'])
            for i, probability in enumerate(all_probabilities_top1):
                csv_writer.writerow([f'sample_{i}', probability])

        print(f'TOP-1 probabilities saved to {csv_filename}')

        precision, recall, f1, matrix = calculate_metrics(all_labels, all_predicted_top1)
        print('Top-1 Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}'.format(precision, recall, f1))
        writer.add_scalar('valid_precision_top1', precision, global_step=epoch)
        writer.add_scalar('valid_recall_top1', recall, global_step=epoch)
        writer.add_scalar('valid_f1_top1', f1, global_step=epoch)

        avg_confidence_top1 = np.mean(all_confidences_list_top1)
        writer.add_scalar('Average Confidence_top1', avg_confidence_top1, global_step=epoch)
        print('Average Confidence_top1: {:.4f}'.format(avg_confidence_top1))

def train(epoch, net, criterion, optimizer, train_loader, writer, save_iter=100):
    net.train()
    sum_loss = 0.0
    total = 0
    correct = 0

    start_time = time.time()

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            labels = labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        if (i + 1) % save_iter == 0:
            batch_loss = sum_loss / save_iter
            acc = 100 * correct / total

            print(f'epoch: {epoch}, batch: {i + 1}, loss: {batch_loss:.03f}, acc: {acc:.04f}')
            writer.add_scalar('train_loss', batch_loss, global_step=i + len(train_loader) * epoch)
            writer.add_scalar('train_acc', acc, global_step=i + len(train_loader) * epoch)

            for name, layer in net.named_parameters():
                writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(),
                                     global_step=i + len(train_loader) * epoch)
                writer.add_histogram(name + '_data', layer.cpu().data.numpy(),
                                     global_step=i + len(train_loader) * epoch)

            total = 0
            correct = 0
            sum_loss = 0.0

    end_time = time.time()
    total_time = end_time - start_time

    return batch_loss, acc, total_time

if __name__ == "__main__":

    epochs = 100
    batch_size = 64
    lr = 0.01

    data_path = r'./data2'
    log_path = r'logs/batch_{}_lr_{}'.format(batch_size, lr)
    save_path = r'checkpoints/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = dataloader(path=data_path, transform=transform)
    print("Training dataset size:", dataset.train_size)
    print("Testing dataset size:", dataset.test_size)
    print("Number of classes:", dataset.num_classes)
    trainloader, testloader = dataset.get_loader(batch_size)

    net = efficientnetv2_s(dataset.num_classes)
    if torch.cuda.is_available():
        net = net.to(device)

    print('Network structure:\n')
    summary(net, input_size=(3, 64, 64), device='cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    writer = SummaryWriter(log_path)
    save_interval = 10

    for epoch in range(epochs):
        train_loss, train_acc, train_time = train(epoch, net, criterion, optimizer, trainloader, writer=writer)
        valid(epoch, net, testloader, writer=writer)

        if (epoch + 1) % save_interval == 0:
            print(f"Epoch {epoch} ended, saving model...")
            torch.save(net.state_dict(), save_path + f'Network_{epoch:03d}.pth')
