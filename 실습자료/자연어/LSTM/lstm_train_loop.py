import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def calculate_accuracy(output_pred, labels) :
    predicted = torch.argmax(output_pred, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc

def evaluate(model, valid_dataloader, criterion, device) :
    print("Start valid ... ")

    val_loss = 0
    val_correct = 0
    val_total = 0

    model.eval()
    with torch.no_grad() :
        for inputs_x, inputs_y in valid_dataloader :
            inputs_x, inputs_y = inputs_x.to(device), inputs_y.to(device)

            output_pred = model(inputs_x)

            # loss
            loss = criterion(output_pred, inputs_y)

            # 정확도와 손실을 계산
            val_loss += loss.item()
            val_correct += calculate_accuracy(output_pred, inputs_y) * inputs_y.size(0)
            val_total += inputs_y.size(0)

        val_acc = val_correct / val_total
        val_loss /= len(valid_dataloader)

    return val_loss, val_acc

def lstm_train_loop(y_train, y_valid, padded_x_train, padded_x_valid ,model,device, num_epoch) :
    print("CPU와 CUDA 중 다음 기기로 학습 진행 : ", device)

    best_val_loss = float('inf')

    # tensor 형태로 변환 작업
    train_label_tensor = torch.tensor(np.array(y_train))
    valid_label_tensor = torch.tensor(np.array(y_valid))

    # 데이터셋, 데이터로더 정의
    encoded_train= torch.tensor(padded_x_train).to(torch.int64)
    train_dataset = TensorDataset(encoded_train, train_label_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    encoded_valid = torch.tensor(padded_x_valid).to(torch.int64)
    valid_dataset = TensorDataset(encoded_valid, valid_label_tensor)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    # 하이퍼파라미터 지정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    num_epochs = num_epoch

    # train loop
    for epoch in range(num_epochs) :
        print("Start train ... ")

        train_loss = 0
        train_correct = 0
        train_total = 0
        model.train()

        for inputs_x , inputs_y in train_loader:
            inputs_x, inputs_y = inputs_x.to(device), inputs_y.to(device)

            output_pred = model(inputs_x)
            loss = criterion(output_pred, inputs_y)

            # backward pass and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training acc and loss
            train_loss += loss.item()
            train_correct += calculate_accuracy(output_pred=output_pred, labels=inputs_y) * inputs_y.size(0)
            train_total += inputs_y.size(0)
            print("train loss : ", loss.item() , "epoch : ", epoch+1)

        train_acc = round(train_correct / train_total, 4)
        train_loss /= len(train_loader)

        # Validation
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train loss {train_loss:.4f}, Train acc : {train_acc:.4f}")
        print(f"Validation loss {val_loss:.4f}, Validation acc : {val_acc:.4f}")

        # 체크 포인트 저장 val loss 기준 최소 값일 경우 저장 되는 형태
        # best_val_loss = float('inf')
        if val_loss < best_val_loss :
            print(f"체크포인트 저장 val_loss : {val_loss}, best_val_loss : {best_val_loss}")
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_checkpoint.pth")


