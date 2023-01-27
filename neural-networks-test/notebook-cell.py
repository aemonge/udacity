# TODO: Build and train your network
import torchvision.models as models
#model = models.squeezenet1_0(pretrained=True) # Slow
#model = models.vgg16(pretrained=True) # Slow
model = models.alexnet(pretrained=True)
criterion = Nn.CrossEntropyLoss()
optimizer = Optim.SGD(model.parameters(), lr=0.3) # , momentum=0.9)

test_loader_count = len(dataloaders['test'].dataset)
train_loader_count = len(dataloaders['train'].dataset)


EPOCHS = 5

print(f"Test Load N: {test_loader_count}, Trian Load N: {train_loader_count}, Epochs: {EPOCHS}")

for epoch in range(EPOCHS):
    tr_loss = 0

    loading_str = "⋅"
    for (images, labels) in iter(dataloaders["train"]):
        # images = images.view()
        # labels = labels [0].long() ?

        print(loading_str + "|", end='\r')
        output = model(images)

        print(loading_str + "/", end='\r')
        loss = criterion(output, labels)

        print(loading_str + "-", end='\r')
        loss.backward()

        print(loading_str + "\\", end='\r')
        optimizer.step()

        print(loading_str + "|", end='\r')
        tr_loss += loss.item()

        print(loading_str + "/", end='\r')
        loading_str += '⋅'
    else:
        t_loss = 0
        t_correct = 0

        with torch.no_grad():
            model.eval()
            for (images, labels) in iter(dataloaders["test"]):
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                t_loss += loss.item()

                ps = torch.exp(log_ps)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                t_correct += equals.sum().item()

        model.train()

        training_loss = tr_loss / train_loader_count
        test_loss = t_loss / test_loader_count

        print(flush=True)
        print(f"  Epoch: {epoch},",
              f" Losses [ training: {training_loss:.4f}, test: {test_loss:.4f} ],",
              f" Accuracy: {(t_correct * 100 / test_loader_count):.2f}%",
              f"\n    TR: {tr_loss}, TL: {t_loss}",
              f" nCorrect: {t_correct},"
        )
