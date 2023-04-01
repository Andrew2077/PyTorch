import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Training step
def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn: torch.nn.Module,
    epoch: int,
    show_progress: bool =True,
    show_every: int = 1,
    device: torch.device = device,
):
    loss_train, acc_train = 0, 0

    model.train()

    for batch, (X, y) in enumerate(data_loader):
        x, y = X.to(device), y.to(device)

        logits = model(x)
        ypreds = torch.softmax(logits, dim=1).argmax(dim=1)
        loss = loss_fn(logits, y)
        acc = accuracy_fn(y_pred=ypreds, y_true=y)

        loss_train += loss
        acc_train += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_train /= len(data_loader)
    acc_train /= len(data_loader)

    if show_progress :
        if epoch % show_every == 0:
            # clear_output()
            print(
                f"Epoch: {epoch+1}\n------ train Acc = {acc_train:.2f}% || train Loss = {loss_train:.5f}"
            )
    
    return loss_train, acc_train,


## Testing Step
def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn: torch.nn.Module,
    epoch: int,
    show_progress: bool = True,
    show_every: int = 1,
    device: torch.device = device,
):
    loss_test, acc_test = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            ypreds = torch.softmax(logits, dim=1).argmax(dim=1)
            loss = loss_fn(logits, y)
            acc = accuracy_fn(y_pred=ypreds, y_true=y)

            loss_test += loss
            acc_test += acc

        loss_test /= len(data_loader)
        acc_test /= len(data_loader)
        if show_progress : 
            if epoch % show_every == 0:
                # clear_output()
                print(f"------ test Acc = {acc_test:.2f}% || test Loss = {loss_test:.5f}")
        return loss_test, acc_test

def train(
    model : torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_function :torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_function,
    epochs :int ,
    show_progress: bool = True,
    show_every: int = 1,
    device: torch.device = device,
):
    results = {
        "Train_loss": [],
        "Train_acc": [],
        "Test_loss": [],
        "Test_acc": [],
    }
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model = model,
            data_loader= train_dataloader,
            loss_fn= loss_function,
            optimizer= optimizer,
            accuracy_fn= accuracy_function,
            epoch = epoch ,
            show_progress= show_progress,
            show_every = show_every,
            device= device,   
        )
        
        test_loss, test_acc = test_step(
            model = model,
            data_loader= test_dataloader,
            loss_fn= loss_function,
            optimizer= optimizer,
            accuracy_fn= accuracy_function,
            epoch = epoch ,
            show_progress= show_progress,
            show_every = show_every,
            device= device,   
        )
        results["Train_loss"].append(train_loss.item())
        results["Train_acc"].append(train_acc)
        results["Test_loss"].append(test_loss.item())
        results["Test_acc"].append(test_acc)
    
    return results

def eval_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
):

    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            ypreds = model(X)

            loss += loss_fn(ypreds, y)
            acc += accuracy_fn(y_true=y, y_pred=ypreds.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_acc": str(round(acc, 2)) + "%",
    }


def save_model(
    model: torch.nn.Module,
    name: str,
    direcorty: str = "models",
    extenstion: str = "pth",
):
    """_summary_
         A function that saves model parameter in `pt` or `pth`
    Args:
        `model` (torch.nn.Module): Model class
        `name` (str) : Name of the file/model
        `direcorty` (str, optional) : Directory that should have the model pth or pt file. Defaults to 'models'.
        `extenstion` (str, optional) : extension of the model saved file can be either `pt` or `pth`. Defaults to 'pth'.
    """
    from pathlib import Path
    from torch import save

    # * Create direcoty path
    path = Path(direcorty)
    path.mkdir(parents=True, exist_ok=True)

    # * create save path
    name = name + "." + extenstion
    save_path = path / name
    print(f"saving model to {save_path}")
    save(obj=model.state_dict(), f=save_path)
