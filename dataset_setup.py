from torchvision import datasets, transforms

def load_cifar10():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),      # resize for ResNet
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    train_data = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    print(f"✅ Download complete!")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples:  {len(test_data)}")

    return train_data, test_data


if __name__ == "__main__":
    load_cifar10()

