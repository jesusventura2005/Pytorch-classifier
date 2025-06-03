import torch
import torch.nn as nn
from torchvision import datasets, transforms
from network import Network
import matplotlib.pyplot as plt
from helper import view_classify

def train_model(model, trainloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f}')

def evaluate_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Precisión en el conjunto de prueba: {accuracy:.2f}%')
    return accuracy

def make_prediction(model, testloader):
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    img = images[0]
    
    with torch.no_grad():
        logps = model(img.unsqueeze(0))
    ps = torch.exp(logps)
    
    view_classify(img.view(1, 28, 28), ps, version='Fashion')
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    print(f"\nEtiqueta real: {class_names[labels[0]]}")

def create_model():
    # Parámetros configurables por el usuario
    num_hidden_layers = int(input("Ingrese el número de capas ocultas (default=2): ") or "2")

    # Crear lista de tamaños de capas ocultas
    hidden_size = []
    for i in range(num_hidden_layers):
        nodes = int(input(f"Ingrese el número de filtros para la capa convolucional {i+1} (default=32): ") or "32")
        hidden_size.append(nodes)

    print(f"\nArquitectura de la red:")
    print(f"Entrada: Imagen 28x28 = 784 neuronas ")
    for i, size in enumerate(hidden_size):
        print(f"Capa convolucional {i+1}: {size} neuronas")
    print(f"Salida: 10 capas\n")

    # Crear la instancia de la red
    model = Network(784, hidden_size, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    return model, optimizer

def main():
    # Configuración de matplotlib
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['figure.dpi'] = 100

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    
    model, optimizer = create_model()
    criterion = nn.NLLLoss()
    print("\nIniciando entrenamiento inicial...")
    train_model(model, trainloader, criterion, optimizer, epochs=10)
    evaluate_model(model, testloader)
    
    while True:
        print("\n¿Qué desea hacer?")
        print("1. Hacer una nueva predicción")
        print("2. Entrenar la red por más épocas")
        print("3. Evaluar el modelo")
        print("4. Crear un nuevo modelo")
        print("5. Salir")
        
        opcion = input("\nSeleccione una opción (1-5): ")
        
        if opcion == "1":
            make_prediction(model, testloader)
        elif opcion == "2":
            epochs = int(input("¿Cuántas épocas adicionales desea entrenar? "))
            train_model(model, trainloader, criterion, optimizer, epochs)
        elif opcion == "3":
            evaluate_model(model, testloader)
        elif opcion == "4":
            model, optimizer = create_model()
            print("\nIniciando entrenamiento del nuevo modelo...")
            train_model(model, trainloader, criterion, optimizer, epochs=10)
            evaluate_model(model, testloader)
        elif opcion == "5":
            print("¡Hasta luego!")
            break
        else:
            print("Opción no válida. Por favor, seleccione una opción del 1 al 5.")

if __name__ == "__main__":
    main() 