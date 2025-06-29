from dataset import load_dataset
from train import train_model

def main():
    train_data,test_data = load_dataset()
    model = train_model(train_data, test_data)
    print("Training complete")

if __name__ == '__main__':
    main()

