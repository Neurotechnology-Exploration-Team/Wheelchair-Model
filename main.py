from EEGDataLoader import get_dataLoader
import EEGModel
def main():
    dataLoader = get_dataLoader("collected_data.csv")
    model = EEGModel.createModel()
    EEGModel.trainModel(model,dataLoader)
    print(model)

if __name__ == "__main__":
    main()
