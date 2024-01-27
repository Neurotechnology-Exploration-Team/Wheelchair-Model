from EEGDataLoader import get_dataLoader, get_shape
import EEGModel
def main():
    dataLoader = get_dataLoader("collected_data.csv")
    #shape = get_shape(data_loader=dataLoader)
    #print(shape)
    model = EEGModel.createModel()
    EEGModel.trainModel(model,dataLoader)
    print(model)

if __name__ == "__main__":
    main()
