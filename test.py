from scoring_model.dataset.loading import RawDataLoader

if __name__ == "__main__":
    loader = RawDataLoader()
    files = loader.load("bank.csv")
    print(files)