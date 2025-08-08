import pandas as pd  # importing Pandas module


class dataset_Analyse:

    def __init__(self):
        self.data = None  # set data as None
        self.cleaned = None  # set Final Cleaned Data as None

    def Dataset(self):
        try:
            data = pd.read_csv("Dataset/sentences_full_language.csv")
            sampled = data.sample(n=1500000, random_state=42)  # random choice 1 m Data from everywhere
            self.data = sampled.reset_index(drop=True)

            print("Successfully Read Dataset")  # loaded Dataset
            print("Columns:", self.data.columns.tolist())  # show names of DataFrame
            # اگر ستونی به نام 'label' داری و میخوای شمارش انجام بدی:
            if 'label' in self.data.columns:
                print("Labels:", self.data['label'].unique())  # show all label values
                print("Label Count:", self.data['label'].nunique())  # show Length of the label values
                print("Label Value Counts:\n", self.data['label'].value_counts())  # show counts of label per value
            print("Unique Texts:", self.data['text'].nunique())  # show number of unique text

            dup = self.data['text'].value_counts()  # show count of text
            print("Repeated Texts (>3 times):\n", dup[dup > 3])  # Show every text repeated more than 3 times

            return self.data  # return data

        except Exception as e:
            print(f"Error while loading dataset: {e}")
            return None  # Return None except

    def Cleaned_DS(self):
        if self.data is None:
            print("Data not loaded. Loading now...")
            self.Dataset()  # Load dataset from upper function

        try:
            print("Cleaning Dataset...")  # starting to clean data
            value_cnts = self.data["text"].value_counts()  # get all values of text
            repeated = value_cnts[value_cnts > 3].index  # texts repeated more than 3 times
            self.cleaned = self.data[~self.data["text"].isin(repeated)]  # remove repeated texts
            print("Cleaning Done. New Shape:", self.cleaned.shape)  # show cleaned data shape
            return self.cleaned  # return cleaned data

        except Exception as e:
            print(f"Failed to Clean Dataset: {e}")
            return None  # Return None except


# ** usage example **

# from dataset import dataset_Analyse

# if __name__ == "__main__":
    # analyzer = dataset_Analyse()
    # main_data = analyzer.Dataset()
    # cleaned = analyzer.Cleaned_DS()
    # print(cleaned.head())

