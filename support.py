import os
from glob import glob
import torchvision

class Work_With_Data():

    rm_var = '/home/helgenro/NSU_subjects/Introduction_to_AI/Smipsons_classification/simpsons_dataset/'
    directory = '/home/helgenro/NSU_subjects/Introduction_to_AI/Smipsons_classification/simpsons_dataset/*/'

    def __init__(self) -> None:
        full_dataset = torchvision.datasets.ImageFolder(
            root=self.rm_var, transform=None)
        self.mas = [x.replace(self.rm_var, '') for x in glob(self.directory, recursive = False)]

        print(full_dataset[0])
    def __len__(self):
        return len(self.mas)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
Work_With_Data()