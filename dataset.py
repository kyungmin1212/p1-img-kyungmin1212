import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.transforms import *
from albumentations import *
from albumentations.pytorch import ToTensorV2

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class MyAugmentation:
    def __init__(self,resize,mean,std,**args):
        self.transform = Compose([
            CenterCrop(320,256),
            Resize(resize[0], resize[1], p=1.0),
            HorizontalFlip(p=0.5),
            # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            # GaussNoise(p=0.5),
            # ShiftScaleRotate(rotate_limit=(-15,15),p=0.5),
            # GridDropout(p=0.2),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    
    def __call__(self, image):
        return self.transform(image=np.array(image))['image']

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskBaseDataset(data.Dataset):
    num_classes = 3 * 2 * 3 # 정의를 해줘야사용가능

    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2

    class GenderLabels:
        male = 0
        female = 1

    class AgeGroup:
        map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 58 else 2

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "incorrect_mask1": MaskLabels.incorrect,
        "incorrect_mask2": MaskLabels.incorrect,
        "normal": MaskLabels.normal,
        "normal1": MaskLabels.normal,
        "normal2": MaskLabels.normal
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = getattr(self.GenderLabels, gender)
                age_label = self.AgeGroup.map_label(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index):
        return self.mask_labels[index]

    def get_gender_label(self, index):
        return self.gender_labels[index]

    def get_age_label(self, index):
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        # data_dir : './data/train/images'
        profiles = os.listdir(self.data_dir)
        # profiles : ['000001_female_Asian_45','000002_female_Asian_52',...]
        profiles = [profile for profile in profiles if not profile.startswith(".")] # "." 로 시작하는 폴더는 무시
        # split_profiles = {"train":set(전체 인덱스집합에서 val인덱스집합를 뺀 인덱스집합) , "val":set(val인덱스집합)}
        split_profiles = self._split_profile(profiles, self.val_ratio)

        # cnt 는 split_profile 을 통해 나누어진 index를 나타내기 위함이 아니라
        # split_profile 을 통해 나누어진 index를 이용하여 image_paths , mask_labels , gender_labels ,age_labels 에 들어간 데이터의 index를 나타내기 위함
        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(self.GenderLabels, gender)
                    age_label = self.AgeGroup.map_label(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self):
        return [torch.utils.data.Subset(self, indices) for phase, indices in self.indices.items()]

class MaskCheckDataset(data.Dataset):

    num_classes = 3 # 정의를 해줘야사용가능

    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2

    # class GenderLabels:
    #     male = 0
    #     female = 1

    # class AgeGroup:
    #     map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2

    image_paths = []
    mask_labels = []
    # gender_labels = []
    # age_labels = []

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "incorrect_mask1": MaskLabels.incorrect,
        "incorrect_mask2": MaskLabels.incorrect,
        "normal": MaskLabels.normal,
        "normal1": MaskLabels.normal,
        "normal2": MaskLabels.normal
    }

    def __init__(self,data_dir,mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()
    
    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }
    # _split_profile 은 이미지폴더들을 train 과 valide에 랜덤 indice를 만들어준다 index를 만들어준다고 생각 
    

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")] # profiles 는 images 안에 들어있는 이미지폴더들을 말한다. 000020_female_Asian_50이런 이미지 폴더들이 모여져있는것이 profiles이다.
        split_profiles = self._split_profile(profiles, self.val_ratio)
        # split_profiles 는 사람별로 묶어져있다. {train:train으로지정된이미지폴더index번호,val:val으로지정된이미지폴더index번호}
        cnt = 0
        for phase, indices in split_profiles.items(): #phase : train, val 를 말한다.
            for _idx in indices: # 각각의 인덱스 번호를 불러와서 profiles에 있는 한사람폴더를가져온다.
                profile = profiles[_idx] # profile :000020_female_Asian_50 이런 한사람폴더를 말한다.
                img_folder = os.path.join(self.data_dir, profile) # img_folder 절대경로 생성
                for file_name in os.listdir(img_folder): # img_folder안에 있는 사진들꺼낸다. incorrect_mask.jpg
                    _file_name, ext = os.path.splitext(file_name) #os.path.splitext는 확장자만 떨어뜨린다. ('incorrect_mask','.jpg')이런식으로
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # 이미지한장 절대경로 생성
                    mask_label = self._file_names[_file_name]

                    # id, gender, race, age = profile.split("_")
                    # gender_label = getattr(self.GenderLabels, gender)
                    # age_label = self.AgeGroup.map_label(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    # self.gender_labels.append(gender_label)
                    # self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1
                    # indices 딕셔너리에 {'Train':[0,1,2,...,train개수-1],'val':[train개수,train개수+1,train개수+2,...,전체이미지개수-1]}를 만들어주기위함

    # 이미 위에서 _split_profile로 섞인 이미지를 각각의 번호를 메겨서 그거에 해당하는거를 뽑아오기위함. 
    # train 이든 val 이든 상관없이 이미 _split으로 섞어놨으니까 그거를 그냥 순서대로 train,val 순으로 image_paths 나 labels 에 넣어주고 인덱스롤 불러와서뽑으면된다.
    def split_dataset(self):
        return [Subset(self, indices) for phase, indices in self.indices.items()]
    
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        # gender_label = self.get_gender_label(index)
        # age_label = self.get_age_label(index)
        # multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, mask_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index):
        return self.mask_labels[index]

    # def get_gender_label(self, index):
    #     return self.gender_labels[index]

    # def get_age_label(self, index):
    #     return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    # 일종의 방어 코드 mean과 std 가 주어지지 않았을때 계산하라는것이다.
    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

class GenderCheckDataset(data.Dataset):

    num_classes = 2 # 정의를 해줘야사용가능

    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2

    class GenderLabels:
        male = 0
        female = 1

    # class AgeGroup:
    #     map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2

    image_paths = []
    # mask_labels = []
    gender_labels = []
    # age_labels = []

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "incorrect_mask1": MaskLabels.incorrect,
        "incorrect_mask2": MaskLabels.incorrect,
        "normal": MaskLabels.normal,
        "normal1": MaskLabels.normal,
        "normal2": MaskLabels.normal
    }

    def __init__(self,data_dir,mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()
    
    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }
    # _split_profile 은 이미지폴더들을 train 과 valide에 랜덤 indice를 만들어준다 index를 만들어준다고 생각 
    

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")] # profiles 는 images 안에 들어있는 이미지폴더들을 말한다. 000020_female_Asian_50이런 이미지 폴더들이 모여져있는것이 profiles이다.
        split_profiles = self._split_profile(profiles, self.val_ratio)
        # split_profiles 는 사람별로 묶어져있다. {train:train으로지정된이미지폴더index번호,val:val으로지정된이미지폴더index번호}
        cnt = 0
        for phase, indices in split_profiles.items(): #phase : train, val 를 말한다.
            for _idx in indices: # 각각의 인덱스 번호를 불러와서 profiles에 있는 한사람폴더를가져온다.
                profile = profiles[_idx] # profile :000020_female_Asian_50 이런 한사람폴더를 말한다.
                img_folder = os.path.join(self.data_dir, profile) # img_folder 절대경로 생성
                for file_name in os.listdir(img_folder): # img_folder안에 있는 사진들꺼낸다. incorrect_mask.jpg
                    _file_name, ext = os.path.splitext(file_name) #os.path.splitext는 확장자만 떨어뜨린다. ('incorrect_mask','.jpg')이런식으로
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # 이미지한장 절대경로 생성
                    # mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(self.GenderLabels, gender)
                    # age_label = self.AgeGroup.map_label(age)

                    self.image_paths.append(img_path)
                    # self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    # self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1
                    # indices 딕셔너리에 {'Train':[0,1,2,...,train개수-1],'val':[train개수,train개수+1,train개수+2,...,전체이미지개수-1]}를 만들어주기위함

    # 이미 위에서 _split_profile로 섞인 이미지를 각각의 번호를 메겨서 그거에 해당하는거를 뽑아오기위함. 
    # train 이든 val 이든 상관없이 이미 _split으로 섞어놨으니까 그거를 그냥 순서대로 train,val 순으로 image_paths 나 labels 에 넣어주고 인덱스롤 불러와서뽑으면된다.
    def split_dataset(self):
        return [Subset(self, indices) for phase, indices in self.indices.items()]
    
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        # mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        # age_label = self.get_age_label(index)
        # multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, gender_label

    def __len__(self):
        return len(self.image_paths)

    # def get_mask_label(self, index):
    #     return self.mask_labels[index]

    def get_gender_label(self, index):
        return self.gender_labels[index]

    # def get_age_label(self, index):
    #     return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    # 일종의 방어 코드 mean과 std 가 주어지지 않았을때 계산하라는것이다.
    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

class AgeCheckDataset(data.Dataset):

    num_classes = 3 # 정의를 해줘야사용가능

    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2

    # class GenderLabels:
    #     male = 0
    #     female = 1

    class AgeGroup:
        map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 58 else 2

    image_paths = []
    # mask_labels = []
    # gender_labels = []
    age_labels = []

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "incorrect_mask1": MaskLabels.incorrect,
        "incorrect_mask2": MaskLabels.incorrect,
        "normal": MaskLabels.normal,
        "normal1": MaskLabels.normal,
        "normal2": MaskLabels.normal
    }

    def __init__(self,data_dir,mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()
    
    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }
    # _split_profile 은 이미지폴더들을 train 과 valide에 랜덤 indice를 만들어준다 index를 만들어준다고 생각 
    

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")] # profiles 는 images 안에 들어있는 이미지폴더들을 말한다. 000020_female_Asian_50이런 이미지 폴더들이 모여져있는것이 profiles이다.
        split_profiles = self._split_profile(profiles, self.val_ratio)
        # split_profiles 는 사람별로 묶어져있다. {train:train으로지정된이미지폴더index번호,val:val으로지정된이미지폴더index번호}
        cnt = 0
        for phase, indices in split_profiles.items(): #phase : train, val 를 말한다.
            for _idx in indices: # 각각의 인덱스 번호를 불러와서 profiles에 있는 한사람폴더를가져온다.
                profile = profiles[_idx] # profile :000020_female_Asian_50 이런 한사람폴더를 말한다.
                img_folder = os.path.join(self.data_dir, profile) # img_folder 절대경로 생성
                for file_name in os.listdir(img_folder): # img_folder안에 있는 사진들꺼낸다. incorrect_mask.jpg
                    _file_name, ext = os.path.splitext(file_name) #os.path.splitext는 확장자만 떨어뜨린다. ('incorrect_mask','.jpg')이런식으로
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # 이미지한장 절대경로 생성
                    # mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    # gender_label = getattr(self.GenderLabels, gender)
                    age_label = self.AgeGroup.map_label(age)

                    self.image_paths.append(img_path)
                    # self.mask_labels.append(mask_label)
                    # self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1
                    # indices 딕셔너리에 {'Train':[0,1,2,...,train개수-1],'val':[train개수,train개수+1,train개수+2,...,전체이미지개수-1]}를 만들어주기위함

    # 이미 위에서 _split_profile로 섞인 이미지를 각각의 번호를 메겨서 그거에 해당하는거를 뽑아오기위함. 
    # train 이든 val 이든 상관없이 이미 _split으로 섞어놨으니까 그거를 그냥 순서대로 train,val 순으로 image_paths 나 labels 에 넣어주고 인덱스롤 불러와서뽑으면된다.
    def split_dataset(self):
        return [Subset(self, indices) for phase, indices in self.indices.items()]
    
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        # mask_label = self.get_mask_label(index)
        # gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        # multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, age_label

    def __len__(self):
        return len(self.image_paths)

    # def get_mask_label(self, index):
    #     return self.mask_labels[index]

    # def get_gender_label(self, index):
    #     return self.gender_labels[index]

    def get_age_label(self, index):
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    # 일종의 방어 코드 mean과 std 가 주어지지 않았을때 계산하라는것이다.
    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255


class TestDataset(data.Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
            CenterCrop(320,256),
            Resize(resize[0], resize[1]),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image

    def __len__(self):
        return len(self.img_paths)
