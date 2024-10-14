import os
import sys
import shutil

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def progress_bar(iteration, total, prefix='', suffix="", length=40, fill='█'):
    percent = '{0:.1f}'.format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% Completed | {suffix}')
    sys.stdout.flush()


class F4KData:
    species_cluster_id_map = {
        1: "Dascyllus reticulatus",
        2: "Plectroglyphidodon dickii",
        3: "Chromis chrysura",
        4: "Amphiprion clarkii",
        5: "Chaetodon lunulatus",
        6: "Chaetodon trifascialis",
        7: "Myripristis kuntee",
        8: "Acanthurus nigrofuscus",
        9: "Hemigymnus fasciatus",
        10: "Neoniphon sammara",
        11: "Abudefduf vaigiensis",
        12: "Canthigaster valentini",
        13: "Pomacentrus moluccensis",
        14: "Zebrasoma scopas",
        15: "Hemigymnus melapterus",
        16: "Lutjanus fulvus",
        17: "Scolopsis bilineata",
        18: "Scaridae",
        19: "Pempheris vanicolensis",
        20: "Zanclus cornutus",
        21: "Neoglyphidodon nigroris",
        22: "Balistapus undulatus",
        23: "Siganus fuscescens"
    }

    def __init__(self, dataset_path, ):
        self.clusters = []
        self.clusters_csv = pd.DataFrame()

        self.dataset_path = dataset_path
        self.fish_image_dir = os.path.join(self.dataset_path, 'fish_image')
        self.preprocessed_image_dir = os.path.join(self.dataset_path, 'preprocessed_image')
        self.mask_image_dir = os.path.join(self.dataset_path, 'mask_image')
        self.train_image_dir = os.path.join(self.dataset_path, 'train')
        self.validation_image_dir = os.path.join(self.dataset_path, 'validation')
        self.test_image_dir = os.path.join(self.dataset_path, 'test')

        self.load_data()
        self.image_cnt = len(self.clusters)
        if not os.path.exists(self.preprocessed_image_dir) or len(os.listdir(self.preprocessed_image_dir)) == 0:
            i = 0
            for image in self.clusters:
                self.enhance_image(image["fish_image"], image["preprocessed_image"])
                progress_bar(i + 1, self.image_cnt, prefix=f"Enhancing images({i+1}/{self.image_cnt})", suffix=f"{image['preprocessed_image']}")
                i += 1
        if not os.path.exists(self.train_image_dir) or len(os.listdir(self.train_image_dir)) == 0:
            self.train_test_split_dir()
            self.clusters_csv = pd.read_csv(os.path.join(self.dataset_path, 'clusters.csv'))

    def load_data(self):
        cluster_range = [(int(x.split("_")[1]), x) for x in os.listdir(self.fish_image_dir) if "fish" in x]
        for cluster_id, cluster_dir in cluster_range:
            fish_images = os.listdir(os.path.join(self.fish_image_dir, cluster_dir))
            for image in fish_images:
                self.clusters.append({
                    "id": cluster_id,
                    "species": self.species_cluster_id_map[cluster_id],
                    "fish_image": os.path.join(self.fish_image_dir, cluster_dir, image),
                    "preprocessed_image": os.path.join(self.preprocessed_image_dir, cluster_dir, image),
                    "mask_image": os.path.join(self.mask_image_dir,
                                                cluster_dir.replace("fish", "mask"),
                                                image.replace("fish", "mask")),
                    "file_name": image
                })
        self.clusters_csv = pd.DataFrame(self.clusters).sort_values(by=["species"])
        self.clusters_csv.set_index("id", inplace=True)
        self.clusters_csv.to_csv(os.path.join(self.dataset_path, 'clusters.csv'))

    @staticmethod
    def enhance_image(image_path, output_path):
        # Read the image
        image = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to improve contrast
        equalized = cv2.equalizeHist(gray)

        # Use Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

        # Apply sharpening filter
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(blurred, -1, kernel)
        output_path = image_path.replace("fish_image", "preprocessed_image")
        if not os.path.exists("/".join(output_path.split("/")[:-1])):
            os.makedirs("/".join(output_path.split("/")[:-1]))
        cv2.imwrite(output_path, sharpened)
        # return sharpened

    def train_test_split_dir(self, training_size=0.8):
        train, test = train_test_split(self.clusters_csv, train_size=training_size, random_state=42)
        train, validation = train_test_split(train, train_size=training_size, random_state=42)
        train = list(train["preprocessed_image"])
        validation = list(validation["preprocessed_image"])
        test = list(test["preprocessed_image"])
        print(f"Training images: {len(train)}\nValidation images: {len(validation)}\nTest images: {len(test)}")
        i = 0
        for image in self.clusters:
            if image["preprocessed_image"] in train:
                if not os.path.exists(os.path.join(self.train_image_dir, str(image["id"]))):
                    os.makedirs(os.path.join(self.train_image_dir, str(image["id"])))
                shutil.copy(image["preprocessed_image"], os.path.join(self.train_image_dir, str(image["id"])))
                image["dataset"] = "train"
            elif image["preprocessed_image"] in validation:
                if not os.path.exists(os.path.join(self.validation_image_dir, str(image["id"]))):
                    os.makedirs(os.path.join(self.validation_image_dir, str(image["id"])))
                shutil.copy(image["preprocessed_image"], os.path.join(self.validation_image_dir, str(image["id"])))
                image["dataset"] = "validaiton"
            else:
                if not os.path.exists(os.path.join(self.test_image_dir, str(image["id"]))):
                    os.makedirs(os.path.join(self.test_image_dir, str(image["id"])))
                shutil.copy(image["preprocessed_image"], os.path.join(self.test_image_dir, str(image["id"])))
                image["dataset"] = "test"
            progress_bar(i + 1, self.image_cnt, prefix=f"Copying images({i+1}/{self.image_cnt})", suffix=f"{image['preprocessed_image']}")
            i += 1
        self.clusters_csv = pd.DataFrame(self.clusters).sort_values(by=["species"])
        self.clusters_csv.set_index("id", inplace=True)
        self.clusters_csv.to_csv(os.path.join(self.dataset_path, 'clusters.csv'))


if __name__ == '__main__':
    f4k = F4KData(dataset_path="./fishRecognition_GT")
