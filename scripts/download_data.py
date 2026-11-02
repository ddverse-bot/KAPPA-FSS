import argparse
import kagglehub

DATASETS = {
    "kvasir": "fkarimovv/kvasir-seg",
    "isic2016": "mahmudulhasantasin/isic-2016-original-dataset",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    args = parser.parse_args()
    path = kagglehub.dataset_download(DATASETS[args.dataset])
    print(path)

if __name__ == "__main__":
    main()
