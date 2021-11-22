import os

ALLOWED_EXTENSIONS = set(['.png', '.jpg', '.JPG', '.PNG',
                         '.jpeg', '.JPEG', '.gif', '.GIF', '.bmp', '.BMP'])

def write_dataset_list_without_label(dir_path, list_path):
    objs = os.listdir(dir_path)
    files = []
    for obj in objs:
        if not os.path.isdir(obj):
            ext = os.path.splitext(obj)[-1]
            if ext in ALLOWED_EXTENSIONS:
                #files.append(dir_path + "/" + obj)
                files.append(obj)
            else:
                print("{} is not in allowed_extensions".format(obj))
        else:
            print("{} is a dir".format(obj))
    print(len(files))
    with open(list_path, "w") as f:
        for file in files:
            f.writelines([file, "\n"])

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--config_file', type=str,
    #                    default='./configs/create_dataset_config.yaml')
    #args = parser.parse_args()
    #with open(args.config_file) as f:
    #    config = yaml.load(f)
    # process_file(argparse.Namespace(**config))
    dir_path = '/home/kezhiying/document_forgery/SRNet_pytorch/content/srnet_data/i_s'
    list_path = '/home/kezhiying/document_forgery/SRNet_pytorch/content/total_train.txt'
    write_dataset_list_without_label(dir_path, list_path)