from src.utils import get_file_and_dir_path_in_dir, load_from_file, save_to_file

path_to_exps = 'output/determinist_cifar10'

files, _ = get_file_and_dir_path_in_dir(path_to_exps, 'arguments.pkl')

for file in files:
    args = load_from_file(file)
    args['number_of_tests'] = 1
    print(file, 'changed')
    save_to_file(args, file)
