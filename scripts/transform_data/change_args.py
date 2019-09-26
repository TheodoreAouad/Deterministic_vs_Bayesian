from src.utils import get_file_and_dir_path_in_dir, load_from_file, save_to_file

path_to_exps = 'polyaxon_results/groups/259'

files, _ = get_file_and_dir_path_in_dir(path_to_exps, 'arguments.pkl')

for file in files:
    args = load_from_file(file)
    if 'split_labels' in args.keys() and 'type_of_unseen' in args.keys():
        if args['type_of_unseen'] != 'unseen_classes':
            args['split_labels'] = 10
            save_to_file(args, file)
