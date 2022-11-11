import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('-num_aug')
parser.add_argument('-save_dir')

args = parser.parse_args() 

print(args.num_aug,args.save_dir)