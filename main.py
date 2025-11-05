import argparse

from src.pipeline import Pipeline


parser = argparse.ArgumentParser(description='Gets arguments for running pipeline')

# Defining the parser arguments
parser.add_argument('-rpr',
                    '--remove_prev_runs',
                    action='store_true',  # Default value is False
                    help='specifies whether you want to remove previous runs')
parser.add_argument('-p',
                    '--prepare',
                    action='store_true',  # Default value is False
                    help='specifies whether you want to implement data preparation')
parser.add_argument('-t',
                    '--train',
                    action='store_true',  # Default value is False
                    help='specifies whether you want to implement training')
parser.add_argument('-e',
                    '--export',
                    action='store_true',  # Default value is False
                    help='specifies whether you want to export a saved model')
args = parser.parse_args()


def main(remove_prev_runs: bool,
         prepare: bool,
         train: bool,
         export: bool):
    Pipeline('config.yaml',
             remove_prev_runs=remove_prev_runs,
             prepare=prepare,
             train=train,
             export=export,).run()


if __name__ == '__main__':
    main(args.remove_prev_runs,
         args.prepare,
         args.train,
         args.export)
