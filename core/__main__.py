import argparse
from . import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='config_yaml/config.yaml')
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    args = parser.parse_args()

    stages = utils.eval_config(config=utils.load_yaml(args.config))

    # training
    if 'trainer' in stages:
        trainer = stages['trainer']
        trainer(
            num_epochs=args.num_epochs,
            num_gpus=args.num_gpus,
            resume_path=args.resume_path,
            checkpoint_path=args.checkpoint_path,
        )
    # evaluation
    elif 'evaluator' in stages:
        evaluator = stages['evaluator']
        evaluator(
            checkpoint_path=args.checkpoint_path,
            num_gpus=args.num_gpus,
        )
    else:
        print('No Run !.')
