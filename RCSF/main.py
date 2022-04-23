import argparse
from trainer import Trainer
import random
import numpy as np
import torch
import logging
import os
import sys
from logging import StreamHandler, FileHandler
# torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def init_logger(log_filename):
    handler1 = StreamHandler(stream=sys.stdout)
    handler2 = FileHandler(filename=log_filename, mode='a', delay=False)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[handler1, handler2]
    )
    # stream_handler = logging.StreamHandler()
    # stream_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    # stream_handler.setFormatter(stream_format)
    # logger.addHandler(stream_handler)


def main(args):
    init_logger(args.log_filename)
    set_seed(args)
    logger.info('**********************Job Start**********************')
    logger.info(f'Learning rate = {args.lr}')
    logger.info(f'Warmup rate = {args.warmup_rate}')
    logger.info(f'Batch_size = {args.batch_size}')
    logger.info(f'Gradient_accumulation_steps = {args.gradient_accumulation_steps}')
    logger.info(f'Train epoch = {args.num_epochs}')
    logger.info(f'Pretrained_model name = {args.pretrained_model}')
    if args.do_train:
        trainer = Trainer(args)
        trainer.train()
    if args.do_test:
        checkpoint = torch.load(os.path.join(args.model_dir, args.target_domain, str(args.n_samples), 'model.pth'))
        # if args.num_gpus == 2:
        #     trainer.model = nn.DataParallel(trainer.model.to('cuda'), device_ids=[0])
        #     trainer.model.module.load_state_dict(checkpoint['model_state_dict'])
        #     trainer.model = trainer.model.module
        # else:
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        # trainer = load_model(args.model_dir, args.target_domain, args.n_samples)
        logger.info('Starting test...')
        with torch.no_grad():
            trainer.evaluate('test')
        # trainer = Trainer.load_model(args.model_dir, args.target_domain, args.n_samples)
        # logger.info('Starting test...')
        # trainer.evaluate('test')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--early_stop', default=10, type=int)
    parser.add_argument('--pretrained_model', default='deepset/bert-large-uncased-whole-word-masking-squad2', type=str, help='whcih pretrained model to use')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_samples', default=0, type=int)
    parser.add_argument('--target_domain', default='AddToPlaylist', type=str)
    parser.add_argument('--max_len', default=128, type=int)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    # parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--warmup_rate", default=0.0, type=float, help="Linear warmup rate")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--num_epochs', default=2, type=int)
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader number of workers')
    parser.add_argument('--num_gpus', default=1, type=int, help='how many gpus used to train')
    parser.add_argument("--final_div_factor", type=float, default=1e4, help="final div factor of linear decay scheduler")
    parser.add_argument("--n_top", type=int, default=5)
    parser.add_argument('--model_dir', type=str, default='model_dir')
    parser.add_argument('--log_dir', type=str, default='log_dir')
    args = parser.parse_args()
    args.log_filename = os.path.join(args.log_dir, args.target_domain + '_' + str(args.n_samples) + '.log')
    main(args)