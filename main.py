import argparse
import train
import test
import eval
from datasets.dataset_dota import DOTA
from datasets.dataset_hrsc import HRSC
from models import ctrbox_net
import decoder
import os


def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors Implementation')
    parser.add_argument('--num_epoch', type=int, default=60, help='Number of epochs') # used to be 80
    parser.add_argument('--batch_size', type=int, default=2, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.0e-5, help='Initial learning rate') # used to be 1.25e-4
    parser.add_argument('--input_h', type=int, default=1508, help='Resized image height') # used to be 608
    parser.add_argument('--input_w', type=int, default=1008, help='Resized image width')  # used to be 608
    parser.add_argument('--K', type=int, default=200, help='Maximum of objects')          # used to be 500
    parser.add_argument('--conf_thresh', type=float, default=0.18, help='Confidence threshold, 0.1 for general evaluation') # for train=0.18
    parser.add_argument('--ngpus', type=int, default=2, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='weights_dota/model_20.pth', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='model_18.pth', help='Weights resumed in testing and evaluation')
    parser.add_argument('--dataset', type=str, default='dota', help='Name of dataset')
    parser.add_argument('--data_dir', type=str, default='./data_dir_roadmarking', help='Data directory')
    parser.add_argument('--phase', type=str, default='train', help='Phase choice= {train, test, eval}')
    parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')
    args  = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = {'dota': DOTA, 'hrsc': HRSC}
    num_classes = {'dota': 13, 'hrsc': 1} # dota 16, dota road marking 13
    heads = {'hm': num_classes[args.dataset],
             'wh': 10,
             'reg': 2,
             'cls_theta': 1,
             'forward': 1  # add by mixiaoxin to predict the forward direction
             }
    down_ratio = 4
    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=256)

    decoder = decoder.DecDecoder(K=args.K,
                                 conf_thresh=args.conf_thresh,
                                 num_classes=num_classes[args.dataset])
    if args.phase == 'train':
        ctrbox_obj = train.TrainModule(dataset=dataset,
                                       num_classes=num_classes,
                                       model=model,
                                       decoder=decoder,
                                       down_ratio=down_ratio)

        ctrbox_obj.train_network(args)
    elif args.phase == 'test':
        ctrbox_obj = test.TestModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.test(args, down_ratio=down_ratio)
    else:
        ctrbox_obj = eval.EvalModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.evaluation(args, down_ratio=down_ratio)