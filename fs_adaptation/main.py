import torch
from fs_adaptation.data import MnistData_arithmetic
from tqdm import tqdm
import pickle
import argparse
import numpy as np
from model.smn import SMN
import os
import copy
from utils import pprint, str2bool, test_model_iid, fs_finetune
import random

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str2bool, default=True, help='use gpu or not')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--model', type=str, default='SMN', help='CNN | SMN | Transformer')
parser.add_argument('--encoder', type=str, default='twoconv')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--loadsaved', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='debug')
parser.add_argument('--train', type=str2bool, default=True)
parser.add_argument('--finetune_num', type=int, default=15)
parser.add_argument('--finetune_iter', type=int, default=100)
parser.add_argument('--finetune_lr', type=float, default=0.01)

# for SMN
parser.add_argument('--routing_iter', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--add', type=int, default=0, help='number of added modules for finetuning')
parser.add_argument('--modulated', type=str2bool, default=True)
parser.add_argument('--im_coef', type=float, default=0.001)

# for transformer
parser.add_argument("--img_size", type=int, default=28, help="Img size")
parser.add_argument("--patch_size", type=int, default=14, help="Patch Size")
parser.add_argument("--embed_dim", type=int, default=84, help="dimensionality of the latent space")
parser.add_argument("--n_attention_heads", type=int, default=4, help="number of heads to be used")
parser.add_argument("--forward_mul", type=int, default=2, help="forward multiplier")
parser.add_argument("--n_layers", type=int, default=2, help="number of encoder layers")
parser.add_argument("--load_model", type=bool, default=False, help="Load saved model")

args = vars(parser.parse_args())
pprint(args)

if args['model'] == 'SMN':
    mode = SMN
else:
    AssertionError("not implement!")

log_dir = 'checkpoint/' + args['log_dir']
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

def log(obj, filename='log.txt'):
    print(obj)
    with open(os.path.join(log_dir, filename), 'a') as f:
        print(obj, file=f)

def train_model(model, epochs, data, run):
    best_acc = 0.0
    start_epoch, ctr = 0, 0
    if args['loadsaved'] == 1:
        with open(log_dir + '/accstats.pickle', 'rb') as f:
            acc = pickle.load(f)
        with open(log_dir + '/lossstats.pickle', 'rb') as f:
            losslist = pickle.load(f)
        start_epoch = len(acc) - 1
        best_acc = 0
        for i in acc:
            if i[0] > best_acc:
                best_acc = i[0]
        ctr = len(losslist) - 1
        saved = torch.load(log_dir + '/best_model.pt')
        model.load_state_dict(saved['net'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(start_epoch, epochs):
        epoch_loss, iter_ctr, norm, t_accuracy = 0., 0., 0, 0
        model.train()
        num = 0
        for i in tqdm(range(data.train_len())):
            iter_ctr += 1.
            inp_x, inp_y = data.train_get(i)
            inp_x = model.to_device(inp_x)
            inp_y = model.to_device(inp_y)
            output, l = model(inp_x, inp_y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            norm += model.grad_norm()
            epoch_loss += l.item()
            preds = torch.argmax(output, dim=1)

            correct = preds == inp_y.long()
            t_accuracy += correct.sum().item()

            ctr += 1
            num += inp_x.shape[0]

        if (epoch + 1) % 1 == 0:
            v_accuracy = test_model_iid(model, data)
            log('** epoch {} ** epoch_loss: {:.3f}, val_acc: {:.2f}, train_acc: {:.2f}, grad_norm: {:.2f} '.format(
                epoch + 1, epoch_loss / (iter_ctr), v_accuracy, t_accuracy / num * 100, norm / iter_ctr))
            if best_acc < v_accuracy:
                best_acc = v_accuracy

                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                    'ctr': ctr,
                    'best_acc': best_acc
                }
                with open(log_dir + '/best_model_{}.pt'.format(run), 'wb') as f:
                    torch.save(state, f)
    return best_acc

def main():
    # print model details
    model = mode(args).cuda()
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print("total parameters: %.4fK" % (total / 1e3))

    ft_iter = args['finetune_iter']
    ft_lr = args['finetune_lr']
    ft_num = args['finetune_num']
    add_num = args['add']
    runs = 10
    test_acc_all_iid, test_acc_all = [], []

    data = MnistData_arithmetic(args['batch_size'], task_idx='A2B', train_num=60000, finetune_num=ft_num)
    for run in range(runs):
        print("********** Run {} **********".format(run + 1))
        model = mode(args).cuda()

        if args['train']:
            train_model(model, args['epochs'], data, run)

        saved = torch.load(log_dir + '/best_model_{}.pt'.format(run))
        model.load_state_dict(saved['net'])

        acc_iid = test_model_iid(model, data)
        test_acc_all_iid.append(acc_iid)

        add_module = True if add_num > 0 else False
        accuracy = fs_finetune(args, model, data, ft_iter, ft_lr, num_classes=2, log=True,
                             add_module=add_module, add_num=add_num)
        test_acc_all.append(accuracy)

        print("*** Test accuracy: {} ***".format(accuracy))
    test_acc_all = np.array(test_acc_all)
    print("Few-shot test acc: {} (std: {})".format(test_acc_all.mean(), test_acc_all.std()))

    test_acc_all_iid = np.array(test_acc_all_iid)
    print("IID test acc: {} (std: {})".format(test_acc_all_iid.mean(), test_acc_all_iid.std()))

if __name__ == '__main__':
    main()