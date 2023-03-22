import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from auxiliary.settings import DEVICE, USE_CONFIDENCE_WEIGHTED_POOLING, make_deterministic
from auxiliary.utils import print_metrics, log_metrics
from classes.core.Evaluator import Evaluator
from classes.core.LossTracker import LossTracker
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelFC4 import ModelFC4

# --------------------------------------------------------------------------------------------------------------------

RANDOM_SEED = 0
EPOCHS = 2000
BATCH_SIZE = 1
LEARNING_RATE = 0.0003
FOLD_NUM = 0

# The subset of test images to be monitored (set to empty list to skip saving visualizations and speed up training)
# For example: TEST_VIS_IMG = ["IMG_0753", "IMG_0438", "IMG_0397"]
TEST_VIS_IMG = []

RELOAD_CHECKPOINT = False
PATH_TO_PTH_CHECKPOINT = os.path.join("../trained_models", "fold_{}".format(FOLD_NUM))


# --------------------------------------------------------------------------------------------------------------------

def main(opt):
    fold_num, epochs, batch_size, lr = opt.fold_num, opt.epochs, opt.batch_size, opt.lr

    path_to_log = os.path.join("train", "logs", "fold_{}_{}".format(str(fold_num), str(time.time())))
    os.makedirs(path_to_log, exist_ok=True)
    path_to_metrics_log = os.path.join(path_to_log, "metrics.csv")

    model = ModelFC4()

    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(PATH_TO_PTH_CHECKPOINT))
        model.load(PATH_TO_PTH_CHECKPOINT)

    model.print_network()
    model.log_network(path_to_log)
    model.set_optimizer(lr)

    training_set = ColorCheckerDataset(train=True, folds_num=fold_num)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)
    print("\n Training set size ... : {}".format(len(training_set)))

    test_set = ColorCheckerDataset(train=False, folds_num=fold_num)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=True)
    print(" Test set size ....... : {}\n".format(len(test_set)))

    path_to_vis = os.path.join(path_to_log, "test_vis")
    if TEST_VIS_IMG:
        print("Test vis for monitored image {} will be saved at {}\n".format(TEST_VIS_IMG, path_to_vis))
        os.makedirs(path_to_vis)

    print("\n**************************************************************")
    print("\t\t\t Training FC4 - Fold {}".format(fold_num))
    print("**************************************************************\n")

    evaluator = Evaluator()
    best_val_loss, best_metrics = 100.0, evaluator.get_best_metrics()
    train_loss, val_loss = LossTracker(), LossTracker()

    for epoch in range(epochs):

        model.train_mode()
        train_loss.reset()
        start = time.time()

        for i, (img, label, _) in enumerate(training_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            loss = model.optimize(img, label)
            train_loss.update(loss)

            if i % 5 == 0:
                print("[ Epoch: {}/{} - Batch: {} ] | [ Train loss: {:.4f} ]".format(epoch, epochs, i, loss))

        train_time = time.time() - start

        val_loss.reset()
        start = time.time()

        if epoch % 5 == 0:
            evaluator.reset_errors()
            model.evaluation_mode()

            print("\n--------------------------------------------------------------")
            print("\t\t\t Validation")
            print("--------------------------------------------------------------\n")

            with torch.no_grad():
                for i, (img, label, file_name) in enumerate(test_loader):
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    pred, rgb, confidence = model.predict(img, return_steps=True)
                    loss = model.get_loss(pred, label).item()
                    val_loss.update(loss)
                    evaluator.add_error(model.get_loss(pred, label).item())

                    if i % 5 == 0:
                        print("[ Epoch: {}/{} - Batch: {}] | Val loss: {:.4f} ]".format(epoch, epochs, i, loss))

                    img_id = file_name[0].split(".")[0]
                    if USE_CONFIDENCE_WEIGHTED_POOLING:
                        if img_id in TEST_VIS_IMG:
                            model.save_vis({"img": img, "label": label, "pred": pred, "rgb": rgb, "c": confidence},
                                           os.path.join(path_to_vis, img_id, "epoch_{}.png".format(epoch)))

            print("\n--------------------------------------------------------------\n")

        val_time = time.time() - start

        metrics = evaluator.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(train_loss.avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ..... : {:.4f}".format(val_time))
            print(" Val Loss ..... : {:.4f}".format(val_loss.avg))
            print("....................................................................")
            print_metrics(metrics, best_metrics)
        print("********************************************************************\n")

        if 0 < val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            best_metrics = evaluator.update_best_metrics()
            print("Saving new best model... \n")
            model.save(path_to_log)

        log_metrics(train_loss.avg, val_loss.avg, metrics, best_metrics, path_to_metrics_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_num", type=int, default=FOLD_NUM)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    opt = parser.parse_args()
    make_deterministic(opt.random_seed)

    print("\n *** Training configuration ***")
    print("\t Fold num ........ : {}".format(opt.fold_num))
    print("\t Epochs .......... : {}".format(opt.epochs))
    print("\t Batch size ...... : {}".format(opt.batch_size))
    print("\t Learning rate ... : {}".format(opt.lr))
    print("\t Random seed ..... : {}".format(opt.random_seed))

    main(opt)
