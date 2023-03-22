import os
import time

import matplotlib.pyplot as plt
import torch.utils.data
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from auxiliary.settings import DEVICE
from auxiliary.utils import correct, rescale, scale
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.fc4.ModelFC4 import ModelFC4
from classes.core.Evaluator import Evaluator

# Set to -1 to process all the samples in the test set of the current fold
NUM_SAMPLES = -1

# The number of folds to be processed (either 1, 2 or 3)
NUM_FOLDS = 1

# Where to save the generated visualizations
PATH_TO_SAVED = os.path.join("vis", "plots", "train_{}".format(time.time()))


def main():
    evaluator = Evaluator()
    model = ModelFC4()
    os.makedirs(PATH_TO_SAVED)

    for num_fold in range(NUM_FOLDS):
        test_set = ColorCheckerDataset(train=False, folds_num=num_fold)
        dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=20)

        path_to_pretrained = os.path.join("trained_models", "baseline", "fc4_cwp", "fold_{}".format(num_fold))
        model.load(path_to_pretrained)
        model.evaluation_mode()

        print("\n *** Generating visualizations for FOLD {} *** \n".format(num_fold))
        print(" * Test set size: {}".format(len(test_set)))
        print(" * Using pretrained model stored at: {} \n".format(path_to_pretrained))

        with torch.no_grad():
            for i, (img, label, file_name) in enumerate(dataloader):
                if NUM_SAMPLES > -1 and i > NUM_SAMPLES - 1:
                    break

                img, label = img.to(DEVICE), label.to(DEVICE)
                pred, rgb, confidence = model.predict(img, return_steps=True)
                loss = model.get_loss(pred, label).item()
                evaluator.add_error(loss)
                file_name = file_name[0].split(".")[0]
                print('\t - Input: {} - Batch: {} | Loss: {:f}'.format(file_name, i, loss))

                original = transforms.ToPILImage()(img.squeeze()).convert("RGB")
                gt_corrected, est_corrected = correct(original, label), correct(original, pred)

                size = original.size[::-1]

                scaled_rgb = rescale(rgb, size).squeeze(0).permute(1, 2, 0)
                scaled_confidence = rescale(confidence, size).squeeze(0).permute(1, 2, 0)

                weighted_est = scale(rgb * confidence)
                scaled_weighted_est = rescale(weighted_est, size).squeeze().permute(1, 2, 0)

                masked_original = scale(F.to_tensor(original).permute(1, 2, 0) * scaled_confidence)

                fig, axs = plt.subplots(2, 3)

                axs[0, 0].imshow(original)
                axs[0, 0].set_title("Original")
                axs[0, 0].axis("off")

                axs[0, 1].imshow(masked_original, cmap="gray")
                axs[0, 1].set_title("Confidence Mask")
                axs[0, 1].axis("off")

                axs[0, 2].imshow(est_corrected)
                axs[0, 2].set_title("Correction")
                axs[0, 2].axis("off")

                axs[1, 0].imshow(scaled_rgb)
                axs[1, 0].set_title("Per-patch Estimate")
                axs[1, 0].axis("off")

                axs[1, 1].imshow(scaled_confidence, cmap="gray")
                axs[1, 1].set_title("Confidence")
                axs[1, 1].axis("off")

                axs[1, 2].imshow(scaled_weighted_est)
                axs[1, 2].set_title("Weighted Estimate")
                axs[1, 2].axis("off")

                fig.suptitle("Image ID: {} | Error: {:.4f}".format(file_name, loss))
                fig.tight_layout(pad=0.25)

                path_to_save = os.path.join(PATH_TO_SAVED, "fold_{}".format(num_fold), file_name)
                os.makedirs(path_to_save)

                fig.savefig(os.path.join(path_to_save, "stages.png"), bbox_inches='tight', dpi=200)
                original.save(os.path.join(path_to_save, "original.png"))
                est_corrected.save(os.path.join(path_to_save, "est_corrected.png"))
                gt_corrected.save(os.path.join(path_to_save, "gt_corrected.png"))

                plt.clf()

    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {}".format(metrics["mean"]))
    print(" Median .......... : {}".format(metrics["median"]))
    print(" Trimean ......... : {}".format(metrics["trimean"]))
    print(" Best 25% ........ : {}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {} \n".format(metrics["wst5"]))


if __name__ == '__main__':
    main()
