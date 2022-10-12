import argparse
import os.path

import torch
import cv2
import numpy as np

from amnet import AMNet
from config import get_amnet_config

from run_script import AMNetDLR

def main():
    parser = argparse.ArgumentParser(description='AMNet Image memorability prediction with attention')
    parser.add_argument('--gpu', default=-1, type=int, help='GPU ID. If -1 the application will run on CPU')
    parser.add_argument('--model-weights', default='', type=str, help='pkl file with the model weights')

    parser.add_argument('--cnn', default='ResNet50FC', type=str, help='Name of CNN model for features extraction [ResNet18FC, ResNet50FC, ResNet101FC, VGG16FC, ResNet50FT]')
    parser.add_argument('--att-off', action="store_true", help='Runs training/testing without the visual attention')
    parser.add_argument('--lstm-steps', default=3, type=int,
                        help='Number of LSTM steps. Default 3. To disable LSTM set to zero')

    parser.add_argument('--last-step-prediction', action="store_true",
                        help='Predicts memorability only at the last LSTM step')

    parser.add_argument('--test', action='store_true', help='Run evaluation')

    parser.add_argument('--eval-images', default='', type=str, help='Directory or a csv file with images to predict memorability for')
    parser.add_argument('--csv-out', default='', type=str, help='File where to save prediced memorabilities in csv format')
    parser.add_argument('--att-maps-out', default='', type=str, help='Directory where to store attention maps')

    # Training
    parser.add_argument('--epoch-max', default=-1, type=int,
                        help='If not specified, number of epochs will be set according to selected dataset')
    parser.add_argument('--epoch-start', default=0, type=int,
                        help='Allows to resume training from a specific epoch')

    parser.add_argument('--train-batch-size', default=-1, type=int,
                        help='If not specified a default size will be set according to selected dataset')
    parser.add_argument('--test-batch-size', default=-1, type=int,
                        help='If not specified a default size will be set according to selected dataset')

    # Dataset configuration
    parser.add_argument('--dataset', default='lamem', type=str, help='Dataset name [lamem, sun]')
    parser.add_argument('--experiment', default='', type=str, help='Experiment name. Usually no need to set' )
    parser.add_argument('--dataset-root', default='', type=str, help='Dataset root directory')
    parser.add_argument('--images-dir', default='images', type=str, help='Relative path to the test/train images')
    parser.add_argument('--splits-dir', default='splits', type=str, help='Relative path to directory with split files')
    parser.add_argument('--train-split', default='', type=str, help='Train split filename e.g. train_2')
    parser.add_argument('--val-split', default='', type=str, help='Validation split filename e.g. val_2')
    parser.add_argument('--test-split', default='', type=str, help='Test split filename e.g. test_2')

    args = parser.parse_args()
    hps = get_amnet_config(args)

    print("Configuration")
    print("----------------------------------------------------------------------")
    print(hps)

    amnet = AMNet()
    amnet.init(hps)

    model = amnet.model
    print(model)

    input_shape = [1, 3, 224, 224]

    base_filename, _ = os.path.splitext(hps.model_weights)

    trace = True
    if trace:
        traced = torch.jit.trace(model.float().eval(), torch.rand(input_shape).float())
        base_filename, _ = os.path.splitext(hps.model_weights)
        traced_filename = base_filename + '_traced' + '.pth'
        traced.save(traced_filename)
        print('Traced model was saved to:', traced_filename)
    else:
        with torch.no_grad():
            model = model.cpu().eval()
        scripted = torch.jit.script(model)
        scripted_filename = base_filename + '_scripted' + '.pth'
        scripted.save(scripted_filename)
        print('Scripted model was saved to:', scripted_filename)

    amnet_runtime = AMNetDLR(model.double())
    img = amnet_runtime.load_img('../test_images/airplane.jpg')
    # img = np.array(img)
    # preproc = amnet.test_transform(img).double()
    # preproc = torch.unsqueeze(preproc, 0) # fake batch
    preproc = amnet_runtime.preprocess(img)
    res = amnet_runtime.infer(preproc)
    output, outputs, all_heatmaps = amnet_runtime.postprocess(res)
    # print(output)
    # print(outputs)
    for i, heatmap in enumerate(all_heatmaps[0]):
        cv2.imwrite(f'heatmap_orig_{i}.jpg', heatmap)


if __name__ == '__main__':
    main()
