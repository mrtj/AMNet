import cv2
import numpy as np
import torch
from PIL import Image

class AMNetDLR:

    INPUT_IMG_SIZE = (224, 224)
    INPUT_IMG_MEAN = np.array([0.485, 0.456, 0.406])
    INPUT_IMG_STD = np.array([0.229, 0.224, 0.225])
    RESULT_SCALE = 2.0
    RESULT_MEAN = 0.754

    def __init__(self, model):
        self.model = model

    def load_img(self, image_path):
        img = Image.open(image_path)
        return img

    def preprocess(self, img: Image, interpolation=Image.Resampling.BILINEAR):
        print('img size before:', img.size)
        resized = img.resize(AMNetDLR.INPUT_IMG_SIZE, interpolation)
        resized = np.array(resized) / 255.0
        print('img size after:', resized.size)
        print('min/max before norm:', np.min(np.array(resized)), np.max(np.array(resized)))

        normalized = (resized - AMNetDLR.INPUT_IMG_MEAN) / AMNetDLR.INPUT_IMG_STD
        print('min/max after norm:', np.min(np.array(normalized)), np.max(np.array(normalized)))
        res = np.expand_dims(normalized.transpose(2, 0, 1), axis=0)
        print('resulting dimensione:', res.shape)
        return torch.tensor(res).double()

    def infer(self, inp):
        with torch.no_grad():
            res = self.model(inp)
        return res

    def postprocess(self, res):
        output, outputs, att_maps = res
        output = (outputs).sum(1)
        output = output / outputs.shape[1]
        output /= AMNetDLR.RESULT_SCALE
        output = output + AMNetDLR.RESULT_MEAN
        outputs = (outputs / (outputs.shape[1] * AMNetDLR.RESULT_SCALE)) + AMNetDLR.RESULT_MEAN / outputs.shape[1]

        num_images = len(att_maps)
        seq_len = outputs.shape[1]
        ares = int(np.sqrt(att_maps.shape[2]))

        all_heatmaps = []

        for b in range(num_images):
            heatmaps = []
            for s in range(seq_len):
                img_alpha = att_maps[b, s]
                img_alpha = img_alpha.reshape((ares, ares))

                # Normalize & convert to uint8
                Min = img_alpha.min()
                img_alpha -= Min
                Max = img_alpha.max()
                if (Max != 0):
                    img_alpha = img_alpha/Max

                img_alpha = img_alpha.numpy()
                img_alpha = img_alpha * 255
                img_alpha = img_alpha.astype(np.uint8)

                heat_map_img = cv2.applyColorMap(img_alpha, cv2.COLORMAP_JET)
                heatmaps.append(heat_map_img)

            all_heatmaps.append(heatmaps)

        return output, outputs, all_heatmaps


if __name__ == '__main__':
    # amnet = AMNetDLR('./data/lamem_ResNet50FC_lstm3_train_1/weights_37_scripted.pth')
    model_path = './data/lamem_ResNet50FC_lstm3_train_1/weights_37_scripted.pth'
    # model_path = './data/lamem_ResNet50FC_lstm3_train_1/weights_37_traced.pth'
    model = torch.jit.load(model_path).double()
    amnet = AMNetDLR(model)

    img = amnet.load_img('../test_images/cat.jpeg')
    preproc = amnet.preprocess(img)
    res = amnet.infer(preproc)
    output, outputs, all_heatmaps = amnet.postprocess(res)
    # print(output)
    # print(outputs)
    for i, heatmap in enumerate(all_heatmaps[0]):
        cv2.imwrite(f'heatmap_{i}.jpg', heatmap)





# Load model.
# /path/to/model is a directory containing the compiled model artifacts (.so, .params, .json)
# model = dlr.DLRModel('./lamem_ResNet50FC_lstm3', 'cpu', 0)

# # Prepare some input data.
# x = np.random.rand(1, 3, 224, 224)

# # Run inference.
# y = model.run(x)

# print(y)
